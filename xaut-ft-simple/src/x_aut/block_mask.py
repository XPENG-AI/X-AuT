
"""Block-B attention mask for Qwen3-ASR audio encoder distillation.

Applies a "hard block" attention geometry to the audio tower's transformer layers:
within each block of B encoder frames, attention is bidirectional; across blocks,
it is purely causal (past blocks visible, future blocks masked).

This module follows the same monkey-patch pattern as ``causal_patch.py``: it
replaces ``audio_tower.forward`` with a replica of the official conv frontend +
transformer stack, differing only in the per-layer attention mask.

Reference
---------
stream_encoder/experiments/20260805_test_encoder_arch/masks.py  (MaskSpec / build_layer_masks)
stream_encoder/experiments/20260805_test_encoder_arch/unified_encoder.py  (make_masked_forward)
X-AuT/src/x_aut/causal_patch.py  (monkey-patch pattern)
"""
from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from qwen_asr.core.transformers_backend import modeling_qwen3_asr as M

_LOGGER = logging.getLogger(__name__)





def _zero_scalar(tensor_like: torch.Tensor) -> torch.Tensor:
    return tensor_like.new_zeros(())






@dataclass(frozen=True)
class BlockMaskSpec:
    """Hard-block attention geometry.

    Parameters
    ----------
    block : int
        Block size B in encoder frames (1 frame ≈ 76.92 ms).
    left : int or None
        Left window size W in encoder frames.  ``None`` means no left truncation
        (see all past blocks).  A finite value caps the visible history.
    region : str
        ``"native"`` — respect the official cu_seqlens attention windows.
        ``"global"`` — treat the entire sequence as one window.
    """

    block: int
    left: Optional[int] = None
    region: str = "native"

    def __post_init__(self) -> None:
        if self.block < 1:
            raise ValueError(f"block must be >= 1, got {self.block}")
        if self.region not in {"native", "global"}:
            raise ValueError(f"region must be 'native' or 'global', got {self.region!r}")

    def label(self) -> str:
        left = "inf" if self.left is None else str(self.left)
        return f"block_B{self.block}_left{left}_{self.region}"






def _allow_matrix_block(
    length: int,
    B: int,
    left: Optional[int] = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean visibility matrix (L, L) for a single region with hard blocks.

    Frame *i* (query) can see frame *j* (key) iff *j* belongs to a block ≤ the
    block containing *i* AND *j* is within the left window (if set).

    Returns ``True`` where key is visible to query.
    """
    i = torch.arange(length, device=device).unsqueeze(1)
    j = torch.arange(length, device=device).unsqueeze(0)
    blk_i = i // B

    j_max = (blk_i + 1) * B - 1
    allow = j <= j_max
    if left is not None:
        allow = allow & (j >= i - left + 1)
    return allow


def _region_bounds(cu_seqlens: torch.Tensor, region: str) -> List[tuple[int, int]]:
    """Split *cu_seqlens* into attention regions."""
    if region == "global":
        return [(0, int(cu_seqlens[-1].item()))]

    cu = cu_seqlens.tolist()
    return [(int(cu[k - 1]), int(cu[k])) for k in range(1, len(cu))]


def _build_block_layer_mask(
    spec: BlockMaskSpec,
    cu_seqlens: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a single additive mask (1, 1, T, T) for one layer.

    All layers share the same block geometry (unlike the "causal" kind which
    supports per-layer lookahead schedules), so the result can be reused across
    every transformer layer.
    """
    total = int(cu_seqlens[-1].item())
    neg = torch.finfo(dtype).min
    mask = torch.full((total, total), neg, device=device, dtype=dtype)
    for s, e in _region_bounds(cu_seqlens, spec.region):
        length = e - s
        allow = _allow_matrix_block(length, spec.block, spec.left, device=device)
        mask[s:e, s:e] = torch.where(
            allow,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), neg, device=device, dtype=dtype),
        )
    return mask[None, None, :, :]






def _conv_frontend(
    audio_tower: nn.Module,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    align_80ms: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official Qwen3-ASR conv frontend.

    Returns ``(hidden_states, cu_seqlens)`` where *hidden_states* has shape
    ``(T_enc, d_model)`` and *cu_seqlens* is an int32 tensor of region boundaries.

    With ``align_80ms=True`` the 13 Hz padding frames are trimmed to the exact
    12.5 Hz grid (``floor(mel/8)`` frames, one per 80 ms) and the attention
    windows are rebuilt on that grid (8s = ``n_window_infer//8`` = 100 frames,
    not the official 13 Hz ``13*8 = 104`` which would leak an extra 320 ms of
    future context per window).
    """
    aftercnn_lens = M._get_feat_extract_output_lengths(feature_lens)
    n_win2 = audio_tower.n_window * 2
    chunk_num = torch.ceil(feature_lens / n_win2).long()
    chunk_lengths = torch.tensor(
        [n_win2] * int(chunk_num.sum()),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % n_win2
    chunk_lengths[chunk_lengths == 0] = n_win2

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = M._get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [torch.ones(int(length), dtype=torch.bool, device=padded_feature.device)
         for length in feature_lens_after_cnn],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)
    padded_embeds = []
    for chunk in padded_feature.split(audio_tower.conv_chunksize, dim=0):
        e = F.gelu(audio_tower.conv2d1(chunk))
        e = F.gelu(audio_tower.conv2d2(e))
        e = F.gelu(audio_tower.conv2d3(e))
        padded_embeds.append(e)
    padded_embed = torch.cat(padded_embeds, dim=0)
    b, c, f, t = padded_embed.size()
    padded_embed = audio_tower.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )
    positional_embedding = (
        audio_tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    hidden_states = padded_embed[padded_mask_after_cnn]

    if align_80ms:







        target_lens = (feature_lens // 8).long()
        keep = torch.zeros(int(aftercnn_lens.sum()), dtype=torch.bool, device=hidden_states.device)
        offset = 0
        for after_len, target_len in zip(aftercnn_lens.tolist(), target_lens.tolist()):
            keep[offset : offset + target_len] = True
            offset += after_len
        hidden_states = hidden_states[keep]
        total_frames = int(target_lens.sum())
        window_aftercnn = audio_tower.n_window_infer // 8
        cu_chunk_lens = [0]
        cu_chunk_lens += [window_aftercnn] * (total_frames // window_aftercnn)
        remainder = total_frames % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=hidden_states.device).cumsum(
            -1, dtype=torch.int32
        )
        return hidden_states, cu_seqlens

    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (
        audio_tower.n_window_infer // n_win2
    )
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (int(cnn_len) // window_aftercnn)
        remainder = int(cnn_len) % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=hidden_states.device).cumsum(
        -1, dtype=torch.int32
    )
    return hidden_states, cu_seqlens


def _head_project(audio_tower: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Bridge projection: ln_post → proj1 → act → proj2."""
    hidden_states = audio_tower.ln_post(hidden_states)
    hidden_states = audio_tower.proj1(hidden_states)
    hidden_states = audio_tower.act(hidden_states)
    hidden_states = audio_tower.proj2(hidden_states)
    return hidden_states






def make_block_forward(audio_tower: nn.Module, spec: BlockMaskSpec, align_80ms: bool = False):
    """Build a replacement ``forward`` for ``Qwen3ASRAudioEncoder`` that applies
    hard-block attention masks.

    The returned function replicates the official conv frontend exactly; the only
    difference is that the per-region bidirectional attention mask is replaced by
    the hard-block geometry defined in *spec*.  With ``align_80ms=True`` the
    frames are first trimmed to the 12.5 Hz grid (80 ms/frame).
    """

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None):
        hidden_states, cu_seqlens = _conv_frontend(
            self, input_features, feature_lens, align_80ms=align_80ms
        )


        block_mask = _build_block_layer_mask(
            spec,
            cu_seqlens,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        masks = [block_mask] * len(self.layers)

        for layer_idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
                attention_mask=masks[layer_idx],
            )
            hidden_states = layer_outputs[0]

        hidden_states = _head_project(self, hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    return forward






def apply_block_mask_patch(
    audio_tower: nn.Module,
    *,
    B: int,
    left: Optional[int] = None,
    align_80ms: bool = False,
) -> BlockMaskSpec:
    """Monkey-patch ``audio_tower.forward`` to use hard-block attention with block size *B*.

    Parameters
    ----------
    audio_tower : nn.Module
        The ``Qwen3ASRAudioEncoder`` module (``model.thinker.audio_tower``).
    B : int
        Block size in encoder frames.  B=2 gives ~38 ms latency; B=8 gives
        ~264 ms latency with near-zero quality loss.
    left : int or None
        Left window size.  ``None`` (default) means no truncation — every block
        can see all past blocks.
    align_80ms : bool
        Trim the 13 Hz padding frames to the exact 12.5 Hz grid (one encoder
        frame per 80 ms) before building the block geometry.  The block size B
        then refers to 80 ms frames (B=2 -> 160 ms blocks).

    Returns
    -------
    BlockMaskSpec
        The resolved spec (for logging / inspection).
    """
    spec = BlockMaskSpec(block=B, left=left, region="native")



    audio_tower.config._attn_implementation = "eager"
    audio_tower.forward = types.MethodType(
        make_block_forward(audio_tower, spec, align_80ms=align_80ms), audio_tower
    )

    num_layers = len(audio_tower.layers)
    _LOGGER.info(
        "Applied block-B attention mask | B=%d left=%s layers=%d region=%s align_80ms=%s",
        B,
        "inf" if left is None else str(left),
        num_layers,
        spec.region,
        align_80ms,
    )
    return spec
