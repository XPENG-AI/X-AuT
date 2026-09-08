from __future__ import annotations

import logging
import types
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from x_aut.utils import deep_get

_LOGGER = logging.getLogger(__name__)

_CAUSAL_MODES = ("front", "even", "back")


def make_causal_forward(
    audio_tower: nn.Module,
    lookahead_schedule: Sequence[int],
):
    """Build a replacement ``forward`` for ``Qwen3ASRAudioEncoder``.

    The logic mirrors the official forward exactly (same 1s-chunk convolutions,
    same aftercnn chunking, same positional embedding, same projections); the
    only difference is the additive attention mask: within each chunk, frame
    ``i`` can attend to frames ``j <= i + lookahead[layer_idx]`` instead of the
    full bidirectional block.
    """

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None):
        from qwen_asr.core.transformers_backend import modeling_qwen3_asr as M

        aftercnn_lens = M._get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long, device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = M._get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(l, dtype=torch.bool, device=padded_feature.device) for l in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            e = F.gelu(self.conv2d1(chunk))
            e = F.gelu(self.conv2d2(e))
            e = F.gelu(self.conv2d3(e))
            padded_embeds.append(e)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0).to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]

        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)


        mask_cache = {
            la: _build_limited_causal_mask(hidden_states, cu_seqlens, la)
            for la in set(int(x) for x in lookahead_schedule)
        }
        for layer_idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
                attention_mask=mask_cache[int(lookahead_schedule[layer_idx])],
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    return forward


def build_total_lookahead_schedule(num_layers: int, total_lookahead: int, mode: str = "even") -> list[int]:
    """Distribute an end-to-end right-context budget (in encoder frames) across layers."""
    if total_lookahead < 0:
        raise ValueError("total_lookahead must be non-negative")
    if mode not in _CAUSAL_MODES:
        raise ValueError(f"mode must be one of: {', '.join(_CAUSAL_MODES)}")

    schedule = [0] * num_layers
    base, extra = divmod(total_lookahead, num_layers)
    if base:
        schedule = [base] * num_layers

    if extra == 0:
        return schedule
    if mode == "front":
        idxs = range(extra)
    elif mode == "back":
        idxs = range(num_layers - extra, num_layers)
    else:
        if extra == 1:
            idxs = [num_layers // 2]
        else:
            idxs = [round(i * (num_layers - 1) / (extra - 1)) for i in range(extra)]
    for idx in idxs:
        schedule[int(idx)] += 1
    return schedule


def _build_limited_causal_mask(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lookahead: Optional[int],
) -> torch.Tensor:
    """Build a (1, 1, total, total) additive mask. Within each chunk, frame ``i``
    sees ``j <= i + lookahead`` (``lookahead=None`` falls back to bidirectional)."""
    total = hidden_states.shape[0]
    dtype = hidden_states.dtype
    device = hidden_states.device
    neg = torch.finfo(dtype).min
    mask = torch.full((total, total), neg, device=device, dtype=dtype)
    for i in range(1, len(cu_seqlens)):
        s, e = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
        length = e - s
        if lookahead is None:
            block = torch.zeros(length, length, device=device, dtype=dtype)
        else:
            idx = torch.arange(length, device=device)
            allow = idx.unsqueeze(0) <= idx.unsqueeze(1) + lookahead
            block = torch.where(
                allow,
                torch.zeros((), device=device, dtype=dtype),
                torch.full((), neg, device=device, dtype=dtype),
            )
        mask[s:e, s:e] = block
    return mask[None, None, :, :]


def apply_causal_audio_tower_patch(
    audio_tower: nn.Module,
    *,
    total_lookahead_frames: int,
    schedule_mode: str = "even",
) -> list[int]:
    """Monkey-patch ``audio_tower.forward`` with a limited-causal attention mask.

    ``total_lookahead_frames`` is the end-to-end right-context budget in encoder
    frames (40ms each; 5 frames = 200ms), distributed across layers according to
    ``schedule_mode`` ("front" / "even" / "back"). The attention implementation
    is forced to eager because (a) flash_attention_2 ignores the additive 4D
    mask and (b) sdpa is buggy on the target hardware.
    """
    num_layers = len(audio_tower.layers)
    schedule = build_total_lookahead_schedule(num_layers, int(total_lookahead_frames), schedule_mode)
    audio_tower.config._attn_implementation = "eager"
    audio_tower.forward = types.MethodType(
        make_causal_forward(audio_tower, schedule), audio_tower
    )
    _LOGGER.info(
        "Applied causal audio tower patch | layers=%d total_lookahead_frames=%d mode=%s schedule=%s",
        num_layers,
        int(total_lookahead_frames),
        schedule_mode,
        schedule,
    )
    return schedule


def resolve_causal_lookahead_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve ``model.causal_lookahead`` from the run config.

    Returns ``None`` when disabled; otherwise a dict with
    ``total_lookahead_frames`` and ``schedule_mode``.
    """
    causal_cfg = deep_get(config, "model.causal_lookahead", None)
    if not isinstance(causal_cfg, dict) or not bool(causal_cfg.get("enabled", False)):
        return None
    total_lookahead_frames = int(causal_cfg.get("total_lookahead_frames", 5))
    if total_lookahead_frames < 0:
        raise ValueError(
            f"model.causal_lookahead.total_lookahead_frames must be >= 0, got {total_lookahead_frames}"
        )
    schedule_mode = str(causal_cfg.get("schedule_mode", "even") or "even")
    if schedule_mode not in _CAUSAL_MODES:
        raise ValueError(
            f"model.causal_lookahead.schedule_mode must be one of {_CAUSAL_MODES}, got {schedule_mode!r}"
        )
    return {
        "total_lookahead_frames": total_lookahead_frames,
        "schedule_mode": schedule_mode,
    }
