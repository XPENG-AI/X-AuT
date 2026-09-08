from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

_VALID_PADDING_SIDES = {"left", "right"}


def _resolve_padding_side(padding_side: str) -> str:
    resolved = str(padding_side or "right").strip().lower()
    if resolved not in _VALID_PADDING_SIDES:
        raise ValueError(f"Unsupported padding_side={padding_side!r}; expected one of {_VALID_PADDING_SIDES}")
    return resolved


def pad_token_sequences(
    sequences: list[list[int]],
    *,
    pad_value: int,
    device: torch.device,
    padding_side: str = "right",
) -> torch.Tensor:
    if not sequences:
        raise ValueError("Expected at least one token sequence")
    resolved_padding_side = _resolve_padding_side(padding_side)
    max_len = max(len(sequence) for sequence in sequences)
    padded = torch.full((len(sequences), max_len), int(pad_value), dtype=torch.long, device=device)
    for row_index, sequence in enumerate(sequences):
        if not sequence:
            continue
        row = torch.tensor(sequence, dtype=torch.long, device=device)
        if resolved_padding_side == "left":
            padded[row_index, max_len - len(sequence) :] = row
        else:
            padded[row_index, : len(sequence)] = row
    return padded


def pad_attention_masks(
    sequences: list[list[int]],
    *,
    device: torch.device,
    padding_side: str = "right",
) -> torch.Tensor:
    if not sequences:
        raise ValueError("Expected at least one token sequence")
    resolved_padding_side = _resolve_padding_side(padding_side)
    max_len = max(len(sequence) for sequence in sequences)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for row_index, sequence in enumerate(sequences):
        if not sequence:
            continue
        if resolved_padding_side == "left":
            attention_mask[row_index, max_len - len(sequence) :] = 1
        else:
            attention_mask[row_index, : len(sequence)] = 1
    return attention_mask


@contextmanager
def temporary_tokenizer_padding_side(tokenizer: Any, *, padding_side: str) -> Iterator[None]:
    resolved_padding_side = _resolve_padding_side(padding_side)
    previous_padding_side = getattr(tokenizer, "padding_side", None)
    if previous_padding_side == resolved_padding_side:
        yield
        return
    setattr(tokenizer, "padding_side", resolved_padding_side)
    try:
        yield
    finally:
        if previous_padding_side is None:
            try:
                delattr(tokenizer, "padding_side")
            except AttributeError:
                pass
        else:
            setattr(tokenizer, "padding_side", previous_padding_side)


def normalize_generation_inputs_for_decoder_only(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError(
            "Decoder-only generation expects 2D input_ids and attention_mask, "
            f"got shapes {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "input_ids and attention_mask must share the same shape for generation, "
            f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
        )
    normalized_input_ids = torch.full_like(input_ids, int(pad_token_id))
    normalized_attention_mask = torch.zeros_like(attention_mask)
    bool_attention_mask = attention_mask.to(dtype=torch.bool)
    for row_index in range(input_ids.size(0)):
        valid_tokens = input_ids[row_index][bool_attention_mask[row_index]]
        valid_length = int(valid_tokens.numel())
        if valid_length <= 0:
            raise ValueError(f"Generation prompt at row {row_index} is empty after applying attention_mask")
        normalized_input_ids[row_index, -valid_length:] = valid_tokens
        normalized_attention_mask[row_index, -valid_length:] = 1
    return normalized_input_ids, normalized_attention_mask
