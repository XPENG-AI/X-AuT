from __future__ import annotations

import datetime
import json
import logging
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import yaml


def deep_get(mapping: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    current: Any = mapping
    for key in str(path).split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def save_config(config: dict[str, Any], path: str | os.PathLike[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def save_json(payload: Any, path: str | os.PathLike[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | os.PathLike[str], rows: Iterator[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    distributed: bool
    device: torch.device


@dataclass
class _DataObservationInterval:
    step_count: int = 0
    sample_count: int = 0
    microbatch_count: int = 0
    microbatch_unique_dataset_sum: float = 0.0
    microbatch_dominant_ratio_sum: float = 0.0
    microbatch_padding_ratio_sum: float = 0.0
    microbatch_max_duration_sum: float = 0.0
    step_unique_dataset_sum: float = 0.0
    step_dominant_ratio_sum: float = 0.0
    step_padding_ratio_sum: float = 0.0
    step_cost_sum: float = 0.0
    step_cost_max: float = 0.0
    step_max_duration_sum: float = 0.0
    replacement_count: int = 0


class DataObservationTracker:
    def __init__(self) -> None:
        self._interval = _DataObservationInterval()
        self._reset_pending_step()

    def _reset_pending_step(self) -> None:
        self._pending_sample_count = 0
        self._pending_dataset_counts: dict[str, int] = {}
        self._pending_true_feature_frames = 0
        self._pending_padded_feature_frames = 0
        self._pending_step_cost = 0.0
        self._pending_step_max_duration = 0.0

    def observe_micro_batch(self, batch: dict[str, Any]) -> None:
        dataset_keys = [str(value or "") for value in batch.get("dataset_keys", [])]
        if not dataset_keys:
            return
        feature_lengths = [max(int(value), 0) for value in batch.get("feature_lengths", [])]
        audio_durations = [max(float(value), 0.0) for value in batch.get("audio_durations_seconds", [])]
        replacement_count = sum(int(value) for value in batch.get("replacement_flags", []))
        dataset_counts: dict[str, int] = {}
        for dataset_key in dataset_keys:
            dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + 1
            self._pending_dataset_counts[dataset_key] = self._pending_dataset_counts.get(dataset_key, 0) + 1
        batch_sample_count = len(dataset_keys)
        max_feature_length = max(feature_lengths) if feature_lengths else 0
        total_feature_frames = sum(feature_lengths)
        padded_feature_frames = max_feature_length * len(feature_lengths)
        padding_ratio = 0.0
        if padded_feature_frames > 0:
            padding_ratio = 1.0 - (total_feature_frames / float(padded_feature_frames))
        dominant_ratio = max(dataset_counts.values()) / float(batch_sample_count)
        max_duration = max(audio_durations) if audio_durations else 0.0
        batch_cost = float(max_feature_length * batch_sample_count)

        self._interval.sample_count += batch_sample_count
        self._interval.microbatch_count += 1
        self._interval.microbatch_unique_dataset_sum += float(len(dataset_counts))
        self._interval.microbatch_dominant_ratio_sum += dominant_ratio
        self._interval.microbatch_padding_ratio_sum += padding_ratio
        self._interval.microbatch_max_duration_sum += max_duration
        self._interval.replacement_count += replacement_count

        self._pending_sample_count += batch_sample_count
        self._pending_true_feature_frames += total_feature_frames
        self._pending_padded_feature_frames += padded_feature_frames
        self._pending_step_cost += batch_cost
        self._pending_step_max_duration = max(self._pending_step_max_duration, max_duration)

    def finish_global_step(self) -> None:
        if self._pending_sample_count <= 0:
            return
        dominant_ratio = max(self._pending_dataset_counts.values()) / float(self._pending_sample_count)
        padding_ratio = 0.0
        if self._pending_padded_feature_frames > 0:
            padding_ratio = 1.0 - (self._pending_true_feature_frames / float(self._pending_padded_feature_frames))
        self._interval.step_count += 1
        self._interval.step_unique_dataset_sum += float(len(self._pending_dataset_counts))
        self._interval.step_dominant_ratio_sum += dominant_ratio
        self._interval.step_padding_ratio_sum += padding_ratio
        self._interval.step_cost_sum += self._pending_step_cost
        self._interval.step_cost_max = max(self._interval.step_cost_max, self._pending_step_cost)
        self._interval.step_max_duration_sum += self._pending_step_max_duration
        self._reset_pending_step()

    def flush(self, *, dist_ctx: DistContext | None) -> dict[str, float] | None:
        if self._interval.step_count <= 0 or self._interval.microbatch_count <= 0 or self._interval.sample_count <= 0:
            self._interval = _DataObservationInterval()
            self._reset_pending_step()
            return None

        device = dist_ctx.device if dist_ctx is not None else torch.device("cpu")
        interval_tensor = torch.tensor(
            [
                float(self._interval.step_count),
                float(self._interval.sample_count),
                float(self._interval.microbatch_count),
                self._interval.microbatch_unique_dataset_sum,
                self._interval.microbatch_dominant_ratio_sum,
                self._interval.microbatch_padding_ratio_sum,
                self._interval.microbatch_max_duration_sum,
                self._interval.step_unique_dataset_sum,
                self._interval.step_dominant_ratio_sum,
                self._interval.step_padding_ratio_sum,
                self._interval.step_cost_sum,
                self._interval.step_max_duration_sum,
                float(self._interval.replacement_count),
            ],
            dtype=torch.float64,
            device=device,
        )
        step_cost_max_tensor = torch.tensor([self._interval.step_cost_max], dtype=torch.float64, device=device)
        local_rank_cost = torch.tensor(
            [
                self._interval.step_cost_sum / float(max(self._interval.step_count, 1)),
                self._interval.step_cost_max,
            ],
            dtype=torch.float64,
            device=device,
        )

        gathered_rank_costs: list[torch.Tensor] = [local_rank_cost]
        if dist_ctx is not None and dist_ctx.distributed and dist.is_initialized():
            dist.all_reduce(interval_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_cost_max_tensor, op=dist.ReduceOp.MAX)
            gathered_rank_costs = [torch.zeros_like(local_rank_cost) for _ in range(dist_ctx.world_size)]
            dist.all_gather(gathered_rank_costs, local_rank_cost)

        total_step_count = max(float(interval_tensor[0].item()), 1.0)
        total_sample_count = max(float(interval_tensor[1].item()), 1.0)
        total_microbatch_count = max(float(interval_tensor[2].item()), 1.0)
        rank_cost_tensor = torch.stack(gathered_rank_costs)
        rank_cost_mean = float(rank_cost_tensor[:, 0].mean().item())
        rank_cost_std = float(rank_cost_tensor[:, 0].std(unbiased=False).item())
        rank_cost_max = float(rank_cost_tensor[:, 1].max().item())

        metrics = {
            "data/microbatch_unique_dataset_mean": float(interval_tensor[3].item() / total_microbatch_count),
            "data/microbatch_dominant_ratio_mean": float(interval_tensor[4].item() / total_microbatch_count),
            "data/microbatch_padding_ratio_mean": float(interval_tensor[5].item() / total_microbatch_count),
            "data/microbatch_max_duration_mean": float(interval_tensor[6].item() / total_microbatch_count),
            "data/global_step_unique_dataset_mean": float(interval_tensor[7].item() / total_step_count),
            "data/global_step_dominant_ratio_mean": float(interval_tensor[8].item() / total_step_count),
            "data/global_step_padding_ratio_mean": float(interval_tensor[9].item() / total_step_count),
            "data/global_step_cost_mean": float(interval_tensor[10].item() / total_step_count),
            "data/global_step_cost_max": float(step_cost_max_tensor.item()),
            "data/global_step_max_duration_mean": float(interval_tensor[11].item() / total_step_count),
            "data/replacement_rate": float(interval_tensor[12].item() / total_sample_count),
            "data/rank_cost_mean": rank_cost_mean,
            "data/rank_cost_std": rank_cost_std,
            "data/rank_cost_max": rank_cost_max,
            "data/interval_steps": total_step_count,
        }

        self._interval = _DataObservationInterval()
        self._reset_pending_step()
        return metrics


def init_distributed(device_type: str = "cuda") -> DistContext:
    rank = int(os.environ.get("RANK", "0") or 0)
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")) or 0)
    distributed = world_size > 1
    use_cuda = device_type.startswith("cuda")
    if distributed and not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        timeout_minutes = max(int(os.environ.get("X_AUT_DDP_TIMEOUT_MINUTES", "180") or 180), 1)
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=timeout_minutes))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=distributed,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier(dist_ctx: DistContext | None) -> None:
    if dist_ctx is not None and dist_ctx.distributed and dist.is_initialized():
        if dist_ctx.device.type == "cuda":
            dist.barrier(device_ids=[dist_ctx.local_rank])
            return
        dist.barrier()


def is_main_process(dist_ctx: DistContext | None) -> bool:
    return dist_ctx is None or dist_ctx.rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def seed_everything(seed: int, *, rank: int = 0, use_cuda: bool = False) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(effective_seed)


def resolve_amp_dtype(raw_value: Any, *, device: torch.device) -> torch.dtype | None:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"", "false", "none", "off", "0"}:
        return None
    if device.type != "cuda":
        return None
    if normalized in {"true", "auto", "bf16", "bfloat16"}:
        is_bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        return torch.bfloat16 if is_bf16_supported else torch.float16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return None
    raise ValueError(f"Unsupported AMP dtype: {raw_value!r}")


def resolve_model_dtype(raw_value: Any, *, device: torch.device) -> torch.dtype:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"", "auto"}:
        if device.type == "cuda":
            is_bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            return torch.bfloat16 if is_bf16_supported else torch.float16
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16 if device.type == "cuda" else torch.float32
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported model.load_dtype: {raw_value!r}")


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def count_parameters(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def setup_logger(output_dir: str | os.PathLike[str], dist_ctx: DistContext | None) -> logging.Logger:
    logger = logging.getLogger("x_aut")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    output_path = ensure_dir(output_dir)
    rank = 0 if dist_ctx is None else dist_ctx.rank
    formatter = logging.Formatter(
        fmt="%(asctime)s | rank=%(rank)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = rank
            return True

    if is_main_process(dist_ctx):
        file_handler = logging.FileHandler(output_path / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RankFilter())
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RankFilter())
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def format_parameter_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)
