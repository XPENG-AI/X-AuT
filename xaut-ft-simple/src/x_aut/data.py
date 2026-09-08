from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from x_aut.audio_io import AudioResolver, AudioResolverConfig, load_audio_mono_16k
from x_aut.manifest_utils import build_target_text

if TYPE_CHECKING:
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

_LOGGER = logging.getLogger(__name__)


def _import_qwen_processor() -> type["Qwen3ASRProcessor"]:
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    return Qwen3ASRProcessor


def load_train_index(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"train_index must be a JSON object: {path}")
    return payload


DATASET_MIX_INDEX_FORMAT = "x-aut-dataset-mix-v1"
DEFAULT_CLASS_LABELED_TRAIN_MANIFEST = "train_with_class.jsonl"


def is_dataset_mix_index(payload: dict[str, Any]) -> bool:
    """判断 train_index 是否为「每个 dataset_key 一个目录」的加权混合新格式。"""
    return str(payload.get("format") or "").strip() == DATASET_MIX_INDEX_FORMAT


def _parse_class_selector(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"class selector must not be a boolean: {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        classes: set[int] = set()
        for part in text.replace("_", ",").split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if end < start:
                    raise ValueError(f"class range must be ascending: {value!r}")
                classes.update(range(start, end + 1))
                continue
            classes.add(int(token))
        if not classes:
            return None
        return tuple(sorted(classes))
    if isinstance(value, (list, tuple, set)):
        classes: set[int] = set()
        for item in value:
            parsed = _parse_class_selector(item)
            if parsed is None:
                continue
            classes.update(parsed)
        if not classes:
            return None
        return tuple(sorted(classes))
    try:
        return (int(value),)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported class selector: {value!r}") from exc


def _normalize_class_selector(value: Any) -> str | None:
    parsed = _parse_class_selector(value)
    if parsed is None:
        return None
    return "_".join(str(class_id) for class_id in parsed)


def _resolve_dataset_train_manifest(
    *,
    data_root: str,
    dataset_key: str,
    class_selector_value: Any,
) -> tuple[str, str, str | None]:
    class_selector = _normalize_class_selector(class_selector_value)
    manifest_name = DEFAULT_CLASS_LABELED_TRAIN_MANIFEST
    if class_selector is not None:
        manifest_name = f"train_class_{class_selector}.jsonl"
    train_path = str(Path(data_root) / dataset_key / manifest_name)
    return train_path, manifest_name, class_selector


def _sidecar_path(jsonl_path: str, suffix: str) -> Path:
    """为 jsonl 文件派生同目录的索引边路文件路径（train.jsonl -> train<suffix>）。"""
    p = Path(jsonl_path)
    if p.name.endswith(".jsonl"):
        return p.with_name(p.name[: -len(".jsonl")] + suffix)
    return p.with_name(p.name + suffix)


def _sidecar_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.meta.json")


def _source_signature(jsonl_path: str) -> dict[str, Any]:
    source_path = Path(jsonl_path).expanduser().resolve()
    stat_result = source_path.stat()
    return {
        "source_path": str(source_path),
        "source_size": int(stat_result.st_size),
        "source_mtime_ns": int(stat_result.st_mtime_ns),
    }


def _load_sidecar_metadata(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOGGER.warning("failed to read sidecar metadata %s: %s", meta_path, exc)
        return None
    if not isinstance(payload, dict):
        _LOGGER.warning("sidecar metadata must be an object: %s", meta_path)
        return None
    return payload


def _is_sidecar_current(*, jsonl_path: str, metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    current_signature = _source_signature(jsonl_path)
    return all(metadata.get(key) == value for key, value in current_signature.items())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temp_path.open("wb") as handle:
        np.save(handle, array)
    os.replace(temp_path, path)


def _read_cached_sidecar_array(*, jsonl_path: str, cache_path: Path) -> np.ndarray | None:
    metadata_path = _sidecar_metadata_path(cache_path)
    metadata = _load_sidecar_metadata(metadata_path)
    if not _is_sidecar_current(jsonl_path=jsonl_path, metadata=metadata):
        return None
    if not cache_path.exists():
        return None
    try:
        array = np.load(cache_path)
    except Exception as exc:
        _LOGGER.warning("failed to load cached sidecar %s: %s", cache_path, exc)
        return None
    if array.ndim != 1:
        _LOGGER.warning("cached sidecar must be 1-D: %s", cache_path)
        return None
    return array.astype(np.int64, copy=False)


def _acquire_sidecar_lock(lock_path: Path) -> tuple[int, str]:
    timeout_seconds = max(float(os.environ.get("X_AUT_SIDECAR_LOCK_TIMEOUT_SECONDS", "300") or 300.0), 1.0)
    heartbeat_interval = max(float(os.environ.get("X_AUT_SIDECAR_LOCK_HEARTBEAT_SECONDS", "30") or 30.0), 1.0)
    configured_stale_seconds = float(os.environ.get("X_AUT_SIDECAR_LOCK_STALE_SECONDS", str(timeout_seconds)) or timeout_seconds)
    stale_seconds = max(min(configured_stale_seconds, timeout_seconds), heartbeat_interval * 2.0)
    deadline = time.monotonic() + timeout_seconds
    last_lock_mtime_ns: int | None = None
    while True:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            token = f"{os.getpid()}:{time.time_ns()}"
            os.write(handle, f"{token}\n".encode("utf-8"))
            return handle, token
        except FileExistsError:
            try:
                stat_result = lock_path.stat()
                lock_age_seconds = max(time.time() - stat_result.st_mtime, 0.0)
                lock_mtime_ns = int(stat_result.st_mtime_ns)
            except OSError:
                lock_age_seconds = 0.0
                lock_mtime_ns = None
            if lock_mtime_ns is not None and lock_mtime_ns != last_lock_mtime_ns:
                last_lock_mtime_ns = lock_mtime_ns
                deadline = time.monotonic() + timeout_seconds
            if lock_age_seconds >= stale_seconds:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for sidecar build lock: {lock_path}")
            time.sleep(0.2)


def _release_sidecar_lock(lock_path: Path, handle: int | None, token: str | None) -> None:
    if handle is None:
        return
    try:
        os.close(handle)
    except OSError:
        pass
    if token is None:
        return
    try:
        current_token = lock_path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (FileNotFoundError, IndexError, OSError):
        return
    if current_token != token:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _build_or_load_sidecar_array(
    *,
    jsonl_path: str,
    suffix: str,
    builder,
) -> np.ndarray:
    cache_path = _sidecar_path(jsonl_path, suffix)
    cached = _read_cached_sidecar_array(jsonl_path=jsonl_path, cache_path=cache_path)
    if cached is not None:
        return cached
    metadata_path = _sidecar_metadata_path(cache_path)
    lock_path = cache_path.with_name(f"{cache_path.name}.lock")
    lock_handle: int | None = None
    lock_token: str | None = None
    heartbeat_stop: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    try:
        lock_handle, lock_token = _acquire_sidecar_lock(lock_path)
        heartbeat_stop = threading.Event()

        def _heartbeat() -> None:
            interval_seconds = max(float(os.environ.get("X_AUT_SIDECAR_LOCK_HEARTBEAT_SECONDS", "30") or 30.0), 1.0)
            while not heartbeat_stop.wait(interval_seconds):
                try:
                    current_token = lock_path.read_text(encoding="utf-8").splitlines()[0].strip()
                except (FileNotFoundError, IndexError, OSError):
                    return
                if current_token != lock_token:
                    return
                try:
                    os.utime(lock_path, None)
                except FileNotFoundError:
                    return

        heartbeat_thread = threading.Thread(target=_heartbeat, name="x-aut-sidecar-lock", daemon=True)
        heartbeat_thread.start()
        cached = _read_cached_sidecar_array(jsonl_path=jsonl_path, cache_path=cache_path)
        if cached is not None:
            return cached
        metadata: dict[str, Any] | None = None
        array: np.ndarray | None = None
        for attempt in range(3):
            metadata_before = _source_signature(jsonl_path)
            built_array = builder(jsonl_path).astype(np.int64, copy=False)
            metadata_after = _source_signature(jsonl_path)
            if metadata_before == metadata_after:
                metadata = metadata_after
                array = built_array
                break
            _LOGGER.warning(
                "source changed while building sidecar %s (attempt %d/3); retrying",
                cache_path,
                attempt + 1,
            )
        if metadata is None or array is None:
            raise RuntimeError(f"source changed repeatedly while building sidecar: {jsonl_path}")
        try:
            _write_npy_atomic(cache_path, array)
            _write_json_atomic(metadata_path, metadata)
        except Exception as exc:
            _LOGGER.warning("failed to write sidecar %s: %s", cache_path, exc)
        return array
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        _release_sidecar_lock(lock_path, lock_handle, lock_token)


def _resolve_record_length_proxy(record: dict[str, Any]) -> float:
    """长度代理优先级：duration 字段 -> token_len -> len(token_ids) -> 0。"""
    duration_seconds = _resolve_record_audio_duration_seconds(record)
    if duration_seconds is not None:
        return float(duration_seconds)
    token_len = record.get("token_len")
    if token_len is not None:
        try:
            value = int(token_len)
            if value > 0:
                return float(value)
        except (TypeError, ValueError):
            pass
    token_ids = record.get("token_ids")
    if isinstance(token_ids, (list, tuple)):
        return float(len(token_ids))
    return 0.0


def build_or_load_line_offsets(jsonl_path: str) -> np.ndarray:
    """返回 jsonl 每个非空行的字节偏移(np.int64)；优先读缓存边路，缺失则扫描构建并缓存。"""
    def _scan_offsets(path: str) -> np.ndarray:
        offsets: list[int] = []
        with open(path, "rb") as handle:
            position = handle.tell()
            line = handle.readline()
            while line:
                if line.strip():
                    offsets.append(position)
                position = handle.tell()
                line = handle.readline()
        return np.asarray(offsets, dtype=np.int64)

    return _build_or_load_sidecar_array(
        jsonl_path=jsonl_path,
        suffix=".offsets.npy",
        builder=_scan_offsets,
    )


def build_or_load_lengths(jsonl_path: str) -> np.ndarray:
    """返回 jsonl 每个非空行的长度代理(np.int64)；优先读缓存边路，缺失则扫描构建并缓存。"""
    def _scan_lengths(path: str) -> np.ndarray:
        lengths: list[int] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    lengths.append(0)
                    continue
                lengths.append(int(_resolve_record_length_proxy(payload)))
        return np.asarray(lengths, dtype=np.int64)

    return _build_or_load_sidecar_array(
        jsonl_path=jsonl_path,
        suffix=".lengths.npy",
        builder=_scan_lengths,
    )


def read_jsonl_line_at_offset(path: str, offset: int, *, handle=None) -> dict[str, Any]:
    """按字节偏移 seek 读取单行记录，行为与 load_jsonl_records 一致。"""
    if handle is None:
        with open(path, "rb") as local_handle:
            local_handle.seek(int(offset))
            raw = local_handle.readline()
    else:
        handle.seek(int(offset))
        raw = handle.readline()
    line = raw.decode("utf-8").strip()
    if not line:
        raise ValueError(f"empty JSONL line at offset {offset} in {path}")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError(f"JSONL record must be an object: {path}")
    payload["__source_path__"] = path
    payload["__source_offset__"] = int(offset)
    return payload


@dataclass(eq=False)
class DatasetSpec:
    dataset_key: str
    train_path: str
    count: int
    weight: float
    offsets: np.ndarray
    lengths: np.ndarray


def load_dataset_specs(payload: dict[str, Any]) -> list["DatasetSpec"]:
    """解析新格式 train_index，为每个 dataset_key 载入 offsets/lengths 索引。"""
    data_root = str(payload.get("data_root") or "").strip()
    if not data_root:
        raise ValueError("dataset-mix train_index is missing data_root")
    default_class_selector_value = payload["class"] if "class" in payload else None
    datasets = payload.get("datasets") or []
    if not datasets:
        raise ValueError("dataset-mix train_index has empty datasets")
    specs: list[DatasetSpec] = []
    for index, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{index}] must be an object")
        dataset_key = str(item.get("dataset_key") or "").strip()
        if not dataset_key:
            raise ValueError(f"datasets[{index}] is missing dataset_key")
        weight = float(item.get("weight", 1.0) or 0.0)
        if weight <= 0:
            continue
        class_selector_value = item["class"] if "class" in item else default_class_selector_value
        train_path, manifest_name, resolved_class_selector = _resolve_dataset_train_manifest(
            data_root=data_root,
            dataset_key=dataset_key,
            class_selector_value=class_selector_value,
        )
        if not Path(train_path).exists():
            if resolved_class_selector is not None:
                _LOGGER.warning(
                    "dataset %s is missing requested manifest %s (class=%s); skipping",
                    dataset_key,
                    manifest_name,
                    resolved_class_selector,
                )
                continue
            raise FileNotFoundError(f"{manifest_name} not found for dataset {dataset_key!r}: {train_path}")
        offsets: np.ndarray | None = None
        lengths: np.ndarray | None = None
        for attempt in range(3):
            signature_before = _source_signature(train_path)
            offsets = build_or_load_line_offsets(train_path)
            lengths = build_or_load_lengths(train_path)
            signature_after = _source_signature(train_path)
            if signature_before == signature_after:
                break
            _LOGGER.warning(
                "train manifest changed while loading sidecars for %s (attempt %d/3); retrying",
                train_path,
                attempt + 1,
            )
        if offsets is None or lengths is None:
            raise RuntimeError(f"failed to load sidecars for dataset {dataset_key!r}: {train_path}")
        if signature_before != signature_after:
            raise RuntimeError(f"train manifest changed repeatedly while loading sidecars: {train_path}")
        count = int(len(offsets))
        if count == 0:
            continue
        if len(lengths) != count:
            aligned = min(count, int(len(lengths)))
            _LOGGER.warning(
                "offsets/lengths mismatch for %s: offsets=%d lengths=%d, aligning to %d",
                train_path,
                count,
                int(len(lengths)),
                aligned,
            )
            offsets = offsets[:aligned]
            lengths = lengths[:aligned]
            count = aligned
        specs.append(
            DatasetSpec(
                dataset_key=dataset_key,
                train_path=train_path,
                count=count,
                weight=weight,
                offsets=offsets,
                lengths=lengths,
            )
        )
    if not specs:
        raise ValueError("dataset-mix train_index produced no usable datasets")
    return specs


def _compute_dataset_targets(specs: list["DatasetSpec"]) -> list[int]:
    """大小系数语义：target_i = round(weight_i x count_i)，下限 1。"""
    targets: list[int] = []
    for spec in specs:
        target = int(round(float(spec.weight) * float(spec.count)))
        targets.append(max(target, 1))
    return targets


def _compute_dataset_mix_budget(
    targets: list[int],
    *,
    per_device_batch_size: int,
    world_size: int,
    grad_accum_steps: int,
    max_train_samples: int | None = None,
) -> tuple[int, list[int], int]:
    """计算加权混合的同步预算。

    各 rank 从每个 dataset 取 prt_i = target_i // world_size 条（与 rank 无关，保证各 rank 等量），
    故所有 rank 天然产出相同 step 数，不依赖跨 rank 通信即可同步。
    返回 (steps_per_epoch, per_rank_targets, usable_per_rank)。
    """
    world = max(int(world_size), 1)
    per_rank_targets = [int(target) // world for target in targets]
    if max_train_samples is not None:
        cap_per_rank = max(int(max_train_samples), 0) // world
        total_per_rank = sum(per_rank_targets)
        if total_per_rank > cap_per_rank and total_per_rank > 0:
            scale = cap_per_rank / float(total_per_rank)
            per_rank_targets = [max(int(value * scale), 0) for value in per_rank_targets]
    per_rank_records = sum(per_rank_targets)
    micro_batch = max(int(per_device_batch_size), 1) * max(int(grad_accum_steps), 1)
    steps_per_epoch = per_rank_records // micro_batch
    if steps_per_epoch <= 0:
        raise ValueError(
            "dataset-mix budget does not contain enough records for one synchronized optimizer step: "
            f"per_rank_records={per_rank_records}, micro_batch={micro_batch}"
        )
    usable_per_rank = steps_per_epoch * micro_batch
    if per_rank_records != usable_per_rank:
        per_rank_targets = _trim_dataset_targets_to_total(per_rank_targets, total=usable_per_rank)
    return steps_per_epoch, per_rank_targets, usable_per_rank


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return int(digest.hexdigest()[:16], 16) % (2**63 - 1)


def _trim_dataset_targets_to_total(targets: list[int], *, total: int) -> list[int]:
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    current_total = sum(int(value) for value in targets)
    if current_total <= total:
        return [int(value) for value in targets]
    if total == 0:
        return [0 for _ in targets]
    scale = total / float(current_total)
    trimmed = [min(int(math.floor(value * scale)), int(value)) for value in targets]
    remainder = total - sum(trimmed)
    if remainder > 0:
        ranked = sorted(
            range(len(targets)),
            key=lambda index: ((targets[index] * scale) - trimmed[index], targets[index], -index),
            reverse=True,
        )
        for index in ranked:
            if remainder <= 0:
                break
            if trimmed[index] >= int(targets[index]):
                continue
            trimmed[index] += 1
            remainder -= 1
    if sum(trimmed) != total:
        raise RuntimeError(
            "failed to trim dataset targets to requested total: "
            f"current_total={current_total}, total={total}, trimmed_total={sum(trimmed)}"
        )
    return trimmed


def _allocate_worker_batch_targets(*, usable_total: int, batch_size: int, num_workers: int) -> list[int]:
    if usable_total < 0:
        raise ValueError(f"usable_total must be >= 0, got {usable_total}")
    worker_count = max(int(num_workers), 1)
    micro_batch = max(int(batch_size), 1)
    total_batches = usable_total // micro_batch
    base_batches = total_batches // worker_count
    remainder = total_batches % worker_count
    return [(base_batches + (1 if worker_index < remainder else 0)) * micro_batch for worker_index in range(worker_count)]


def _allocate_dataset_targets_to_workers(
    per_rank_targets: list[int],
    *,
    worker_targets: list[int],
) -> list[list[int]]:
    allocations = [[0 for _ in worker_targets] for _ in per_rank_targets]
    worker_remaining = [int(value) for value in worker_targets]
    dataset_order = sorted(range(len(per_rank_targets)), key=lambda index: (int(per_rank_targets[index]), -index), reverse=True)
    total_remaining = sum(worker_remaining)
    for dataset_index in dataset_order:
        target = int(per_rank_targets[dataset_index])
        if target <= 0:
            continue
        if total_remaining < target:
            raise RuntimeError(
                "insufficient worker capacity while allocating dataset targets: "
                f"dataset_index={dataset_index}, target={target}, total_remaining={total_remaining}"
            )
        raw_allocations = [
            (target * remaining / float(total_remaining)) if total_remaining > 0 else 0.0
            for remaining in worker_remaining
        ]
        row = [min(int(math.floor(value)), worker_remaining[worker_index]) for worker_index, value in enumerate(raw_allocations)]
        assigned = sum(row)
        remainder = target - assigned
        ranked_workers = sorted(
            range(len(worker_remaining)),
            key=lambda worker_index: (
                raw_allocations[worker_index] - row[worker_index],
                worker_remaining[worker_index] - row[worker_index],
                -worker_index,
            ),
            reverse=True,
        )
        while remainder > 0:
            progressed = False
            for worker_index in ranked_workers:
                capacity = worker_remaining[worker_index] - row[worker_index]
                if capacity <= 0:
                    continue
                row[worker_index] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
            if not progressed:
                raise RuntimeError(
                    "failed to distribute dataset targets to workers without overflow: "
                    f"dataset_index={dataset_index}, target={target}, worker_remaining={worker_remaining}"
                )
        allocations[dataset_index] = row
        for worker_index, value in enumerate(row):
            worker_remaining[worker_index] -= value
        total_remaining -= target
    if any(value != 0 for value in worker_remaining):
        raise RuntimeError(f"worker target allocation left unmatched capacity: {worker_remaining}")
    return allocations


DEFAULT_NUM_LENGTH_BINS = 9
DEFAULT_LENGTH_BIN_CAP_SECONDS = 30.0


def _build_length_bin_edges(cap_seconds: float, num_bins: int) -> tuple[int, ...]:
    """按时长上限 cap 把 [0, cap] 等分为 num_bins 个区间，返回 num_bins-1 个升序整数上界(秒)。

    - length_proxy 与 cap 同为"秒"，故可直接用 cap 等分得到分桶边界；
    - cap<=0 或非有限(未设 duration guard)时回退到 DEFAULT_LENGTH_BIN_CAP_SECONDS；
    - 边界四舍五入到整数秒并保证严格单调递增(相等时 +1 去重)，避免空桶错位。
    """
    bins = max(int(num_bins), 1)
    cap = float(cap_seconds)
    if not math.isfinite(cap) or cap <= 0:
        cap = DEFAULT_LENGTH_BIN_CAP_SECONDS
    if bins <= 1:
        return ()
    edges: list[int] = []
    for index in range(1, bins):
        edge = int(round(cap * index / bins))
        if edges and edge <= edges[-1]:
            edge = edges[-1] + 1
        edges.append(edge)
    return tuple(edges)


def _resolve_length_bin(length_proxy: int, edges: tuple[int, ...]) -> int:
    """按升序上界 edges 把长度代理映射到 bin id(0..len(edges))；bin 总数 = len(edges)+1。"""
    length_value = max(int(length_proxy), 0)
    for bin_id, edge in enumerate(edges):
        if length_value <= edge:
            return bin_id
    return len(edges)


def load_jsonl_records(
    path: str,
    *,
    skip_samples: int = 0,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    if skip_samples < 0:
        raise ValueError(f"skip_samples must be >= 0, got {skip_samples}")
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        skipped = 0
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if skipped < skip_samples:
                skipped += 1
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object: {path}")
            payload["__source_path__"] = path
            payload["__source_line__"] = line_number
            records.append(payload)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def iter_jsonl_record_groups(
    path: str,
    *,
    group_size: int,
    skip_samples: int = 0,
    max_samples: int | None = None,
):
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if skip_samples < 0:
        raise ValueError(f"skip_samples must be >= 0, got {skip_samples}")
    records: list[dict[str, Any]] = []
    emitted = 0
    with open(path, "r", encoding="utf-8") as handle:
        skipped = 0
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if skipped < skip_samples:
                skipped += 1
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object: {path}")
            payload["__source_path__"] = path
            payload["__source_line__"] = line_number
            records.append(payload)
            emitted += 1
            if len(records) >= group_size:
                yield records
                records = []
            if max_samples is not None and emitted >= max_samples:
                break
    if records:
        yield records


def _record_source_string(record: dict[str, Any]) -> str:
    source_path = str(record.get("__source_path__") or "").strip()
    source_line_raw = record.get("__source_line__")
    try:
        source_line = int(source_line_raw) if source_line_raw is not None else None
    except (TypeError, ValueError):
        source_line = None
    if source_path and source_line is not None and source_line > 0:
        return f"{source_path}:{source_line}"
    if source_path:
        return source_path
    if source_line is not None and source_line > 0:
        return f"line={source_line}"
    return ""


def _resolve_target_text(record: dict[str, Any], *, target_format: str) -> str:
    target_text = str(record.get("target_text") or "").strip()
    if target_text:
        return target_text
    lang = str(record.get("lang") or "").strip().lower()
    source_text = str(record.get("text_norm") or record.get("text") or record.get("gold_text") or "").strip()
    return build_target_text(source_text, lang=lang, target_format=target_format)


@dataclass
class DatasetRuntimeConfig:
    qwen_source: str
    target_format: str
    cache_dir: str
    oss_config_path: str = ""
    oss_workers: int = 4
    prefetch_workers: int = 1
    local_audio_roots: tuple[str, ...] = ()
    local_files_only: bool = True
    fix_mistral_regex: bool = True
    add_eos: bool = True
    split_name: str = "train"
    rank: int = 0
    cache_mode: str = "persistent"
    prefetch_mode: str = "none"
    prefetch_groups_ahead: int = 1
    prefetch_group_records: int = 0
    dataloader_timeout_seconds: float = 0.0
    audio_resolve_timeout_seconds: float = 0.0
    max_audio_duration_seconds: float = 0.0
    max_failed_record_fraction: float = 1.0
    num_length_bins: int = DEFAULT_NUM_LENGTH_BINS


@dataclass(frozen=True)
class TrainShard:
    path: str
    count: int
    selection_indices_path: str = ""


@dataclass(frozen=True)
class TrainShardSlice:
    path: str
    start_record: int
    count: int
    selection_indices_path: str = ""


@dataclass(frozen=True)
class PlannedMixedSample:
    stream_index: int
    global_index: int
    dataset_key: str
    length_proxy: int
    length_bin: int


def _record_debug_string(record: dict[str, Any]) -> str:
    parts = [
        f"utt_id={record.get('utt_id', '')!r}, "
        f"audio_ref={record.get('audio_ref', '')!r}, "
        f"audio_path={record.get('audio_path', '')!r}, "
        f"filename={record.get('filename', '')!r}"
    ]
    source = _record_source_string(record)
    if source:
        parts.append(f", source={source!r}")
    return "".join(parts)


def _resolve_record_audio_duration_seconds(record: dict[str, Any]) -> float | None:
    for key in ("duration_sec", "duration", "audio_duration_sec", "wav_duration"):
        raw_value = record.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str) and not raw_value.strip():
            continue
        try:
            duration_seconds = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration_seconds) and duration_seconds > 0:
            return duration_seconds
    return None


def _raise_if_audio_duration_exceeds_cap(
    record: dict[str, Any],
    *,
    duration_seconds: float | None,
    max_duration_seconds: float,
    source: str,
) -> None:
    if max_duration_seconds <= 0:
        return
    if duration_seconds is None or not math.isfinite(duration_seconds):
        return
    if duration_seconds <= max_duration_seconds:
        return
    raise ValueError(
        "audio_duration_guard[{source}] duration={duration:.2f}s exceeds cap={cap:.2f}s for {record}".format(
            source=source,
            duration=duration_seconds,
            cap=max_duration_seconds,
            record=_record_debug_string(record),
        )
    )


def _extract_train_shards(train_index_payload: dict[str, Any]) -> list[TrainShard]:
    shards: list[TrainShard] = []
    for index, item in enumerate(train_index_payload.get("train_shards") or []):
        path = str(item.get("path") or "").strip()
        if not path:
            raise ValueError(f"train_shards[{index}] is missing path")
        count = int(item.get("count") or 0)
        if count < 0:
            raise ValueError(f"train_shards[{index}] has negative count: {count}")
        if count == 0:
            continue
        shards.append(
            TrainShard(
                path=path,
                count=count,
                selection_indices_path=str(item.get("selection_indices_path") or "").strip(),
            )
        )
    if not shards:
        raise ValueError("train_index does not contain non-empty train_shards")
    return shards


def _limit_total_records(shards: list[TrainShard], *, max_train_samples: int | None) -> int:
    total_records = sum(shard.count for shard in shards)
    if max_train_samples is not None:
        total_records = min(total_records, max(int(max_train_samples), 0))
    return total_records


def _compute_synchronized_step_budget(
    *,
    total_records: int,
    per_device_batch_size: int,
    world_size: int,
    grad_accum_steps: int,
) -> tuple[int, int, int]:
    effective_batch = max(int(per_device_batch_size), 1) * max(int(world_size), 1) * max(int(grad_accum_steps), 1)
    steps_per_epoch = total_records // effective_batch
    if steps_per_epoch <= 0:
        raise ValueError(
            "train dataset does not contain enough records for one synchronized optimizer step: "
            f"records={total_records}, effective_batch={effective_batch}"
        )
    usable_total_records = steps_per_epoch * effective_batch
    per_rank_records = usable_total_records // max(int(world_size), 1)
    return steps_per_epoch, usable_total_records, per_rank_records


def _plan_record_range_slices(
    shards: list[TrainShard],
    *,
    start_record: int,
    record_count: int,
) -> list[TrainShardSlice]:
    if start_record < 0:
        raise ValueError(f"start_record must be >= 0, got {start_record}")
    if record_count < 0:
        raise ValueError(f"record_count must be >= 0, got {record_count}")
    if record_count == 0:
        return []
    end_record = start_record + record_count
    cursor = 0
    slices: list[TrainShardSlice] = []
    for shard in shards:
        shard_end = cursor + shard.count
        overlap_start = max(start_record, cursor)
        overlap_end = min(end_record, shard_end)
        if overlap_start < overlap_end:
            slices.append(
                TrainShardSlice(
                    path=shard.path,
                    start_record=overlap_start - cursor,
                    count=overlap_end - overlap_start,
                    selection_indices_path=shard.selection_indices_path,
                )
            )
        if shard_end >= end_record:
            break
        cursor = shard_end
    planned_records = sum(item.count for item in slices)
    if planned_records != record_count:
        raise ValueError(
            "failed to plan shard slices for requested record range: "
            f"start_record={start_record}, record_count={record_count}, planned={planned_records}"
        )
    return slices


def _slice_shard_slices(
    shard_slices: list[TrainShardSlice],
    *,
    start_record: int,
    record_count: int,
) -> list[TrainShardSlice]:
    if start_record < 0:
        raise ValueError(f"start_record must be >= 0, got {start_record}")
    if record_count < 0:
        raise ValueError(f"record_count must be >= 0, got {record_count}")
    if record_count == 0:
        return []
    end_record = start_record + record_count
    cursor = 0
    slices: list[TrainShardSlice] = []
    for shard_slice in shard_slices:
        shard_end = cursor + shard_slice.count
        overlap_start = max(start_record, cursor)
        overlap_end = min(end_record, shard_end)
        if overlap_start < overlap_end:
            slices.append(
                TrainShardSlice(
                    path=shard_slice.path,
                    start_record=shard_slice.start_record + (overlap_start - cursor),
                    count=overlap_end - overlap_start,
                    selection_indices_path=shard_slice.selection_indices_path,
                )
            )
        if shard_end >= end_record:
            break
        cursor = shard_end
    planned_records = sum(item.count for item in slices)
    if planned_records != record_count:
        raise ValueError(
            "failed to split worker shard slices for requested record range: "
            f"start_record={start_record}, record_count={record_count}, planned={planned_records}"
        )
    return slices


def plan_train_rank_slices(
    train_index_payload: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    epoch: int,
    shuffle_shards: bool,
    seed: int,
    per_device_batch_size: int,
    grad_accum_steps: int,
    max_train_samples: int | None = None,
) -> tuple[list[TrainShardSlice], int, int]:
    shard_specs = _extract_train_shards(train_index_payload)
    if shuffle_shards:
        rng = random.Random(int(seed) + int(epoch))
        rng.shuffle(shard_specs)
    total_records = _limit_total_records(shard_specs, max_train_samples=max_train_samples)
    steps_per_epoch, usable_total_records, per_rank_records = _compute_synchronized_step_budget(
        total_records=total_records,
        per_device_batch_size=per_device_batch_size,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
    )
    rank_index = int(rank)
    if rank_index < 0 or rank_index >= max(int(world_size), 1):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    rank_start = rank_index * per_rank_records
    rank_slices = _plan_record_range_slices(
        shard_specs,
        start_record=rank_start,
        record_count=per_rank_records,
    )
    return rank_slices, steps_per_epoch, usable_total_records


class _BaseAudioDataset:
    def __init__(self, runtime_config: DatasetRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self._processor: Qwen3ASRProcessor | None = None
        self._resolver: AudioResolver | None = None
        self._jsonl_handles: dict[str, Any] = {}
        self._jsonl_line_offsets: dict[str, np.ndarray] = {}
        self._selection_indices: dict[str, np.ndarray] = {}

    def _get_processor(self) -> Qwen3ASRProcessor:
        if self._processor is None:
            Qwen3ASRProcessor = _import_qwen_processor()
            self._processor = Qwen3ASRProcessor.from_pretrained(
                self.runtime_config.qwen_source,
                fix_mistral_regex=self.runtime_config.fix_mistral_regex,
                local_files_only=self.runtime_config.local_files_only,
            )
        return self._processor

    def _get_resolver(self) -> AudioResolver:
        if self._resolver is None:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            cache_dir = self._resolve_cache_dir(worker_id=worker_id)
            self._resolver = AudioResolver(
                AudioResolverConfig(
                    cache_dir=cache_dir,
                    oss_config_path=self.runtime_config.oss_config_path,
                    oss_workers=self.runtime_config.oss_workers,
                    prefetch_workers=self.runtime_config.prefetch_workers,
                    local_audio_roots=self.runtime_config.local_audio_roots,
                    cache_mode=self.runtime_config.cache_mode,
                    resolve_timeout_seconds=self.runtime_config.audio_resolve_timeout_seconds,
                )
            )
        return self._resolver

    def _resolve_cache_dir(self, *, worker_id: int) -> str:
        cache_root = Path(self.runtime_config.cache_dir).expanduser().resolve()
        if self.runtime_config.cache_mode == "persistent":
            return str(cache_root)
        return str(cache_root / f"worker{worker_id:04d}")

    def _get_jsonl_handle(self, path: str):
        resolved_path = str(Path(path).expanduser().resolve())
        handle = self._jsonl_handles.get(resolved_path)
        if handle is None or handle.closed:
            handle = open(resolved_path, "rb")
            self._jsonl_handles[resolved_path] = handle
        return handle

    def _get_line_offsets(self, path: str) -> np.ndarray:
        resolved_path = str(Path(path).expanduser().resolve())
        offsets = self._jsonl_line_offsets.get(resolved_path)
        if offsets is None:
            offsets = build_or_load_line_offsets(resolved_path)
            self._jsonl_line_offsets[resolved_path] = offsets
        return offsets

    def _get_selection_indices(self, selection_indices_path: str) -> np.ndarray:
        resolved_path = str(Path(selection_indices_path).expanduser().resolve())
        indices = self._selection_indices.get(resolved_path)
        if indices is None:
            indices = np.load(resolved_path, mmap_mode="r")
            self._selection_indices[resolved_path] = indices
        return indices

    def _read_jsonl_line_at_offset(self, path: str, offset: int) -> dict[str, Any]:
        record = read_jsonl_line_at_offset(path, offset, handle=self._get_jsonl_handle(path))
        if not str(record.get("target_text") or "").strip():
            gold_text = str(record.get("gold_text") or "").strip()
            if gold_text:
                record["target_text"] = gold_text
        if not str(record.get("text") or "").strip():
            gold_text = str(record.get("gold_text") or "").strip()
            if gold_text:
                record["text"] = gold_text
        return record

    def _encode_record(self, record: dict[str, Any]) -> dict[str, Any]:
        processor = self._get_processor()
        resolver = self._get_resolver()
        max_audio_duration_seconds = max(float(self.runtime_config.max_audio_duration_seconds or 0.0), 0.0)
        _raise_if_audio_duration_exceeds_cap(
            record,
            duration_seconds=_resolve_record_audio_duration_seconds(record),
            max_duration_seconds=max_audio_duration_seconds,
            source="metadata",
        )
        audio_path: str | None = None
        try:
            audio_path = resolver.resolve_record(record)
            waveform = load_audio_mono_16k(
                audio_path,
                timeout_seconds=self.runtime_config.audio_resolve_timeout_seconds,
            )
        except RuntimeError:
            audio_ref = str(record.get("audio_ref") or "").strip()
            audio_path_obj = Path(audio_path) if audio_path else Path(str(record.get("audio_path") or audio_ref or ""))
            if not audio_ref.startswith("oss://"):
                raise
            if audio_path_obj.exists():
                audio_path_obj.unlink()
            audio_path = resolver.resolve_oss_uri(audio_ref)
            waveform = load_audio_mono_16k(
                audio_path,
                timeout_seconds=self.runtime_config.audio_resolve_timeout_seconds,
            )
        except (FileNotFoundError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"Failed to materialize audio for {_record_debug_string(record)}") from exc
        waveform_duration_seconds = float(waveform.shape[-1]) / 16000.0
        _raise_if_audio_duration_exceeds_cap(
            record,
            duration_seconds=waveform_duration_seconds,
            max_duration_seconds=max_audio_duration_seconds,
            source="waveform",
        )
        target_text = _resolve_target_text(record, target_format=self.runtime_config.target_format)
        text = f"{processor.audio_token}{target_text}"
        encoded = processor(
            text=text,
            audio=waveform,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0).to(dtype=torch.long)
        attention_mask = encoded["attention_mask"].squeeze(0).to(dtype=torch.long)
        input_features = encoded["input_features"].squeeze(0).to(dtype=torch.float32)
        feature_attention_mask = encoded["feature_attention_mask"].squeeze(0).to(dtype=torch.long)

        eos_token_id = processor.tokenizer.eos_token_id
        if self.runtime_config.add_eos and eos_token_id is not None:
            eos_tensor = torch.tensor([int(eos_token_id)], dtype=torch.long)
            input_ids = torch.cat([input_ids, eos_tensor], dim=0)
            attention_mask = torch.cat([attention_mask, torch.ones_like(eos_tensor)], dim=0)

        audio_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.audio_token))
        labels = input_ids.clone()
        labels[input_ids == audio_token_id] = -100

        return {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_path": audio_path,
            "dataset_key": str(record.get("dataset_key") or ""),
            "audio_duration_seconds": waveform_duration_seconds,
            "feature_length": int(input_features.size(-1)),
            "is_replacement": bool(record.get("__is_replacement__", False)),
            "record": record,
        }

    def _shutdown_runtime(self) -> None:
        for handle in self._jsonl_handles.values():
            try:
                handle.close()
            except OSError:
                pass
        self._jsonl_handles.clear()
        self._jsonl_line_offsets.clear()
        self._selection_indices.clear()
        if self._resolver is not None:
            self._resolver.shutdown()
            self._resolver = None


class Stage1ShardIterableDataset(IterableDataset, _BaseAudioDataset):
    def __init__(
        self,
        shard_slices: list[TrainShardSlice],
        *,
        runtime_config: DatasetRuntimeConfig,
        rank: int,
        epoch: int,
    ) -> None:
        IterableDataset.__init__(self)
        _BaseAudioDataset.__init__(self, runtime_config)
        self.shard_slices = list(shard_slices)
        self.rank = int(rank)
        self.epoch = int(epoch)
        self.rank_record_count = sum(item.count for item in self.shard_slices)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        worker_start = (self.rank_record_count * worker_id) // num_workers
        worker_end = (self.rank_record_count * (worker_id + 1)) // num_workers
        worker_target_records = worker_end - worker_start
        worker_slices = _slice_shard_slices(
            self.shard_slices,
            start_record=worker_start,
            record_count=worker_end - worker_start,
        )

        resolver = self._get_resolver()
        prefetch_queue: deque[tuple[str, list[dict[str, Any]]]] = deque()
        prefetch_ahead = max(int(self.runtime_config.prefetch_groups_ahead), 0)
        group_iter = self._iter_record_groups(
            shard_slices=worker_slices,
            worker_id=worker_id,
        )

        def fill_prefetch_queue() -> None:
            while len(prefetch_queue) < prefetch_ahead + 1:
                try:
                    group_id, records = next(group_iter)
                except StopIteration:
                    break
                resolver.stage_group(group_id, records)
                prefetch_queue.append((group_id, records))

        yielded_records = 0
        skipped_records = 0
        replacement_pool: deque[dict[str, Any]] = deque(maxlen=8)
        worker_label = (
            f"rank={self.rank} epoch={self.epoch} worker={worker_id} "
            f"target_records={worker_target_records}"
        )
        try:
            fill_prefetch_queue()
            while prefetch_queue:
                group_id, records = prefetch_queue.popleft()
                fill_prefetch_queue()
                for record in records:
                    try:
                        encoded = self._encode_record(record)
                    except Exception as exc:
                        skipped_records += 1
                        _LOGGER.warning(
                            "%s | skipping failed training record | %s | error=%s",
                            worker_label,
                            _record_debug_string(record),
                            exc,
                        )
                        continue
                    replacement_pool.append(encoded)
                    yielded_records += 1
                    yield encoded
                resolver.release_group(group_id)
            failed_fraction = (skipped_records / worker_target_records) if worker_target_records > 0 else 0.0
            if skipped_records:
                _LOGGER.warning(
                    "%s | completed with skipped_records=%d/%d (%.2f%%)",
                    worker_label,
                    skipped_records,
                    worker_target_records,
                    failed_fraction * 100.0,
                )
            if failed_fraction > max(float(self.runtime_config.max_failed_record_fraction or 0.0), 0.0):
                raise RuntimeError(
                    f"{worker_label} exceeded max_failed_record_fraction="
                    f"{self.runtime_config.max_failed_record_fraction:.4f} with skipped_records="
                    f"{skipped_records}/{worker_target_records}"
                )
            if yielded_records < worker_target_records:
                if not replacement_pool:
                    raise RuntimeError(f"{worker_label} failed to encode any training records")
                padding_records = worker_target_records - yielded_records
                replacement_items = list(replacement_pool)
                _LOGGER.warning(
                    "%s | padding %d replacement records from a pool of %d recent items to preserve synchronized train steps",
                    worker_label,
                    padding_records,
                    len(replacement_items),
                )
                for index in range(padding_records):
                    yield self._clone_encoded_item(
                        replacement_items[index % len(replacement_items)],
                        is_replacement=True,
                    )
        finally:
            self._shutdown_runtime()

    @staticmethod
    def _clone_encoded_item(item: dict[str, Any], *, is_replacement: bool = False) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            elif key == "record" and isinstance(value, dict):
                cloned[key] = dict(value)
            else:
                cloned[key] = value
        cloned["is_replacement"] = bool(is_replacement)
        if isinstance(cloned.get("record"), dict):
            cloned["record"]["__is_replacement__"] = bool(is_replacement)
        return cloned

    @staticmethod
    def _interleave_records_by_duration(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按音频时长升序排序后顺序输出，使长度相近的样本进入同一 batch，减少 padding 浪费和显存峰值不均。"""
        if len(records) <= 1:
            return records
        indexed = [(i, _resolve_record_audio_duration_seconds(r) or 0.0) for i, r in enumerate(records)]
        indexed.sort(key=lambda x: x[1])
        return [records[i] for i, _ in indexed]

    def _iter_record_groups(
        self,
        *,
        shard_slices: list[TrainShardSlice],
        worker_id: int,
    ):
        def selected_record_groups(
            shard_slice: TrainShardSlice,
            *,
            group_size: int,
        ) -> Iterator[list[dict[str, Any]]]:
            if not shard_slice.selection_indices_path:
                return
            selected_indices = self._get_selection_indices(shard_slice.selection_indices_path)
            if shard_slice.start_record >= int(selected_indices.size):
                return
            selection_stop = min(shard_slice.start_record + shard_slice.count, int(selected_indices.size))
            if selection_stop <= shard_slice.start_record:
                return
            line_offsets = self._get_line_offsets(shard_slice.path)
            group_records: list[dict[str, Any]] = []
            for array_index in range(shard_slice.start_record, selection_stop):
                line_index = int(selected_indices[array_index])
                if line_index < 0 or line_index >= int(line_offsets.size):
                    raise ValueError(
                        "selection_indices_path points outside the source shard: "
                        f"path={shard_slice.path}, selection_indices_path={shard_slice.selection_indices_path}, "
                        f"line_index={line_index}, available={int(line_offsets.size)}"
                    )
                record = self._read_jsonl_line_at_offset(shard_slice.path, int(line_offsets[line_index]))
                group_records.append(record)
                if group_size > 0 and len(group_records) >= group_size:
                    yield group_records
                    group_records = []
            if group_records:
                yield group_records

        assigned_shard_index = 0
        group_size = max(int(self.runtime_config.prefetch_group_records or 0), 0)
        chunk_records = group_size > 0

        for shard_slice in shard_slices:
            if shard_slice.selection_indices_path:
                if not chunk_records:
                    records_iter = selected_record_groups(shard_slice, group_size=shard_slice.count)
                    records = next(records_iter, [])
                    if not records:
                        assigned_shard_index += 1
                        continue
                    records = self._interleave_records_by_duration(records)
                    group_id = f"rank{self.rank:04d}_worker{worker_id:04d}_shard{assigned_shard_index:06d}_group000000"
                    yield group_id, records
                else:
                    has_records = False
                    for group_index, group_records in enumerate(selected_record_groups(shard_slice, group_size=group_size)):
                        has_records = True
                        group_records = self._interleave_records_by_duration(group_records)
                        group_id = (
                            f"rank{self.rank:04d}_worker{worker_id:04d}_"
                            f"shard{assigned_shard_index:06d}_group{group_index:06d}"
                        )
                        yield group_id, group_records
                    if not has_records:
                        assigned_shard_index += 1
                        continue
            elif not chunk_records:
                records = load_jsonl_records(
                    shard_slice.path,
                    skip_samples=shard_slice.start_record,
                    max_samples=shard_slice.count,
                )
                if not records:
                    assigned_shard_index += 1
                    continue
                records = self._interleave_records_by_duration(records)
                group_id = f"rank{self.rank:04d}_worker{worker_id:04d}_shard{assigned_shard_index:06d}_group000000"
                yield group_id, records
            else:
                has_records = False
                for group_index, group_records in enumerate(
                    iter_jsonl_record_groups(
                        shard_slice.path,
                        group_size=group_size,
                        skip_samples=shard_slice.start_record,
                        max_samples=shard_slice.count,
                    )
                ):
                    has_records = True
                    group_records = self._interleave_records_by_duration(group_records)
                    group_id = (
                        f"rank{self.rank:04d}_worker{worker_id:04d}_"
                        f"shard{assigned_shard_index:06d}_group{group_index:06d}"
                    )
                    yield group_id, group_records
                if not has_records:
                    assigned_shard_index += 1
                    continue
            assigned_shard_index += 1


class WeightedInterleaveIterableDataset(IterableDataset, _BaseAudioDataset):
    """加权交织 + 局部池分桶的训练 IterableDataset。

    - 高随机性：按「剩余量」加权轮询多个 dataset 的样本子流，再按长度桶内优先不同 dataset_key 组 batch。
    - 省显存：batch 约束在同一/相邻长度桶内，减少 padding 浪费和单步峰值不均。
    - 负载均衡：每个 step 内的多个 micro-batch 会按估计 cost 重新配平，降低 grad_accum 下的峰值抖动。
    - DDP 同步：各 rank 使用同样的 per-rank target；各 worker 再做精确整数分配，避免静默截断/补齐。
    """

    def __init__(
        self,
        specs: list[DatasetSpec],
        *,
        per_rank_targets: list[int],
        usable_per_rank: int,
        runtime_config: DatasetRuntimeConfig,
        rank: int,
        world_size: int,
        epoch: int,
        seed: int,
        per_device_batch_size: int,
        grad_accum_steps: int,
        bucket_factor: int,
    ) -> None:
        IterableDataset.__init__(self)
        _BaseAudioDataset.__init__(self, runtime_config)
        self.specs = list(specs)
        self.per_rank_targets = [int(value) for value in per_rank_targets]
        self.usable_per_rank = int(usable_per_rank)
        self.rank = int(rank)
        self.world_size = max(int(world_size), 1)
        self.epoch = int(epoch)
        self.seed = int(seed)
        self.per_device_batch_size = max(int(per_device_batch_size), 1)
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.bucket_factor = max(int(bucket_factor), 1)


        self.num_length_bins = max(
            int(getattr(self.runtime_config, "num_length_bins", DEFAULT_NUM_LENGTH_BINS)), 1
        )
        self.length_bin_edges = _build_length_bin_edges(
            float(getattr(self.runtime_config, "max_audio_duration_seconds", 0.0) or 0.0),
            self.num_length_bins,
        )

    def _dataset_seed(self, dataset_key: str) -> int:
        return _stable_seed("dataset-mix", self.seed, self.epoch, dataset_key)

    def _group_seed(self, *, worker_id: int, group_index: int) -> int:
        return _stable_seed("dataset-group", self.seed, self.epoch, self.rank, worker_id, group_index)

    @staticmethod
    def _build_oversampled_perm(count: int, length: int, rng: np.random.Generator) -> np.ndarray:
        """产生长度为 length 的 index 序列；length<=count 取随机子集，length>count 则循环过采样。"""
        if length <= 0 or count <= 0:
            return np.empty(0, dtype=np.int64)
        if length <= count:
            return rng.permutation(count)[:length].astype(np.int64, copy=False)
        chunks: list[np.ndarray] = []
        remaining = length
        while remaining > 0:
            perm = rng.permutation(count)
            take = min(remaining, count)
            chunks.append(perm[:take])
            remaining -= take
        return np.concatenate(chunks).astype(np.int64, copy=False)

    @staticmethod
    def _microbatch_cost(batch: list[PlannedMixedSample]) -> float:
        if not batch:
            return 0.0
        max_length = max(sample.length_proxy for sample in batch)
        return float(max_length * len(batch))

    def _compose_group_items(
        self,
        group_items: list[tuple[int, int]],
        *,
        stream_specs: list[DatasetSpec],
        group_seed: int,
    ) -> list[tuple[int, int]]:
        if not group_items:
            return []
        planned_samples = [
            PlannedMixedSample(
                stream_index=stream_index,
                global_index=global_index,
                dataset_key=stream_specs[stream_index].dataset_key,
                length_proxy=int(stream_specs[stream_index].lengths[global_index]),
                length_bin=_resolve_length_bin(
                    int(stream_specs[stream_index].lengths[global_index]), self.length_bin_edges
                ),
            )
            for stream_index, global_index in group_items
        ]
        queues_by_bin: dict[int, dict[str, deque[PlannedMixedSample]]] = defaultdict(dict)
        bin_totals: dict[int, int] = defaultdict(int)
        for sample in sorted(planned_samples, key=lambda item: (item.length_proxy, item.dataset_key, item.global_index)):
            dataset_queue = queues_by_bin.setdefault(sample.length_bin, {}).setdefault(sample.dataset_key, deque())
            dataset_queue.append(sample)
            bin_totals[sample.length_bin] += 1

        preferred_bin_offset = group_seed % self.num_length_bins
        microbatches: list[list[PlannedMixedSample]] = []

        def active_bins() -> list[int]:
            return [bin_id for bin_id, total in bin_totals.items() if total > 0]

        def ordered_dataset_keys(bin_id: int, *, used_keys: set[str] | None = None) -> list[str]:
            used = used_keys or set()
            return sorted(
                [
                    key
                    for key, queue in queues_by_bin.get(bin_id, {}).items()
                    if queue and (key not in used)
                ],
                key=lambda key: (
                    len(queues_by_bin[bin_id][key]),
                    -(_stable_seed(group_seed, "dataset-order", bin_id, key)),
                ),
                reverse=True,
            )

        def pop_from_bin(bin_id: int, batch: list[PlannedMixedSample], *, distinct_only: bool) -> bool:
            used_keys = {sample.dataset_key for sample in batch}
            dataset_keys = ordered_dataset_keys(bin_id, used_keys=used_keys if distinct_only else None)
            progressed = False
            for dataset_key in dataset_keys:
                if len(batch) >= self.per_device_batch_size:
                    break
                queue = queues_by_bin[bin_id][dataset_key]
                if not queue:
                    continue
                batch.append(queue.popleft())
                bin_totals[bin_id] -= 1
                progressed = True
            return progressed

        microbatch_index = 0
        while active_bins():
            preferred_bin = (preferred_bin_offset + microbatch_index) % self.num_length_bins
            target_bin = max(
                active_bins(),
                key=lambda bin_id: (bin_totals[bin_id], -abs(bin_id - preferred_bin), -bin_id),
            )
            batch: list[PlannedMixedSample] = []
            pop_from_bin(target_bin, batch, distinct_only=True)
            while len(batch) < self.per_device_batch_size and pop_from_bin(target_bin, batch, distinct_only=False):
                continue
            if len(batch) < self.per_device_batch_size:
                fallback_bins = sorted(
                    active_bins(),
                    key=lambda bin_id: (abs(bin_id - target_bin), -bin_totals[bin_id], bin_id),
                )
                for fallback_bin in fallback_bins:
                    if fallback_bin == target_bin:
                        continue
                    pop_from_bin(fallback_bin, batch, distinct_only=True)
                    while len(batch) < self.per_device_batch_size and pop_from_bin(fallback_bin, batch, distinct_only=False):
                        continue
                    if len(batch) >= self.per_device_batch_size:
                        break
            if len(batch) != self.per_device_batch_size:
                raise RuntimeError(
                    "failed to compose a full mixed micro-batch from planned samples: "
                    f"group_size={len(group_items)}, batch_size={self.per_device_batch_size}, built={len(batch)}"
                )
            microbatches.append(batch)
            microbatch_index += 1

        if not microbatches:
            return []

        step_capacity = max(self.grad_accum_steps, 1)
        step_count = max(math.ceil(len(microbatches) / step_capacity), 1)
        step_buckets = [{"cost": 0.0, "batches": []} for _ in range(step_count)]
        for batch in sorted(microbatches, key=self._microbatch_cost, reverse=True):
            available_steps = [index for index, bucket in enumerate(step_buckets) if len(bucket["batches"]) < step_capacity]
            if not available_steps:
                raise RuntimeError("mixed batch rebalancing exhausted all step buckets unexpectedly")
            chosen_step = min(
                available_steps,
                key=lambda index: (step_buckets[index]["cost"], len(step_buckets[index]["batches"]), index),
            )
            step_buckets[chosen_step]["batches"].append(batch)
            step_buckets[chosen_step]["cost"] += self._microbatch_cost(batch)

        ordered_samples: list[tuple[int, int]] = []
        for bucket in step_buckets:
            for batch in sorted(bucket["batches"], key=self._microbatch_cost):
                ordered_samples.extend((sample.stream_index, sample.global_index) for sample in batch)
        return ordered_samples

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        pdbs = self.per_device_batch_size
        worker_targets = _allocate_worker_batch_targets(
            usable_total=self.usable_per_rank,
            batch_size=pdbs,
            num_workers=num_workers,
        )
        worker_target = worker_targets[worker_id] if worker_id < len(worker_targets) else 0
        worker_allocations = _allocate_dataset_targets_to_workers(
            self.per_rank_targets,
            worker_targets=worker_targets,
        )

        worker_label = (
            f"rank={self.rank} epoch={self.epoch} worker={worker_id} target_records={worker_target}"
        )


        worker_streams: list[np.ndarray] = []
        stream_specs: list[DatasetSpec] = []
        for dataset_index, (spec, prt) in enumerate(zip(self.specs, self.per_rank_targets)):
            if prt <= 0:
                continue
            perm = self._build_oversampled_perm(
                spec.count,
                prt * self.world_size,
                np.random.default_rng(self._dataset_seed(spec.dataset_key)),
            )
            rank_indices = perm[self.rank * prt : (self.rank + 1) * prt]
            dataset_worker_allocations = worker_allocations[dataset_index]
            if worker_id >= len(dataset_worker_allocations):
                continue
            dataset_worker_target = int(dataset_worker_allocations[worker_id])
            if dataset_worker_target <= 0:
                continue
            w_start = sum(dataset_worker_allocations[:worker_id])
            w_end = w_start + dataset_worker_target
            worker_indices = rank_indices[w_start:w_end]
            if len(worker_indices) == 0:
                continue
            worker_streams.append(worker_indices)
            stream_specs.append(spec)

        if worker_target <= 0 or not worker_streams:
            return

        interleave_rng = random.Random(_stable_seed("interleave", self.seed, self.epoch, self.rank, worker_id))

        resolver = self._get_resolver()
        group_size = int(self.runtime_config.prefetch_group_records or 0)
        if group_size <= 0:
            group_size = pdbs * self.bucket_factor
        group_size = max((group_size // pdbs) * pdbs, pdbs)
        prefetch_ahead = max(int(self.runtime_config.prefetch_groups_ahead), 0)

        yielded_records = 0
        skipped_records = 0
        replacement_pool: deque[dict[str, Any]] = deque(maxlen=max(pdbs * 4, 32))

        def interleaved_items():
            positions = [0] * len(worker_streams)
            remaining = [len(stream) for stream in worker_streams]
            total = sum(remaining)
            while total > 0:
                pick = interleave_rng.random() * total
                acc = 0.0
                chosen = -1
                for stream_index, rem in enumerate(remaining):
                    if rem <= 0:
                        continue
                    acc += rem
                    if pick < acc:
                        chosen = stream_index
                        break
                if chosen < 0:
                    for stream_index, rem in enumerate(remaining):
                        if rem > 0:
                            chosen = stream_index
                            break
                global_index = int(worker_streams[chosen][positions[chosen]])
                positions[chosen] += 1
                remaining[chosen] -= 1
                total -= 1
                yield chosen, global_index

        def iter_groups():
            group_items: list[tuple[int, int]] = []
            group_index = 0
            for item in interleaved_items():
                group_items.append(item)
                if len(group_items) >= group_size:
                    yield group_index, group_items
                    group_index += 1
                    group_items = []
            if group_items:
                yield group_index, group_items

        group_iter = iter_groups()
        prefetch_queue: deque[tuple[str, list[dict[str, Any] | None]]] = deque()

        def materialize_group(group_index: int, group_items: list[tuple[int, int]]) -> tuple[str, list[dict[str, Any] | None]]:
            ordered_items = self._compose_group_items(
                group_items,
                stream_specs=stream_specs,
                group_seed=self._group_seed(worker_id=worker_id, group_index=group_index),
            )
            group_id = f"rank{self.rank:04d}_worker{worker_id:04d}_group{group_index:06d}"
            records: list[dict[str, Any] | None] = []
            for stream_index, global_index in ordered_items:
                spec = stream_specs[stream_index]
                try:
                    record = self._read_jsonl_line_at_offset(spec.train_path, int(spec.offsets[global_index]))
                    record.setdefault("dataset_key", spec.dataset_key)
                except Exception as exc:
                    _LOGGER.warning(
                        "%s | failed to read record at offset | dataset=%s | error=%s",
                        worker_label,
                        spec.dataset_key,
                        exc,
                    )
                    record = None
                records.append(record)
            resolver.stage_group(group_id, [record for record in records if record is not None])
            return group_id, records

        def fill_prefetch_queue() -> None:
            while len(prefetch_queue) < prefetch_ahead + 1:
                try:
                    group_index, group_items = next(group_iter)
                except StopIteration:
                    break
                prefetch_queue.append(materialize_group(group_index, group_items))

        def process_group(group_id: str, records: list[dict[str, Any] | None]):
            nonlocal yielded_records, skipped_records
            try:
                for record in records:
                    if yielded_records >= worker_target:
                        break
                    if record is None:
                        skipped_records += 1
                        continue
                    try:
                        encoded = self._encode_record(record)
                    except Exception as exc:
                        skipped_records += 1
                        _LOGGER.warning(
                            "%s | skipping failed training record | %s | error=%s",
                            worker_label,
                            _record_debug_string(record),
                            exc,
                        )
                        continue
                    replacement_pool.append(encoded)
                    yielded_records += 1
                    yield encoded
            finally:
                resolver.release_group(group_id)

        try:
            fill_prefetch_queue()
            while prefetch_queue and yielded_records < worker_target:
                group_id, records = prefetch_queue.popleft()
                fill_prefetch_queue()
                yield from process_group(group_id, records)

            failed_fraction = (skipped_records / worker_target) if worker_target > 0 else 0.0
            if skipped_records:
                _LOGGER.warning(
                    "%s | completed with skipped_records=%d/%d (%.2f%%)",
                    worker_label,
                    skipped_records,
                    worker_target,
                    failed_fraction * 100.0,
                )
            if failed_fraction > max(float(self.runtime_config.max_failed_record_fraction or 0.0), 0.0):
                raise RuntimeError(
                    f"{worker_label} exceeded max_failed_record_fraction="
                    f"{self.runtime_config.max_failed_record_fraction:.4f} with skipped_records="
                    f"{skipped_records}/{worker_target}"
                )
            if yielded_records < worker_target:
                if not replacement_pool:
                    raise RuntimeError(f"{worker_label} failed to encode any training records")
                replacement_items = list(replacement_pool)
                padding_records = worker_target - yielded_records
                _LOGGER.warning(
                    "%s | padding %d replacement records from a pool of %d recent items to preserve synchronized train steps",
                    worker_label,
                    padding_records,
                    len(replacement_items),
                )
                for index in range(padding_records):
                    yielded_records += 1
                    yield Stage1ShardIterableDataset._clone_encoded_item(
                        replacement_items[index % len(replacement_items)],
                        is_replacement=True,
                    )
        finally:
            self._shutdown_runtime()


class JsonlAudioDataset(Dataset, _BaseAudioDataset):
    def __init__(self, records: list[dict[str, Any]], *, runtime_config: DatasetRuntimeConfig) -> None:
        Dataset.__init__(self)
        _BaseAudioDataset.__init__(self, runtime_config)
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._encode_record(self.records[index])

    def __del__(self) -> None:
        self._shutdown_runtime()


class XAuTCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch")

        batch_size = len(batch)
        feature_dim = int(batch[0]["input_features"].size(0))
        max_feature_len = max(int(item["input_features"].size(-1)) for item in batch)
        max_token_len = max(int(item["input_ids"].size(0)) for item in batch)

        input_features = torch.zeros((batch_size, feature_dim, max_feature_len), dtype=torch.float32)
        feature_attention_mask = torch.zeros((batch_size, max_feature_len), dtype=torch.long)
        input_ids = torch.full((batch_size, max_token_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_token_len), dtype=torch.long)
        labels = torch.full((batch_size, max_token_len), -100, dtype=torch.long)

        records: list[dict[str, Any]] = []
        audio_paths: list[str] = []
        dataset_keys: list[str] = []
        audio_durations_seconds: list[float] = []
        feature_lengths: list[int] = []
        replacement_flags: list[int] = []
        for row_index, item in enumerate(batch):
            current_feature_len = int(item["input_features"].size(-1))
            current_token_len = int(item["input_ids"].size(0))
            input_features[row_index, :, :current_feature_len] = item["input_features"][:, :current_feature_len]
            feature_attention_mask[row_index, :current_feature_len] = item["feature_attention_mask"][:current_feature_len]
            input_ids[row_index, :current_token_len] = item["input_ids"][:current_token_len]
            attention_mask[row_index, :current_token_len] = item["attention_mask"][:current_token_len]
            labels[row_index, :current_token_len] = item["labels"][:current_token_len]
            records.append(item["record"])
            audio_paths.append(str(item["audio_path"]))
            dataset_keys.append(str(item.get("dataset_key") or item["record"].get("dataset_key") or ""))
            audio_durations_seconds.append(float(item.get("audio_duration_seconds") or 0.0))
            feature_lengths.append(current_feature_len)
            replacement_flags.append(1 if bool(item.get("is_replacement", False)) else 0)

        return {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "records": records,
            "audio_paths": audio_paths,
            "dataset_keys": dataset_keys,
            "audio_durations_seconds": audio_durations_seconds,
            "feature_lengths": feature_lengths,
            "replacement_flags": replacement_flags,
        }


def estimate_train_steps(
    train_index_payload: dict[str, Any],
    *,
    per_device_batch_size: int,
    world_size: int,
    grad_accum_steps: int,
    max_train_samples: int | None = None,
) -> int:
    if is_dataset_mix_index(train_index_payload):
        specs = load_dataset_specs(train_index_payload)
        targets = _compute_dataset_targets(specs)
        steps_per_epoch, _, _ = _compute_dataset_mix_budget(
            targets,
            per_device_batch_size=per_device_batch_size,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
            max_train_samples=max_train_samples,
        )
        return steps_per_epoch
    shards = _extract_train_shards(train_index_payload)
    total_records = _limit_total_records(shards, max_train_samples=max_train_samples)
    steps_per_epoch, _, _ = _compute_synchronized_step_budget(
        total_records=total_records,
        per_device_batch_size=per_device_batch_size,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
    )
    return steps_per_epoch


def build_train_dataset(
    train_index_payload: dict[str, Any],
    *,
    runtime_config: DatasetRuntimeConfig,
    rank: int,
    world_size: int,
    epoch: int,
    shuffle_shards: bool,
    seed: int,
    per_device_batch_size: int,
    grad_accum_steps: int,
    max_train_samples: int | None = None,
    bucket_factor: int = 16,
) -> IterableDataset:
    if is_dataset_mix_index(train_index_payload):
        specs = load_dataset_specs(train_index_payload)
        targets = _compute_dataset_targets(specs)
        _, per_rank_targets, usable_per_rank = _compute_dataset_mix_budget(
            targets,
            per_device_batch_size=per_device_batch_size,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
            max_train_samples=max_train_samples,
        )
        return WeightedInterleaveIterableDataset(
            specs,
            per_rank_targets=per_rank_targets,
            usable_per_rank=usable_per_rank,
            runtime_config=runtime_config,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            seed=seed,
            per_device_batch_size=per_device_batch_size,
            grad_accum_steps=grad_accum_steps,
            bucket_factor=bucket_factor,
        )
    rank_slices, _, _ = plan_train_rank_slices(
        train_index_payload,
        rank=rank,
        world_size=world_size,
        epoch=epoch,
        shuffle_shards=shuffle_shards,
        seed=seed,
        per_device_batch_size=per_device_batch_size,
        grad_accum_steps=grad_accum_steps,
        max_train_samples=max_train_samples,
    )
    return Stage1ShardIterableDataset(
        rank_slices,
        runtime_config=runtime_config,
        rank=rank,
        epoch=epoch,
    )


def build_val_dataset(
    val_manifest_path: str,
    *,
    runtime_config: DatasetRuntimeConfig,
    max_val_samples: int | None = None,
) -> JsonlAudioDataset:
    return JsonlAudioDataset(
        load_jsonl_records(val_manifest_path, max_samples=max_val_samples),
        runtime_config=runtime_config,
    )


def build_train_dataloader(
    dataset: IterableDataset,
    *,
    pad_token_id: int,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    prefetch_factor: int | None = None,
    timeout_seconds: float = 0.0,
) -> DataLoader:
    worker_count = max(int(num_workers), 0)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": max(int(batch_size), 1),
        "num_workers": worker_count,
        "persistent_workers": bool(persistent_workers and worker_count > 0),
        "pin_memory": pin_memory,
        "timeout": 0 if worker_count == 0 else max(int(math.ceil(float(timeout_seconds))), 0),
        "collate_fn": XAuTCollator(pad_token_id=pad_token_id),
    }
    if worker_count > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return DataLoader(
        **loader_kwargs,
    )


def build_val_dataloader(
    dataset: JsonlAudioDataset,
    *,
    pad_token_id: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    timeout_seconds: float = 0.0,
) -> DataLoader:
    worker_count = max(int(num_workers), 0)
    return DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=worker_count,
        pin_memory=pin_memory,
        timeout=0 if worker_count == 0 else max(int(math.ceil(float(timeout_seconds))), 0),
        collate_fn=XAuTCollator(pad_token_id=pad_token_id),
    )
