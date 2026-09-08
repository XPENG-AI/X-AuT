from __future__ import annotations

import json
import logging
import os
import random
import signal
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F_audio


_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")
_LOGGER = logging.getLogger(__name__)


class _Oss2Downloader:
    """Thin wrapper around oss2 that satisfies the interface used by AudioResolver."""

    def __init__(self, *, access_key_id: str, access_key_secret: str, endpoint: str) -> None:
        try:
            import oss2
        except ImportError as exc:
            raise ImportError(
                "oss:// audio URIs require the 'oss2' package — install it with `pip install oss2`."
            ) from exc
        self._auth = oss2.Auth(access_key_id, access_key_secret)
        self._endpoint = endpoint
        self._buckets: dict[str, Any] = {}

    def _get_bucket(self, bucket_name: str):
        bucket = self._buckets.get(bucket_name)
        if bucket is None:
            import oss2

            bucket = oss2.Bucket(self._auth, self._endpoint, bucket_name)
            self._buckets[bucket_name] = bucket
        return bucket


def _find_existing_path(candidates: list[str]) -> str:
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value and Path(value).is_file():
            return value
    return ""


@dataclass
class AudioResolverConfig:
    cache_dir: str
    oss_config_path: str = ""
    oss_workers: int = 4
    prefetch_workers: int = 1
    local_audio_roots: tuple[str, ...] = ()
    cache_mode: str = "persistent"
    resolve_timeout_seconds: float = 0.0


class AudioResolver:
    def __init__(self, config: AudioResolverConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_mode = str(config.cache_mode or "persistent").strip().lower()
        self._downloader = None
        self._filename_index: dict[str, str] = {}
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_futures: dict[str, Future[str]] = {}
        self._object_size_cache: dict[str, int] = {}
        self._staged_groups: dict[str, tuple[str, ...]] = {}
        self._resolve_timeout_seconds = max(float(config.resolve_timeout_seconds or 0.0), 0.0)
        self._oss_retry_attempts = max(int(os.environ.get("X_AUT_OSS_RETRY_ATTEMPTS", "8") or 8), 1)
        self._oss_retry_base_delay = max(float(os.environ.get("X_AUT_OSS_RETRY_BASE_DELAY", "0.5") or 0.5), 0.0)
        self._oss_retry_max_delay = max(
            float(os.environ.get("X_AUT_OSS_RETRY_MAX_DELAY", "8.0") or 8.0),
            self._oss_retry_base_delay,
        )
        configured_lock_wait_seconds = float(os.environ.get("X_AUT_OSS_LOCK_WAIT_SECONDS", "0") or 0.0)
        self._lock_wait_seconds = max(
            configured_lock_wait_seconds,
            self._estimate_retry_budget_seconds() + 60.0,
        )

    def resolve_record(self, record: dict[str, Any]) -> str:
        direct_path = _find_existing_path(
            [
                str(record.get("audio_path", "")),
                str(record.get("wav_path", "")),
                str(record.get("audio_local_path", "")),
                str(record.get("local_audio_path", "")),
                str(record.get("path", "")),
            ]
        )
        if direct_path:
            return direct_path
        audio_ref = str(record.get("audio_ref") or "").strip()
        if audio_ref.startswith("oss://"):
            return self.resolve_oss_uri(audio_ref)
        if audio_ref and Path(audio_ref).is_file():
            return audio_ref
        filename = str(record.get("filename") or "").strip()
        if filename:
            resolved = self._resolve_by_filename(filename)
            if resolved:
                return resolved
        raise FileNotFoundError(
            f"Unable to resolve audio for utt_id={record.get('utt_id', '')!r}, "
            f"filename={record.get('filename', '')!r}, audio_ref={audio_ref!r}"
        )

    def _resolve_by_filename(self, filename: str) -> str:
        candidate = self._filename_index.get(filename)
        if candidate and Path(candidate).is_file():
            return candidate
        for root in self.config.local_audio_roots:
            root_path = Path(root)
            direct = root_path / filename
            if direct.is_file():
                resolved = str(direct)
                self._filename_index[filename] = resolved
                return resolved
            for suffix in _AUDIO_SUFFIXES:
                matches = list(root_path.rglob(f"{Path(filename).stem}{suffix}"))
                if matches:
                    resolved = str(matches[0])
                    self._filename_index[filename] = resolved
                    return resolved
            matches = list(root_path.rglob(filename))
            if matches:
                resolved = str(matches[0])
                self._filename_index[filename] = resolved
                return resolved
        return ""

    def stage_group(self, group_id: str, records: list[dict[str, Any]]) -> tuple[str, ...]:
        staged_paths = self._staged_groups.get(group_id)
        if staged_paths is not None:
            return staged_paths
        unique_paths: list[str] = []
        seen_paths: set[str] = set()
        for record in records:
            audio_ref = str(record.get("audio_ref") or "").strip()
            if not audio_ref.startswith("oss://"):
                continue
            local_path = str(self._local_path_for_oss_uri(audio_ref))
            if local_path in seen_paths:
                continue
            seen_paths.add(local_path)
            unique_paths.append(local_path)
            try:
                self._schedule_oss_uri(audio_ref)
            except Exception as exc:
                _LOGGER.warning("Failed to prefetch OSS audio for %s: %s", audio_ref, exc)
        staged_paths = tuple(unique_paths)
        self._staged_groups[group_id] = staged_paths
        return staged_paths

    def release_group(self, group_id: str) -> None:
        self._staged_groups.pop(group_id, None)

    def resolve_oss_uri(self, oss_uri: str) -> str:
        bucket_name, key = self._parse_oss_uri(oss_uri)
        local_path = self._local_path_for_oss_uri(oss_uri)
        expected_size = self._get_expected_object_size(bucket_name=bucket_name, key=key)
        if self._is_complete_local_file(local_path, expected_size=expected_size):
            return str(local_path)
        future = self._schedule_oss_uri(oss_uri)
        if future is not None:
            try:
                future.result(timeout=self._resolve_timeout_seconds or None)
            except FutureTimeoutError as exc:
                raise TimeoutError(
                    "Timed out waiting for OSS audio materialization: "
                    f"{oss_uri} -> {local_path} (timeout={self._resolve_timeout_seconds:.0f}s)"
                ) from exc
        if not self._is_complete_local_file(local_path, expected_size=expected_size):
            raise RuntimeError(f"Failed to materialize OSS audio into cache: {oss_uri}")
        return str(local_path)

    def _schedule_oss_uri(self, oss_uri: str) -> Future[str] | None:
        bucket_name, key = self._parse_oss_uri(oss_uri)
        local_path = self._local_path_for_oss_uri(oss_uri)
        expected_size = self._get_expected_object_size(bucket_name=bucket_name, key=key)
        if self._is_complete_local_file(local_path, expected_size=expected_size):
            return None
        self._cleanup_incomplete_local_file(local_path, expected_size=expected_size)
        cache_key = str(local_path)
        with self._prefetch_lock:
            cached_future = self._prefetch_futures.get(cache_key)
            if cached_future is not None:
                if not cached_future.done():
                    return cached_future
                if self._is_complete_local_file(local_path, expected_size=expected_size):
                    return None
                self._cleanup_incomplete_local_file(local_path, expected_size=expected_size)
            executor = self._ensure_prefetch_executor()
            future = executor.submit(self._resolve_oss_uri_sync, oss_uri)
            self._prefetch_futures[cache_key] = future
            return future

    def _ensure_prefetch_executor(self) -> ThreadPoolExecutor:
        if self._prefetch_executor is not None:
            return self._prefetch_executor
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=max(int(self.config.prefetch_workers), 1),
            thread_name_prefix="x-aut-audio-prefetch",
        )
        return self._prefetch_executor

    def _local_path_for_oss_uri(self, oss_uri: str) -> Path:
        bucket_name, key = self._parse_oss_uri(oss_uri)
        return self.cache_dir / bucket_name / key

    def _estimate_retry_budget_seconds(self) -> float:
        if self._oss_retry_attempts <= 1:
            return 0.0
        budget = 0.0
        for attempt_index in range(self._oss_retry_attempts - 1):
            delay = min(self._oss_retry_base_delay * (2**attempt_index), self._oss_retry_max_delay)
            budget += delay * 1.2
        return budget

    @staticmethod
    def _is_retryable_oss_error(exc: Exception) -> bool:

        status = getattr(exc, "status", None)
        if status in {429, 500, 502, 503, 504}:
            return True

        if isinstance(exc, (ConnectionError, ConnectionResetError, ConnectionRefusedError, BrokenPipeError)):
            return True

        code = str(getattr(exc, "code", "") or "")
        message = str(exc)
        normalized = f"{code} {message}".casefold()
        return any(
            token in normalized
            for token in (
                "partitionqpslimitted",
                "partition qps limitted",
                "please retry later",
                "too many requests",
                "temporarily unavailable",
                "timeout",
                "connection",
                "reset",
                "refused",
                "broken pipe",
                "network",
                "errno",
            )
        )

    def _with_oss_retry(self, operation: str, fn):
        last_error: Exception | None = None
        for attempt in range(1, self._oss_retry_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt >= self._oss_retry_attempts or not self._is_retryable_oss_error(exc):
                    raise
                delay = min(self._oss_retry_base_delay * (2 ** (attempt - 1)), self._oss_retry_max_delay)
                delay += random.random() * max(delay * 0.2, 0.05)
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"OSS retry wrapper exited without result for operation={operation}")

    def _resolve_oss_uri_sync(self, oss_uri: str) -> str:
        bucket_name, key = self._parse_oss_uri(oss_uri)
        local_path = self._local_path_for_oss_uri(oss_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        expected_size = self._get_expected_object_size(bucket_name=bucket_name, key=key)
        if self._is_complete_local_file(local_path, expected_size=expected_size):
            return str(local_path)
        self._cleanup_incomplete_local_file(local_path, expected_size=expected_size)



        time.sleep(random.random() * 2.0)

        lock_path = local_path.with_suffix(local_path.suffix + ".lock")
        deadline = time.monotonic() + self._lock_wait_seconds
        while time.monotonic() < deadline:
            try:
                handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(handle, f"{os.getpid()}\n{time.time():.6f}\n".encode("utf-8"))
                finally:
                    os.close(handle)
                try:
                    self._download_oss_file(
                        bucket_name=bucket_name,
                        key=key,
                        local_path=local_path,
                        expected_size=expected_size,
                    )
                finally:
                    if lock_path.exists():
                        lock_path.unlink()
                return str(local_path)
            except FileExistsError:
                if self._clear_stale_download_lock(lock_path):
                    continue
                if self._is_complete_local_file(local_path, expected_size=expected_size):
                    return str(local_path)
                self._cleanup_incomplete_local_file(local_path, expected_size=expected_size)
                time.sleep(0.25 + random.random() * 0.5)
        raise TimeoutError(f"Timed out waiting for OSS audio download lock: {oss_uri}")

    def _clear_stale_download_lock(self, lock_path: Path) -> bool:
        try:
            metadata = lock_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return False
        raw_pid = metadata[0].strip() if metadata else ""
        if raw_pid:
            try:
                pid = int(raw_pid)
            except ValueError:
                pid = None
            if pid is not None and self._process_exists(pid):
                return False
        else:
            try:
                lock_age_seconds = max(time.time() - lock_path.stat().st_mtime, 0.0)
            except OSError:
                return False
            if lock_age_seconds < self._lock_wait_seconds:
                return False
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _process_exists(pid: int) -> bool:
        if int(pid) <= 0:
            return False
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _ensure_downloader(self):
        if self._downloader is not None:
            return self._downloader
        config_path = str(self.config.oss_config_path or "").strip()
        if not config_path:
            env_path = str(os.environ.get("OSS_CONFIG_PATH", "")).strip()
            if env_path:
                config_path = env_path
        if not config_path:
            raise ValueError("OSS audio requested but data.oss_config_path / OSS_CONFIG_PATH is empty")
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"OSS config file not found: {config_file}")

        credentials = json.loads(config_file.read_text(encoding="utf-8"))
        access_key_id = str(credentials.get("access_key_id") or credentials.get("access_key") or "").strip()
        access_key_secret = str(credentials.get("access_key_secret") or credentials.get("secret_key") or "").strip()
        endpoint = str(credentials.get("endpoint") or "").strip()
        if not access_key_id or not access_key_secret or not endpoint:
            raise ValueError(
                f"OSS config must contain endpoint plus access_key_id/access_key_secret or access_key/secret_key: {config_file}"
            )
        self._downloader = _Oss2Downloader(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            endpoint=endpoint,
        )
        return self._downloader

    def _download_oss_file(
        self,
        *,
        bucket_name: str,
        key: str,
        local_path: Path,
        expected_size: int | None = None,
    ) -> None:
        downloader = self._ensure_downloader()
        bucket = downloader._get_bucket(bucket_name)
        size = int(expected_size) if expected_size is not None else self._get_expected_object_size(bucket_name=bucket_name, key=key)
        temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()
        if local_path.exists():
            local_path.unlink()
        try:
            self._with_oss_retry(
                "get_object_to_file",
                lambda: bucket.get_object_to_file(key, str(temp_path)),
            )
            actual_size = temp_path.stat().st_size if temp_path.is_file() else -1
            if actual_size != size:
                raise RuntimeError(
                    f"Downloaded OSS audio has unexpected size: oss://{bucket_name}/{key} "
                    f"(expected={size}, actual={actual_size})"
                )
            os.replace(temp_path, local_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            if local_path.exists() and not self._is_complete_local_file(local_path, expected_size=size):
                local_path.unlink()
            raise
        if not self._is_complete_local_file(local_path, expected_size=size):
            raise RuntimeError(f"Failed to download OSS audio: oss://{bucket_name}/{key}")

    def _get_expected_object_size(self, *, bucket_name: str, key: str) -> int:
        cache_key = f"{bucket_name}/{key}"
        cached = self._object_size_cache.get(cache_key)
        if cached is not None:
            return cached
        downloader = self._ensure_downloader()
        bucket = downloader._get_bucket(bucket_name)
        size = int(
            self._with_oss_retry(
                "get_object_meta",
                lambda: bucket.get_object_meta(key).content_length,
            )
        )
        self._object_size_cache[cache_key] = size
        return size

    @staticmethod
    def _is_complete_local_file(local_path: Path, *, expected_size: int) -> bool:
        return local_path.is_file() and local_path.stat().st_size == int(expected_size)

    @staticmethod
    def _cleanup_incomplete_local_file(local_path: Path, *, expected_size: int) -> None:
        if not local_path.exists():
            return
        if local_path.is_file() and local_path.stat().st_size == int(expected_size):
            return
        if local_path.is_file():
            local_path.unlink()

    @staticmethod
    def _parse_oss_uri(oss_uri: str) -> tuple[str, str]:
        if not oss_uri.startswith("oss://"):
            raise ValueError(f"Invalid OSS URI: {oss_uri!r}")
        parts = oss_uri[len("oss://") :].split("/", 1)
        bucket_name = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        if not bucket_name or not key:
            raise ValueError(f"Invalid OSS URI: {oss_uri!r}")
        return bucket_name, key

    def shutdown(self) -> None:
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=False)
            self._prefetch_executor = None
        self._prefetch_futures.clear()
        self._staged_groups.clear()


def _can_use_signal_timeout() -> bool:
    return (
        os.name == "posix"
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
    )


def _run_with_timeout(fn, *, timeout_seconds: float, description: str):
    timeout_seconds = max(float(timeout_seconds or 0.0), 0.0)
    if timeout_seconds <= 0.0 or not _can_use_signal_timeout():
        return fn()

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"{description} timed out after {timeout_seconds:g}s")

    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0 or previous_timer[1] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def load_audio_mono_16k(
    path: str,
    *,
    target_sample_rate: int = 16000,
    timeout_seconds: float = 0.0,
) -> np.ndarray:
    def _load_and_resample() -> np.ndarray:
        try:
            waveform, sample_rate = torchaudio.load(path)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "torchaudio failed to decode the audio file. Install a torchaudio "
                "build and audio backend compatible with the installed PyTorch build."
            ) from exc
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sample_rate:
            waveform = F_audio.resample(waveform, sample_rate, target_sample_rate)
        waveform = waveform.squeeze(0).to(dtype=torch.float32).contiguous()
        return waveform.cpu().numpy()

    return _run_with_timeout(
        _load_and_resample,
        timeout_seconds=timeout_seconds,
        description=f"Decoding audio {path}",
    )


def build_audio_filename_index(wav_dir: str) -> dict[str, str]:
    index: dict[str, str] = {}
    root = Path(wav_dir)
    if not root.is_dir():
        return index
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _AUDIO_SUFFIXES:
            continue
        index.setdefault(path.name, str(path))
        index.setdefault(path.stem, str(path))
    return index


def evict_cached_audio_paths(paths: list[str] | tuple[str, ...], *, cache_root: str) -> int:
    resolved_root = Path(cache_root).expanduser().resolve()
    deleted = 0
    seen_paths: set[Path] = set()
    for raw_path in paths:
        candidate = Path(str(raw_path)).expanduser()
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            continue
        if resolved.is_file():
            try:
                resolved.unlink()
                deleted += 1
            except (OSError, NotADirectoryError):
                pass
        try:
            _remove_empty_parents(resolved.parent, stop_at=resolved_root)
        except OSError:
            pass
    return deleted


def cleanup_cache_root(cache_root: str) -> None:
    resolved_root = Path(cache_root).expanduser().resolve()
    if resolved_root.is_dir():
        shutil.rmtree(resolved_root, ignore_errors=True)


def _estimate_lock_wait_seconds() -> float:
    """Compute the same stale-lock threshold used by AudioResolver."""
    configured = float(os.environ.get("X_AUT_OSS_LOCK_WAIT_SECONDS", "0") or 0.0)
    if configured > 0.0:
        return configured
    retry_attempts = max(int(os.environ.get("X_AUT_OSS_RETRY_ATTEMPTS", "8") or 8), 1)
    base_delay = max(float(os.environ.get("X_AUT_OSS_RETRY_BASE_DELAY", "0.5") or 0.5), 0.0)
    max_delay = max(float(os.environ.get("X_AUT_OSS_RETRY_MAX_DELAY", "8.0") or 8.0), base_delay)
    budget = 0.0
    for attempt_index in range(retry_attempts - 1):
        delay = min(base_delay * (2**attempt_index), max_delay)
        budget += delay * 1.2
    return budget + 60.0


def cleanup_stale_download_locks(cache_root: str, lock_wait_seconds: float | None = None) -> int:
    """Remove stale .lock files under cache_root before training starts.

    A lock is considered stale if:
      - it stores a PID and that process no longer exists; or
      - it stores no usable PID and its mtime is older than lock_wait_seconds.

    Returns the number of deleted lock files.
    """
    resolved_root = Path(cache_root).expanduser().resolve()
    if not resolved_root.is_dir():
        return 0
    threshold = lock_wait_seconds if lock_wait_seconds is not None and lock_wait_seconds > 0 else _estimate_lock_wait_seconds()
    deleted = 0
    for lock_path in resolved_root.rglob("*.lock"):
        try:
            if not lock_path.is_file():
                continue
            metadata = lock_path.read_text(encoding="utf-8").splitlines()
            raw_pid = metadata[0].strip() if metadata else ""
            pid = int(raw_pid) if raw_pid else None
            if pid is not None and pid > 0:
                try:
                    os.kill(pid, 0)

                    continue
                except ProcessLookupError:
                    pass
                except PermissionError:

                    continue
            else:

                age = max(time.time() - lock_path.stat().st_mtime, 0.0)
                if age < threshold:
                    continue
            lock_path.unlink()
            deleted += 1
        except (OSError, ValueError):
            continue
    return deleted


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    while True:
        if current == stop_at:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
