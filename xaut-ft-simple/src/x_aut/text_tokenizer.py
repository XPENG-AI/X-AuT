from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


_SPACE_RE = re.compile(r"\s+")
_CJK_CHAR_CLASS = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_PUNCT_CHARS = "，。！？；：、“”‘’（）《》〈〉【】『』「」—…·"
_CJK_PUNCT_CLASS = re.escape(_CJK_PUNCT_CHARS)
_ZH_COMPACT_PATTERNS = (
    re.compile(rf"(?<=[{_CJK_CHAR_CLASS}])\s+(?=[{_CJK_CHAR_CLASS}])"),
    re.compile(rf"(?<=[{_CJK_CHAR_CLASS}])\s+(?=[{_CJK_PUNCT_CLASS}])"),
    re.compile(rf"(?<=[{_CJK_PUNCT_CLASS}])\s+(?=[{_CJK_CHAR_CLASS}])"),
    re.compile(rf"(?<=[{_CJK_PUNCT_CLASS}])\s+(?=[{_CJK_PUNCT_CLASS}])"),
)
_ZH_DATASETS = frozenset(
    {
        "AISHELL-1_Chinese",
        "CommonVoice_Chinese",
        "kespeech",
        "acgn_zh",
        "ZHvoice_Chinese",
        "Emilia-YODAS_Chinese",
    }
)
_EN_DATASETS = frozenset({"CommonVoice_English", "Emilia-YODAS_English", "acgn_en", "gigaspeech_en"})


class TextTokenizer(Protocol):
    base_vocab_size: int
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None

    def infer_lang(self, dataset_key: str) -> str: ...

    def normalize_text(
        self,
        text: str,
        *,
        dataset_key: str | None = None,
        lang: str | None = None,
    ) -> str: ...

    def encode(
        self,
        text: str,
        *,
        dataset_key: str | None = None,
        lang: str | None = None,
        already_normalized: bool = False,
    ) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def vocab_meta(self) -> dict[str, int | str | bool]: ...


def infer_lang(
    dataset_key: str,
    *,
    zh_datasets: frozenset[str] = _ZH_DATASETS,
    en_datasets: frozenset[str] = _EN_DATASETS,
) -> str:
    if dataset_key in zh_datasets:
        return "zh"
    if dataset_key in en_datasets:
        return "en"
    raise KeyError(f"Unsupported dataset_key for language inference: {dataset_key!r}")


def normalize_text(
    text: str,
    *,
    dataset_key: str | None = None,
    lang: str | None = None,
    zh_datasets: frozenset[str] = _ZH_DATASETS,
    en_datasets: frozenset[str] = _EN_DATASETS,
) -> str:
    if text is None:
        raise ValueError("text must not be None")
    resolved_lang = lang or (infer_lang(dataset_key, zh_datasets=zh_datasets, en_datasets=en_datasets) if dataset_key is not None else None)
    if resolved_lang not in {"zh", "en", "de", "fr"}:
        raise ValueError(f"Unsupported language: {resolved_lang!r}")
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u3000", " ").strip()
    normalized = _SPACE_RE.sub(" ", normalized)
    if resolved_lang == "zh":
        for pattern in _ZH_COMPACT_PATTERNS:
            normalized = pattern.sub("", normalized)
    normalized = normalized.strip()
    if not normalized:
        raise ValueError(f"text becomes empty after normalization: {text!r}")
    return normalized


def _hash_files(paths: list[Path], *, fallback: str) -> str:
    hasher = hashlib.sha1()
    hasher.update(fallback.encode("utf-8"))
    saw_file = False
    for path in paths:
        if not path.is_file():
            continue
        saw_file = True
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                if not chunk:
                    break
                hasher.update(chunk)
    digest = hasher.hexdigest()
    return digest[:16] if saw_file else hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:16]


class QwenBPETokenizer:
    def __init__(
        self,
        source: str,
        *,
        trust_remote_code: bool = False,
        local_files_only: bool | None = None,
        fix_mistral_regex: bool = True,
        zh_datasets: frozenset[str] = _ZH_DATASETS,
        en_datasets: frozenset[str] = _EN_DATASETS,
    ) -> None:
        raw_source = str(source).strip()
        if not raw_source:
            raise ValueError("QwenBPETokenizer source must not be empty")
        source_path = Path(raw_source)
        tokenizer_json_path = source_path / "tokenizer.json"
        if local_files_only is None:
            local_files_only = source_path.exists()
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        self.source = raw_source
        self.zh_datasets = zh_datasets
        self.en_datasets = en_datasets
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self.fix_mistral_regex = bool(fix_mistral_regex)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.source,
                use_fast=True,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                fix_mistral_regex=self.fix_mistral_regex,
            )
        except Exception:
            if not tokenizer_json_path.is_file():
                raise
            self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json_path))
        vocab = self._tokenizer.get_vocab()
        if not vocab:
            raise ValueError(f"Tokenizer vocabulary is empty: {self.source}")
        self.base_vocab_size = max(int(token_id) for token_id in vocab.values()) + 1
        self.tokenizer_class = type(self._tokenizer).__name__
        self.tokenizer_name_or_path = str(getattr(self._tokenizer, "name_or_path", self.source))
        self.pad_token_id = None if self._tokenizer.pad_token_id is None else int(self._tokenizer.pad_token_id)
        self.bos_token_id = None if self._tokenizer.bos_token_id is None else int(self._tokenizer.bos_token_id)
        self.eos_token_id = None if self._tokenizer.eos_token_id is None else int(self._tokenizer.eos_token_id)
        self.tokenizer_fingerprint = _hash_files(
            [
                source_path / "tokenizer.json",
                source_path / "tokenizer_config.json",
                source_path / "special_tokens_map.json",
            ],
            fallback=self.source,
        )

    def infer_lang(self, dataset_key: str) -> str:
        return infer_lang(dataset_key, zh_datasets=self.zh_datasets, en_datasets=self.en_datasets)

    def normalize_text(
        self,
        text: str,
        *,
        dataset_key: str | None = None,
        lang: str | None = None,
    ) -> str:
        return normalize_text(
            text,
            dataset_key=dataset_key,
            lang=lang,
            zh_datasets=self.zh_datasets,
            en_datasets=self.en_datasets,
        )

    def encode(
        self,
        text: str,
        *,
        dataset_key: str | None = None,
        lang: str | None = None,
        already_normalized: bool = False,
    ) -> list[int]:
        normalized = text if already_normalized else self.normalize_text(text, dataset_key=dataset_key, lang=lang)
        token_ids = list(self._tokenizer.encode(normalized, add_special_tokens=False))
        if not token_ids:
            raise ValueError(f"text produced no Qwen BPE tokens: {normalized!r}")
        return [int(token_id) for token_id in token_ids]

    def decode(self, token_ids: list[int]) -> str:
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.base_vocab_size:
                raise ValueError(f"token id out of range [0, {self.base_vocab_size - 1}]: {token_id}")
        return str(
            self._tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def vocab_meta(self) -> dict[str, int | str | bool]:
        return {
            "type": "qwen_hf",
            "source": self.source,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "local_files_only": self.local_files_only,
            "base_vocab_size": self.base_vocab_size,
            "pad_token_id": self.pad_token_id if self.pad_token_id is not None else -1,
            "bos_token_id": self.bos_token_id if self.bos_token_id is not None else -1,
            "eos_token_id": self.eos_token_id if self.eos_token_id is not None else -1,
            "fix_mistral_regex": self.fix_mistral_regex,
        }
