from __future__ import annotations

import re

SUPPORTED_TARGET_FORMATS = ("plain_text", "qwen_asr")
QWEN_ASR_LANG_NAMES = {"zh": "Chinese", "en": "English", "de": "German", "fr": "French"}
QWEN_ASR_TEXT_MARKER = "<asr_text>"
_QWEN_LANGUAGE_PREFIX_RE = re.compile(r"^language\s+([A-Za-z_]+)", re.IGNORECASE)
_QWEN_SPECIAL_TOKEN_RE = re.compile(r"^<\|[^>]+\|>")


def _is_known_qwen_language_name(language_name: str) -> bool:
    normalized = str(language_name or "").strip().casefold()
    if not normalized:
        return False
    for known_name in QWEN_ASR_LANG_NAMES.values():
        known = known_name.casefold()
        if normalized == known or normalized.startswith(f"{known}_"):
            return True
    return False


def strip_qwen_asr_control_prefix(text: str) -> str:
    candidate = str(text or "")
    while True:
        updated = candidate.lstrip()
        changed = updated != candidate
        candidate = updated
        if candidate.startswith(QWEN_ASR_TEXT_MARKER):
            candidate = candidate[len(QWEN_ASR_TEXT_MARKER) :]
            continue
        prefix_match = _QWEN_LANGUAGE_PREFIX_RE.match(candidate)
        if prefix_match is not None and _is_known_qwen_language_name(prefix_match.group(1)):
            candidate = candidate[prefix_match.end() :]
            continue
        special_match = _QWEN_SPECIAL_TOKEN_RE.match(candidate)
        if special_match is not None:
            candidate = candidate[special_match.end() :]
            continue
        if not changed:
            break
    return candidate.lstrip()


def build_target_text(text: str, *, lang: str, target_format: str) -> str:
    normalized_format = str(target_format).strip().lower()
    if normalized_format == "plain_text":
        return text
    if normalized_format == "qwen_asr":
        language_name = QWEN_ASR_LANG_NAMES.get(lang)
        if language_name is None:
            raise ValueError(f"Unsupported language for qwen_asr target: {lang!r}")
        return f"language {language_name}{QWEN_ASR_TEXT_MARKER}{text}"
    raise ValueError(f"Unsupported target_format: {target_format!r}")
