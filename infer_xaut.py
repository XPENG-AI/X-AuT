#!/usr/bin/env python3
"""Standalone inference script for X-AuT-14layer (https://huggingface.co/X-AuT/X-AuT).

Follows the official Qwen3-ASR inference recipe:
  1. build the prompt via chat template, appending the language-control suffix
  2. encode audio + text with Qwen3ASRProcessor
  3. call thinker.generate(**inputs, max_new_tokens=...)
  4. decode the generated transcription text

Dependencies: torch, torchaudio, transformers, qwen-asr
(audio is automatically resampled to 16 kHz mono).

Usage::

    # auto-download the released checkpoint from the Hugging Face Hub
    python infer_xaut.py --audio /path/to/test.wav --lang-code zh

    # or point to a local copy of the checkpoint repo
    python infer_xaut.py --audio /path/to/test.wav --model /path/to/X-AuT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "X-AuT/X-AuT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X-AuT-14layer standalone wav-to-text inference")
    parser.add_argument("--audio", required=True, help="Input audio path (wav/mp3/flac ...), resampled to 16 kHz mono")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local directory of the released checkpoint")
    parser.add_argument("--lang-code", default="zh", choices=["zh", "en"], help="Language control prefix")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--attn-implementation", default="sdpa", help="sdpa / flash_attention_2 / eager")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model weight dtype")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of generated tokens")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam size (greedy by default)")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate")
    return parser.parse_args()


def resolve_dtype(dtype_str: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def resolve_source(source: str) -> str:
    """Return a local directory for the checkpoint, downloading from the Hub if needed."""
    if Path(source).exists():
        return source
    from huggingface_hub import snapshot_download

    return snapshot_download(source)


def load_audio_mono_16k(path: str, sr: int = 16000):
    import numpy as np
    import torch
    import torchaudio
    import torchaudio.functional as audio_functional

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != sr:
        waveform = audio_functional.resample(waveform, sample_rate, sr)
    return waveform.squeeze(0).to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)


def build_prompt(processor: Any, lang_code: str) -> str:
    """Chat-template prompt + language-control suffix, matching the official recipe."""
    language = "Chinese" if lang_code == "zh" else "English"

    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base_prompt = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=False,
    )
    return base_prompt + f"language {language}<asr_text>"


def main() -> int:
    args = parse_args()

    import torch
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype)
    source = resolve_source(args.model)

    processor = Qwen3ASRProcessor.from_pretrained(source, fix_mistral_regex=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        source,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    waveform = load_audio_mono_16k(args.audio, sr=args.sr)
    duration_sec = float(waveform.shape[0]) / float(args.sr)

    prompt_text = build_prompt(processor, args.lang_code)
    inputs = processor(
        text=[prompt_text],
        audio=[waveform],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device).to(dtype)
    prompt_len = int(inputs["input_ids"].size(1))

    thinker = model.thinker
    eos_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        generated = thinker.generate(
            **inputs,
            max_new_tokens=max(int(args.max_new_tokens), 1),
            num_beams=max(int(args.num_beams), 1),
            eos_token_id=eos_token_id,
            do_sample=False,
        )

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    new_token_ids = sequences[:, prompt_len:]
    raw_text = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]



    normalized_text = raw_text.strip()

    print(
        json.dumps(
            {
                "audio": args.audio,
                "lang_code": args.lang_code,
                "duration_sec": round(duration_sec, 3),
                "raw_pred_text": raw_text,
                "pred_text": normalized_text,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
