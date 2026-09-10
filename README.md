<p align="center">
  <img src="xpomni_logo.png" alt="XPeng Omni Team" width="320"/>
</p>

<h1 align="center">X-AuT</h1>

<p align="center">
  <b>Progressive Audio-Encoder Compression for Speech LLMs with Cross-Scale Distillation</b><br/>
  XPeng Inc. &nbsp;|&nbsp; 🌐 <a href="https://xpeng-ai.github.io/x-aut">Project Page</a> &nbsp;|&nbsp; 🤗 <a href="https://huggingface.co/X-AuT/X-AuT">Model Weights</a> &nbsp;|&nbsp; <a href="README_zh.md">【中文说明】</a>
</p>

---

## 📖 Introduction

**X-AuT-14layer** is a compressed variant of [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). It reduces the audio tower from 18 to 14 Transformer blocks and from **186.376M to 147.794M parameters (−20.70%)**. The pretrained language-model backbone remains frozen; decoder attention is adapted with LoRA, and the tied output embedding is trainable during distillation and frozen during final finetuning.

Layer removal changes the audio embeddings consumed by the decoder and can cause premature end-of-sequence prediction and severe deletion errors. X-AuT treats this mismatch as an important recovery target while recognizing that retained encoder capacity also matters. The framework combines:

- **Behavior-driven layer screening.** Short-budget recovery probes compare candidates on fixed development/validation subsets. X-AuT prunes progressively, 18 → 16 by removing original layers {1, 18}, then 16 → 14 by removing the selected pair {5, 6}. The tested pair interactions are non-additive; the results motivate explicit pair probing, not a universal rule that adjacent layers are always safer.
- **Transcript-consistency filtering.** A source pool exceeding 280k hours is ranked into nine confidence classes using the source transcript and two offline ASR hypotheses. The reported runs use **class 1 in all three stages**; Stage 2 changes source weights toward target-domain data. The paper does not claim a validated multi-tier curriculum effect.
- **Three-stage recovery.** Stage 0 aligns intermediate/bridge representations and logits; Stage 1 combines teacher-forced distillation with scheduled student-policy contexts; Stage 2 performs lower-rate, gold-transcript LoRA finetuning.
- **Cross-scale distillation.** A frozen [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) teacher supervises the student through learned 2048→1024 projections and is discarded at inference. Under the matched 16-layer Stage 1 recipe, the cross-scale teacher reaches 5.55% macro error versus 8.45% for a same-scale self-teacher. This is a descriptive single-run comparison, not causal proof of capability creation.

Across ten public Chinese-English benchmarks, the 16-layer Stage 2 model reduces macro-average error from **5.61% to 5.27% (−6.1% relative)**. The 14-layer Stage 2 model reaches **5.75% (+2.5% relative)** while reducing audio-tower parameters by 20.70%. The 14-layer encoder latency is reduced by **21.4% on the in-vehicle PPU** and **11.4% on H800**.

## 🏗️ Model Architecture

Only Transformer blocks in the audio tower are removed. ConvStem, Bridge, and `ln_post` retain their structure. The language-model base weights are not pruned; LoRA adapters are added to its attention projections during recovery.

### AuT (audio encoder) configuration

| Item | Value |
|------|-------|
| model_type | `qwen3_asr_audio_encoder` |
| d_model | 896 |
| Attention heads | 14 |
| FFN dim | 3584 |
| Mel bins | 128 |
| Output dim (bridge → LLM hidden) | 1024 |
| Encoder layers (original / this model) | 18 / **14** |
| Retained original layers (1-based) | `[2,3,4,7,8,9,10,11,12,13,14,15,16,17]` (drop {1,18}, then {5,6}) |

### Parameter breakdown

| Module | AuT-18 (baseline) | AuT-14 (this model) |
|--------|------------------:|--------------------:|
| ConvStem (conv2d1/2/3 + conv_out) | 11.03M | 11.03M |
| Transformer encoder layers | 173.62M (18 layers) | **135.04M (14 layers)** |
| Bridge (proj1 + proj2) | 1.72M | 1.72M |
| ln_post | 0.002M | 0.002M |
| **Audio tower total** | **≈186.38M** | **≈147.79M (↓20.7%)** |

### Whole-model parameters (for reference)

| Module | Params | Note |
|--------|-------:|------|
| Audio tower (AuT) | 147.794M | pruned (186.376M → 147.794M) |
| Text decoder (28 layers) | 440.47M | Qwen3-0.6B, frozen |
| Token embedding | 155.58M | tied with `lm_head` |
| **Total (tied weights deduplicated)** | **≈743.84M** | — |

## 📊 Evaluation

Error rates are CER for Chinese and WER for English; lower is better. The paper's aggregate metric is the unweighted macro mean across the ten benchmarks. All values are single-run best-checkpoint results, and boldface does not indicate statistical significance.

### Accuracy–compression tradeoff

| Model | Audio-tower params | Param. change | Macro mean error | Relative mean-error change |
|-------|-------------------:|--------------:|-----------------:|---------------------------:|
| Full-18 baseline | 186.376M | — | 5.61 | — |
| X-AuT-16layer (Stage 2) | 167.085M | −10.35% | **5.27** | **−6.1%** |
| X-AuT-14layer (Stage 2) | 147.794M | −20.70% | 5.75 | +2.5% |

### X-AuT-14layer results

| Benchmark | Full-18 baseline | X-AuT-14layer (Stage 1) | X-AuT-14layer (Stage 2) |
|-----------|:---:|:---:|:---:|
| AISHELL-1 (CER) | **3.33%** | 3.52% | 3.39% |
| Fleurs-zh (CER) | **2.80%** | 3.49% | 3.32% |
| Fleurs-en (WER) | **4.17%** | 5.24% | 5.10% |
| LibriSpeech test-clean (WER) | 2.48% | 3.09% | **2.45%** |
| THCHS-30 (CER) | **3.87%** | 4.23% | 4.17% |
| Tedlium (WER) | **3.35%** | 4.07% | 3.95% |
| LibriSpeech test-other (WER) | **5.39%** | 7.00% | 5.52% |
| CommonVoice v15 zh (CER) | 9.95% | 9.54% | **8.36%** |
| CommonVoice v15 en (WER) | **12.35%** | 13.94% | 12.49% |
| WenetSpeech-meeting (CER) | **8.36%** | 10.71% | 8.78% |
| *Macro mean (%)* | **5.61** | 6.48 | 5.75 |
| *Relative mean-error change (%)* | — | +15.5 | +2.5 |

The finetuned 14-layer model improves LibriSpeech test-clean and CommonVoice zh, while the other eight benchmark values are higher than the baseline. Its largest degradation is +0.93 percentage points on Fleurs-en. These differences are descriptive because repeated seeds and utterance-level confidence intervals are not yet available.

### Inference efficiency

| Metric | PPU (in-vehicle) | GPU (H800) |
|--------|:---:|:---:|
| Encoder latency vs AuT-18 | **↓21.4%** | **↓11.4%** |
| End-to-end latency vs AuT-18 | ↓4.7% | ↓2.6% |
| Peak memory vs AuT-18 | ↓4.4% | ↓2.8% |

> End-to-end improvement is modest because the unpruned 28-layer text decoder dominates total inference time. These measurements are averages over more than 50 utterances; run-to-run variance was not retained.

## 📦 Released Scope

This repository releases the components needed to use the 14-layer checkpoint and adapt it with a compact LoRA recipe:

| Component | Included | Scope |
|-----------|:--------:|-------|
| Standalone checkpoint inference | ✅ | [`infer_xaut.py`](infer_xaut.py) |
| Minimal LoRA finetuning | ✅ | [`xaut-ft-simple/`](xaut-ft-simple/) |
| Example manifests | ✅ | 1,000 metadata records with placeholder audio references; audio is not redistributed |
| Behavior-probe pipeline | ❌ | Not part of this code release |
| Stage 0/1 cross-scale distillation | ❌ | Not part of this code release |
| 280k-hour data/labeling pipeline and proprietary data | ❌ | Described in the paper but not redistributed |

The minimal finetuning directory is a practical **Stage-2-style adaptation example**. It is not a complete reproduction package for the paper's three-stage training pipeline or reported benchmark tables.

## 🚀 Inference

Released weights are hosted at 🤗 [X-AuT/X-AuT](https://huggingface.co/X-AuT/X-AuT) as a **full fine-tuned model** in safetensors format (audio tower already at 14 layers) — it loads directly, with no base-model download or manual layer pruning.

We provide a standalone inference script [`infer_xaut.py`](infer_xaut.py) that does **not** depend on the X-AuT training codebase — only on `torch`, `torchaudio`, `transformers`, `qwen-asr`, and `huggingface_hub` (audio is automatically converted to 16 kHz mono). The inference logic follows the official recipe: chat-template prompt with a language-control suffix → `processor` encoding → `thinker.generate()` → text decoding.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Install the PyTorch/torchaudio build for your accelerator first, then:
pip install -r requirements.txt
```

Validated dependency versions are pinned in [`requirements.txt`](requirements.txt). The repository does not pin PyTorch because CUDA, PPU, and CPU environments require different vendor builds.

### Quickstart

```bash
# checkpoint is auto-downloaded from https://huggingface.co/X-AuT/X-AuT
python infer_xaut.py \
  --audio /path/to/test.wav \
  --lang-code zh \
  --device cuda
```

Or load the model directly in Python:

```python
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

processor = Qwen3ASRProcessor.from_pretrained("X-AuT/X-AuT", fix_mistral_regex=True)
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    "X-AuT/X-AuT", dtype="bfloat16", device_map="cuda",
)
# then follow the standard Qwen3-ASR generation recipe
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | *(required)* | Input audio path (wav/mp3/flac…), auto-resampled to 16 kHz mono |
| `--model` | `X-AuT/X-AuT` | HF repo id or a local directory of the checkpoint |
| `--lang-code` | `zh` | Language control prefix (`zh` / `en`) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--attn-implementation` | `sdpa` | `sdpa` / `flash_attention_2` / `eager` |
| `--dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--max-new-tokens` | `256` | Maximum number of generated tokens |
| `--num-beams` | `1` | Beam size (greedy by default) |
| `--sr` | `16000` | Target sampling rate |

### Output

One line of JSON per run:

```json
{
  "audio": "/path/to/test.wav",
  "lang_code": "zh",
  "duration_sec": 5.14,
  "raw_pred_text": "原始解码文本",
  "pred_text": "归一化后的识别文本"
}
```

- `raw_pred_text`: raw decoded string from the decoder;
- `pred_text`: normalized transcription produced by `qwen_asr.inference.utils.parse_asr_output`.

## 🎛️ Fine-tuning

We provide a minimal, self-contained LoRA fine-tuning recipe in [`xaut-ft-simple/`](xaut-ft-simple/). It contains a compact training script, an example configuration, and 1,000 metadata records (with placeholder audio references), and does not depend on the private X-AuT training codebase. It trains the audio encoder, bridge, and decoder LoRA adapters with transcript cross-entropy; it does not implement the paper's Stage 0/1 teacher losses or behavior-driven pruning probes. See [`xaut-ft-simple/README.md`](xaut-ft-simple/README.md) for usage, data format, and configuration details.

Quick start:

```bash
pip install -r xaut-ft-simple/requirements.txt
cd xaut-ft-simple
bash run_ft.sh
```

Before preparing a dataset, return to the repository root and validate model loading, audio preprocessing, forward/backward, and one optimizer step with the generated license-free sample:

```bash
python xaut-ft-simple/scripts/smoke_train.py --device-type cuda
```

The recorded PPU code-path check and its explicit full-checkpoint boundary are documented in [`xaut-ft-simple/VALIDATION.md`](xaut-ft-simple/VALIDATION.md).

## 📁 Checkpoint

The released checkpoint is a complete model bundle on 🤗 [X-AuT/X-AuT](https://huggingface.co/X-AuT/X-AuT): full fine-tuned weights (audio tower already at 14 layers) plus `config.json`, `generation_config.json`, tokenizer, and preprocessor — so no extra configuration file is needed at inference time. License: [CC BY-NC 4.0](LICENSE).

The model weights and example data are provided for research and evaluation purposes only. Commercial use, production deployment, resale, sublicensing, redistribution, or use to train or improve commercial products or services is prohibited without prior written permission from XPeng Inc.

## ⚠️ Reproducibility Notes

- The public benchmark values are single-run results selected by small fixed development/validation subsets; they should not be interpreted as statistically significant.
- The released LoRA example does not reproduce the private data mixture, cross-scale teacher training, checkpoint selection, or hardware measurements used in the paper.
- The demo JSONL contains transcripts and placeholder audio references only. Replace them with audio that you are licensed to use and update `data/train_index.json` if you change its location.
- The 280k-hour multi-system agreement pipeline used for model training consists of internally authorized, proprietary, or otherwise properly licensed data. This training data is not released with this repository. No third-party audio, transcripts, or datasets are redistributed as part of this release.
- GPU memory and throughput depend on the local PyTorch, CUDA, attention backend, audio duration, and batch configuration.

---

## 📚 Reference

If you find X-AuT useful in your research, please consider citing our paper:

```bibtex
@article{zhang2026xaut,
  title   = {X-AuT: Progressive Audio-Encoder Compression for
             Speech LLMs with Cross-Scale Distillation},
  author  = {Zhang, Haojun and Zou, Yi and Chen, Min and Yu, Qize and
             Fan, Lianrui and Ding, Xini and Zhou, Shuchang and
             Liu, Xianming and Huang, Shiyu},
  journal = {Preprint},
  year    = {2026},
  url     = {https://x-aut.github.io/}
}
```

---

<p align="center">
  © 2026 XPeng Inc.
</p>
