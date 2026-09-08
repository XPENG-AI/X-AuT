# X-AuT LoRA Finetune

A minimal, self-contained LoRA finetune implementation for [X-AuT](https://github.com/Elgin-zhj/X_AuT) models
(weights: [X-AuT/X-AuT](https://huggingface.co/X-AuT/X-AuT) on the Hugging Face Hub).

## Overview

This directory provides a streamlined, Stage-2-style adaptation pipeline for X-AuT with LoRA. It includes:

- **Self-contained code**: the modules required by this finetuning entry point are vendored in `src/x_aut/`
- **Compact training entry point**: transcript cross-entropy training with DDP, gradient accumulation, checkpointing, and cosine decay
- **Example metadata**: 1,000 fully synthetic transcript records (500 Chinese + 500 English) with placeholder audio references

> **Scope:** this example trains the audio encoder, bridge, and decoder LoRA adapters. It does not include the paper's behavior-driven pruning probes, Stage 0/1 cross-scale teacher losses, private data mixture, checkpoint-selection suite, or benchmark evaluation. It therefore does not reproduce the paper's reported results by itself.

## Directory Structure

```
xaut-ft-simple/
├── train.py                 # Training script
├── run_ft.sh                # Training launcher (torchrun)
├── requirements.txt         # Validated training dependencies
├── scripts/
│   └── smoke_train.py       # Generated-audio, one-step training check
├── configs/
│   └── ft_simple.yaml       # Training configuration
├── data/
│   ├── train.jsonl          # Example training data (1000 samples)
│   └── train_index.json     # Training index (train_shards format)
└── src/
    └── x_aut/               # Vendored x_aut library (10 modules)
```

## Requirements

- Python 3.10+
- A hardware-compatible PyTorch/torchaudio build (PyTorch 2.0+)
- CUDA-compatible accelerator with sufficient memory for training
- Optional: `oss2` (only when training data uses `oss://` URIs)

Install PyTorch and torchaudio from the channel supplied for your accelerator, then install the validated upper-layer dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r xaut-ft-simple/requirements.txt  # run from repository root
# pip install oss2                              # optional OSS support
```

The requirements pin `qwen-asr==0.0.6`, its matching `transformers==4.57.6`, and `accelerate==1.12.0`. PyTorch itself is deliberately not pinned because CUDA, PPU, and CPU systems need different builds.

## Usage

### One-step smoke test

The smoke test creates a one-second synthetic WAV and a one-record manifest in a temporary directory, loads the released checkpoint, and requires one forward pass, backward pass, and optimizer update to complete. It disables final checkpoint saving so a validation run does not duplicate the full model state.

```bash
python xaut-ft-simple/scripts/smoke_train.py --device-type cuda
# Reuse an offline model and retain artifacts at a chosen location:
python xaut-ft-simple/scripts/smoke_train.py \
  --model /path/to/X-AuT \
  --work-dir /tmp/xaut-smoke \
  --device-type cuda
```

Success requires a finite logged loss, `Training finished | final_step=1`, and no unexpected full checkpoint. This verifies code-path feasibility, not convergence or reproduction of paper metrics. See [`VALIDATION.md`](VALIDATION.md) for the environment, executed check, result, and scope boundary.

### Prepare audio

The included JSONL contains fully synthetic transcripts (no real speech data); its `audio_path` values are placeholders, so the default command will fail until matching audio is provided. Use one of the following approaches:

1. Replace each `audio_path` with an absolute path or a path relative to `xaut-ft-simple/`.
2. Keep the filenames and add one or more searchable directories to `data.local_audio_roots` in `configs/ft_simple.yaml`.
3. Use `oss://` references and configure `data.oss_config_path` if you have your own OSS credentials and licensed data.

### Training

Single GPU:

```bash
bash run_ft.sh
```

Multi-GPU (8 GPUs):

```bash
GPUS=8 bash run_ft.sh
```

Override the configuration or output directory without editing the launcher:

```bash
CONFIG=/path/to/config.yaml bash run_ft.sh
PYTHONPATH=src python train.py --config configs/ft_simple.yaml --output-dir /path/to/output
```

Training checkpoints are saved to `output/checkpoints/latest.pt` when `train.save_final_checkpoint` is enabled. They contain full training state for resuming this script; they are not automatically exported as a merged Hugging Face checkpoint for `infer_xaut.py`.

### Configuration

Edit `configs/ft_simple.yaml` to customize:

- `model.qwen_source` / `tokenizer.source`: Base model location — the default `X-AuT/X-AuT` downloads the released X-AuT-14layer weights from the Hugging Face Hub; set a local directory path (and `local_files_only: true`) for offline use
- `model.local_files_only` / `tokenizer.local_files_only`: `false` to allow downloading from the Hub
- `train.epochs`: Number of training epochs (default: 2.0)
- `train.per_device_batch_size`: Batch size per GPU (default: 4)
- `train.grad_accum_steps`: Gradient accumulation steps (default: 2)
- `train.lr`: Learning rate (default: 5e-6)
- `data.persistent_workers`: Keep dataloader workers alive between epochs when `num_workers > 0` (default: `true`)
- `train.save_final_checkpoint`: Save `latest.pt` after training (default: `true`; smoke tests set it to `false`)
- `lora.r/alpha`: LoRA parameters (r=32, alpha=64)
- `trainable.*`: Trainable parameter control

## Data Format

Training data uses JSONL format with `train_shards` indexing. Each record:

```json
{
  "utt_id": "1",
  "audio_path": "0001.wav",
  "target_text": "language Chinese<asr_text>transcription text",
  "text": "transcription text",
  "lang": "zh",
  "duration_sec": 6.6
}
```

The `train_index.json` format:
```json
{
  "train_shards": [
    {"path": "data/train.jsonl", "count": 1000}
  ]
}
```

**Note**: `audio_ref`, `wav_path`, and several other local-path fields are also accepted by the loader, but `audio_path` is the documented portable field. The included `data/train.jsonl` contains fully synthetic transcripts with placeholder audio paths; replace or resolve them before training. If you move the JSONL, update the relative path in `data/train_index.json`.

## Key Design Decisions

- **No evaluation during training**: the loop logs training loss but does not compute CER/WER or select checkpoints on the paper's development suite
- **Persistent cache mode**: resolved audio is cached locally for multi-epoch training
- **LoRA on decoder**: applied to q/k/v/o projections (r=32, alpha=64, dropout=0.05)
- **Trainable components**: audio encoder, bridge/`ln_post`, and LoRA adapters
- **Frozen components**: decoder base weights, decoder norm, `lm_head`, and tied output embedding
- **Different scale from the paper run**: the example defaults to per-device batch 4 and gradient accumulation 2; the paper's main Stage 2 run uses per-device batch 8, accumulation 2, and global batch 512

## Checkpoint Format

Saved checkpoints contain:

- `model`: full model state dict, including LoRA parameters
- `optimizer`: optimizer state
- `scheduler`: scheduler state
- `config`: resolved training configuration
- `step`: global optimizer step
- `epoch`: zero-based epoch index

For deployment, add an explicit export step that writes a Hugging Face-compatible directory and verify it with `infer_xaut.py`. This minimal release currently provides training-state checkpoints only.

## License

CC BY-NC 4.0, following the repository-level [LICENSE](../LICENSE).

## Citation

See the [project page](https://x-aut.github.io/) for citation information.
