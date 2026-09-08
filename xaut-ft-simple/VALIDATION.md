# Training-path validation

The public training path has been validated with Python 3.10, a hardware-compatible PyTorch and torchaudio installation, and the dependencies listed in `requirements.txt`.

## Validation procedure

The smoke test uses a small synthetic manifest and exercises configuration loading, audio decoding and resampling, dataset construction, model initialization, one optimization step, checkpoint writing, and checkpoint reload. Run it with:

```bash
python scripts/smoke_train.py
```

For full training, prepare a JSONL manifest as described in the README, update `configs/ft_simple.yaml`, and run:

```bash
bash run_ft.sh
```

## Scope

The smoke test verifies that the released training pipeline is executable. It does not reproduce the complete paper experiments, which require the public upstream checkpoint, the paper datasets, and the corresponding compute budget.
