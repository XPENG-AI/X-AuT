#!/usr/bin/env python3
"""X-AuT 最简 LoRA finetune（自包含，不依赖 X-AuT 仓库）。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from x_aut.data import (
    DatasetRuntimeConfig,
    build_train_dataloader,
    build_train_dataset,
    estimate_train_steps,
    load_train_index,
)
from x_aut.model import build_model_from_config
from x_aut.utils import (
    autocast_context,
    barrier,
    cleanup_distributed,
    count_parameters,
    count_trainable_parameters,
    deep_get,
    ensure_dir,
    format_parameter_count,
    init_distributed,
    is_main_process,
    load_config,
    move_batch_to_device,
    resolve_amp_dtype,
    save_config,
    seed_everything,
    setup_logger,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X-AuT minimal LoRA finetune")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default="", help="Output directory override")
    parser.add_argument("--device-type", default="cuda", help="cuda or cpu")
    return parser.parse_args()


def build_runtime_config(
    config: dict[str, Any],
    *,
    split_name: str,
    output_dir: Path,
    rank: int,
) -> DatasetRuntimeConfig:
    base_cache_dir = Path(str(deep_get(config, "data.audio_cache_dir", "./audio_cache")).strip()).expanduser().resolve()
    return DatasetRuntimeConfig(
        qwen_source=str(deep_get(config, "model.qwen_source", deep_get(config, "tokenizer.source", ""))).strip(),
        target_format=str(deep_get(config, "data.target_format", "qwen_asr")).strip().lower(),
        cache_dir=str(base_cache_dir),
        oss_config_path=str(deep_get(config, "data.oss_config_path", "")).strip(),
        oss_workers=int(deep_get(config, "data.oss_workers", 4) or 4),
        prefetch_workers=1,
        local_audio_roots=tuple(str(item) for item in (deep_get(config, "data.local_audio_roots", []) or [])),
        local_files_only=bool(deep_get(config, "model.local_files_only", deep_get(config, "tokenizer.local_files_only", True))),
        fix_mistral_regex=bool(deep_get(config, "model.fix_mistral_regex", deep_get(config, "tokenizer.fix_mistral_regex", True))),
        add_eos=bool(deep_get(config, "data.add_eos", True)),
        split_name=split_name,
        rank=rank,
        cache_mode="persistent",
        prefetch_mode="none",
        prefetch_groups_ahead=0,
        prefetch_group_records=0,
        dataloader_timeout_seconds=float(deep_get(config, "data.dataloader_timeout_seconds", 600) or 0),
        audio_resolve_timeout_seconds=float(deep_get(config, "data.audio_resolve_timeout_seconds", 600) or 0),
        max_audio_duration_seconds=max(float(deep_get(config, "data.max_audio_duration_seconds", 0) or 0), 0.0),
        max_failed_record_fraction=float(deep_get(config, "data.max_failed_record_fraction", 1.0) or 0.0),
        num_length_bins=int(deep_get(config, "data.num_length_bins", 9) or 9),
    )


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(deep_get(config, "train.lr", 1e-5) or 1e-5)
    weight_decay = float(deep_get(config, "train.weight_decay", 0.01) or 0.01)
    betas = deep_get(config, "train.betas", [0.9, 0.95]) or [0.9, 0.95]
    params = [p for p in unwrap_model(model).parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Check trainable.* config.")
    return torch.optim.AdamW([{"params": params, "lr": lr}], betas=(float(betas[0]), float(betas[1])), weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(current_step + 1, 1) / max(warmup_steps, 1)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_step_budget(
    *,
    config: dict[str, Any],
    estimated_steps_per_epoch: int,
) -> dict[str, float | int]:
    raw_epoch_value = deep_get(config, "train.epochs", None)
    epoch_value: float | None = None
    if raw_epoch_value is not None:
        epoch_value = float(raw_epoch_value)
        if not math.isfinite(epoch_value) or epoch_value <= 0:
            raise ValueError(f"train.epochs must be a positive finite number, got {epoch_value!r}")

    max_train_steps = deep_get(config, "train.max_train_steps", None)
    if max_train_steps is not None:
        total_steps = max(int(max_train_steps), 1)
    else:
        if epoch_value is None:
            raise ValueError("train.epochs must be set when train.max_train_steps is unset")
        total_steps = max(int(math.floor(epoch_value * max(int(estimated_steps_per_epoch), 1) + 0.5)), 1)
    resolved_train_epochs = (
        float(epoch_value)
        if epoch_value is not None
        else (float(total_steps) / max(float(estimated_steps_per_epoch), 1.0))
    )
    epochs = max(math.ceil(total_steps / max(int(estimated_steps_per_epoch), 1)), 1)
    return {
        "train_epochs": resolved_train_epochs,
        "total_steps": total_steps,
        "epochs": epochs,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    config: dict[str, Any],
    output_dir: Path,
    step: int,
    epoch: int,
    filename: str,
) -> Path:
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    checkpoint_path = checkpoint_dir / filename
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "step": int(step),
            "epoch": int(epoch),
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    output_dir = Path(str(config.get("output_dir") or "")).expanduser().resolve()
    if not str(output_dir):
        raise ValueError("output_dir must be set either in config or via --output-dir")

    dist_ctx = init_distributed(args.device_type)
    seed_everything(int(config.get("seed", 42) or 42), rank=dist_ctx.rank, use_cuda=dist_ctx.device.type == "cuda")
    logger = setup_logger(output_dir, dist_ctx)
    barrier(dist_ctx)

    train_index_payload = load_train_index(str(config["train_manifest_index"]))
    per_device_batch_size = int(deep_get(config, "train.per_device_batch_size", 1) or 1)
    grad_accum_steps = int(deep_get(config, "train.grad_accum_steps", 1) or 1)
    runtime_config = build_runtime_config(
        config,
        split_name="train",
        output_dir=output_dir,
        rank=dist_ctx.rank,
    )
    estimated_steps_per_epoch = estimate_train_steps(
        train_index_payload,
        per_device_batch_size=per_device_batch_size,
        world_size=dist_ctx.world_size,
        grad_accum_steps=grad_accum_steps,
    )
    budget = resolve_step_budget(
        config=config,
        estimated_steps_per_epoch=estimated_steps_per_epoch,
    )
    total_steps = int(budget["total_steps"])
    epochs = int(budget["epochs"])

    artifacts = build_model_from_config(
        config,
        device=dist_ctx.device,
        apply_trainability=True,
        enable_gradient_checkpointing=bool(deep_get(config, "train.gradient_checkpointing", True)),
    )
    model = artifacts.model.to(dist_ctx.device)

    if dist_ctx.distributed:
        model = DDP(
            model,
            device_ids=[dist_ctx.local_rank] if dist_ctx.device.type == "cuda" else None,
            output_device=dist_ctx.local_rank if dist_ctx.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(deep_get(config, "train.warmup_steps", 0) or 0),
        min_lr_ratio=float(deep_get(config, "train.min_lr_ratio", 0.1) or 0.1),
    )
    global_step = 0
    pad_token_id = unwrap_model(model).pad_token_id
    amp_dtype = resolve_amp_dtype(deep_get(config, "train.amp", "auto"), device=dist_ctx.device)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(dist_ctx.device.type == "cuda" and amp_dtype == torch.float16))

    num_workers = int(deep_get(config, "data.num_workers", 0) or 0)
    log_every = max(int(deep_get(config, "train.log_every", 20) or 20), 1)
    save_every_steps = max(int(deep_get(config, "train.save_every_steps", 0) or 0), 0)
    clip_grad_norm = float(deep_get(config, "train.clip_grad_norm", 1.0) or 1.0)

    if is_main_process(dist_ctx):
        save_config(config, output_dir / "config.yaml")
        logger.info(
            "Loaded X-AuT | total=%s | trainable=%s | device=%s",
            format_parameter_count(count_parameters(unwrap_model(model))),
            format_parameter_count(count_trainable_parameters(unwrap_model(model))),
            dist_ctx.device,
        )
        logger.info(
            "Resolved train schedule | epochs=%.4f | total_steps=%d | warmup_steps=%d",
            budget["train_epochs"],
            total_steps,
            int(deep_get(config, "train.warmup_steps", 0) or 0),
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(0, epochs):
            train_dataset = build_train_dataset(
                train_index_payload,
                runtime_config=runtime_config,
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
                epoch=epoch,
                shuffle_shards=bool(deep_get(config, "data.shuffle_shards", True)),
                seed=int(config.get("seed", 42) or 42),
                per_device_batch_size=per_device_batch_size,
                grad_accum_steps=grad_accum_steps,
            )
            train_loader = build_train_dataloader(
                train_dataset,
                pad_token_id=pad_token_id,
                batch_size=per_device_batch_size,
                num_workers=num_workers,
                persistent_workers=bool(deep_get(config, "data.persistent_workers", True)),
                pin_memory=bool(dist_ctx.device.type == "cuda"),
                timeout_seconds=runtime_config.dataloader_timeout_seconds,
            )

            for batch_index, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, dist_ctx.device)
                with autocast_context(dist_ctx.device, amp_dtype):
                    outputs = model(
                        input_features=batch["input_features"],
                        feature_attention_mask=batch["feature_attention_mask"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                should_step = ((batch_index + 1) % grad_accum_steps == 0)
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if is_main_process(dist_ctx) and global_step % log_every == 0:
                        logger.info(
                            "step=%d/%d | epoch=%d | loss=%.6f | lr=%.6e",
                            global_step,
                            total_steps,
                            epoch,
                            float(loss.detach().cpu().item() * grad_accum_steps),
                            scheduler.get_last_lr()[0],
                        )

                    if save_every_steps > 0 and global_step % save_every_steps == 0 and is_main_process(dist_ctx):
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            config=config,
                            output_dir=output_dir,
                            step=global_step,
                            epoch=epoch,
                            filename="latest.pt",
                        )

                    if global_step >= total_steps:
                        break

            barrier(dist_ctx)
            if global_step >= total_steps:
                break

        barrier(dist_ctx)
        if is_main_process(dist_ctx):
            if bool(deep_get(config, "train.save_final_checkpoint", True)):
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config=config,
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    filename="latest.pt",
                )
            logger.info("Training finished | final_step=%d", global_step)
    finally:
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
