from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from x_aut.block_mask import apply_block_mask_patch as _apply_block_mask_patch
from x_aut.causal_patch import apply_causal_audio_tower_patch, resolve_causal_lookahead_config
from x_aut.generation_padding import normalize_generation_inputs_for_decoder_only
from x_aut.utils import deep_get, resolve_model_dtype, unwrap_model

if TYPE_CHECKING:
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

_LOGGER = logging.getLogger(__name__)
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
FALLBACK_ATTN_IMPLEMENTATION = "eager"


def _extend_unique_token_ids(values: list[int], raw_value: Any) -> None:
    if raw_value is None:
        return
    if isinstance(raw_value, (list, tuple, set)):
        raw_items = list(raw_value)
    else:
        raw_items = [raw_value]
    for item in raw_items:
        if item is None:
            continue
        token_id = int(item)
        if token_id < 0 or token_id in values:
            continue
        values.append(token_id)


def resolve_generation_eos_token_ids(
    model: Qwen3ASRForConditionalGeneration,
    *,
    processor: Qwen3ASRProcessor,
) -> list[int]:
    token_ids: list[int] = []
    thinker = getattr(model, "thinker", None)
    thinker_model = getattr(thinker, "model", None)
    _extend_unique_token_ids(token_ids, getattr(processor.tokenizer, "eos_token_id", None))
    _extend_unique_token_ids(token_ids, getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    _extend_unique_token_ids(token_ids, getattr(getattr(thinker, "generation_config", None), "eos_token_id", None))
    _extend_unique_token_ids(token_ids, getattr(getattr(thinker_model, "generation_config", None), "eos_token_id", None))
    return token_ids


def coerce_generation_eos_token_id(token_ids: list[int]) -> int | list[int] | None:
    if not token_ids:
        return None
    if len(token_ids) == 1:
        return int(token_ids[0])
    return [int(token_id) for token_id in token_ids]


def _resolve_qwen_source(config: dict[str, Any]) -> str:
    candidates = [
        str(deep_get(config, "model.qwen_source", "") or "").strip(),
        str(deep_get(config, "model.qwen.source", "") or "").strip(),
        str(deep_get(config, "tokenizer.source", "") or "").strip(),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    raise ValueError("Missing Qwen source. Set model.qwen_source or tokenizer.source.")


def _resolve_teacher_qwen_source(config: dict[str, Any]) -> str:
    candidates = [
        str(deep_get(config, "teacher.qwen_source", "") or "").strip(),
        str(deep_get(config, "teacher.source", "") or "").strip(),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return _resolve_qwen_source(config)


def _resolve_local_files_only(config: dict[str, Any]) -> bool:
    value = deep_get(config, "model.local_files_only", None)
    if value is None:
        value = deep_get(config, "tokenizer.local_files_only", True)
    return bool(value)


def _resolve_teacher_local_files_only(config: dict[str, Any]) -> bool:
    value = deep_get(config, "teacher.local_files_only", None)
    if value is None:
        return _resolve_local_files_only(config)
    return bool(value)


def _resolve_fix_mistral_regex(config: dict[str, Any]) -> bool:
    value = deep_get(config, "model.fix_mistral_regex", None)
    if value is None:
        value = deep_get(config, "tokenizer.fix_mistral_regex", True)
    return bool(value)


def _resolve_teacher_fix_mistral_regex(config: dict[str, Any]) -> bool:
    value = deep_get(config, "teacher.fix_mistral_regex", None)
    if value is None:
        return _resolve_fix_mistral_regex(config)
    return bool(value)


def _import_qwen_backend() -> tuple[type["Qwen3ASRForConditionalGeneration"], type["Qwen3ASRProcessor"]]:
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

    return Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor


def _flash_attention_2_supported_on_device(device: torch.device) -> bool:
    if device.type != "cuda":
        return False



    try:
        name = torch.cuda.get_device_name(device).lower()
        if not any(vendor in name for vendor in ("nvidia", "geforce", "quadro", "tesla", "a100", "h100", "h800", "a800", "l40", "rtx", "v100", "t4")):
            _LOGGER.warning(
                "flash_attention_2 disabled: device name '%s' is not a recognised NVIDIA GPU",
                torch.cuda.get_device_name(device),
            )
            return False
    except Exception:


        _LOGGER.warning("flash_attention_2 disabled: unable to query CUDA device name")
        return False
    from transformers.utils import is_flash_attn_2_available

    return bool(is_flash_attn_2_available())


def _resolve_attn_implementation_value(raw_value: Any, *, device: torch.device) -> str:
    requested = str(raw_value).strip() if raw_value is not None else ""
    if not requested:
        requested = DEFAULT_ATTN_IMPLEMENTATION
    if requested != DEFAULT_ATTN_IMPLEMENTATION:
        return requested
    if _flash_attention_2_supported_on_device(device):
        return requested
    _LOGGER.warning(
        "flash_attention_2 is unavailable on device=%s; falling back to %s",
        device,
        FALLBACK_ATTN_IMPLEMENTATION,
    )
    return FALLBACK_ATTN_IMPLEMENTATION


def resolve_attn_implementation(config: dict[str, Any], *, device: torch.device) -> str:
    return _resolve_attn_implementation_value(
        deep_get(config, "model.attn_implementation", None),
        device=device,
    )


def _resolve_teacher_attn_implementation(config: dict[str, Any], *, device: torch.device) -> str:
    raw_value = deep_get(config, "teacher.attn_implementation", None)
    if raw_value is None:
        return resolve_attn_implementation(config, device=device)
    return _resolve_attn_implementation_value(raw_value, device=device)


def _sanitize_generation_config_dict(raw_config: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(raw_config or {})
    if not bool(sanitized.get("do_sample", False)):
        temperature = sanitized.get("temperature", None)
        if temperature is not None:
            try:
                should_drop = float(temperature) != 1.0
            except (TypeError, ValueError):
                should_drop = True
            if should_drop:
                sanitized.pop("temperature", None)
    return sanitized


def _build_sanitized_generation_config(qwen_source: str) -> Any | None:
    source_path = Path(str(qwen_source)).expanduser()
    generation_config_path = source_path / "generation_config.json"
    if not generation_config_path.is_file():
        return None
    try:
        from transformers import GenerationConfig

        raw_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
        return GenerationConfig(**_sanitize_generation_config_dict(raw_config))
    except Exception:
        return None


def _normalize_generation_config(generation_config: Any | None) -> None:
    if generation_config is None:
        return
    if not bool(getattr(generation_config, "do_sample", False)):
        temperature = getattr(generation_config, "temperature", None)
        if temperature is None:
            return
        try:
            is_neutral = float(temperature) == 1.0
        except (TypeError, ValueError):
            is_neutral = False
        if not is_neutral:
            generation_config.temperature = 1.0


def _apply_loaded_model_runtime_defaults(
    model: "Qwen3ASRForConditionalGeneration",
    *,
    attn_implementation: str,
) -> None:
    model.config._attn_implementation = attn_implementation
    model.thinker.config._attn_implementation = attn_implementation
    model.thinker.model.config._attn_implementation = attn_implementation
    _normalize_generation_config(getattr(model, "generation_config", None))
    _normalize_generation_config(getattr(model.thinker, "generation_config", None))
    _normalize_generation_config(getattr(model.thinker.model, "generation_config", None))


def _load_pretrained_qwen_model(
    model_cls: type["Qwen3ASRForConditionalGeneration"],
    *,
    qwen_source: str,
    local_files_only: bool,
    requested_dtype: torch.dtype,
    attn_implementation: str,
) -> "Qwen3ASRForConditionalGeneration":
    pretrained_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
        "dtype": requested_dtype,
    }
    generation_config = _build_sanitized_generation_config(qwen_source)
    if generation_config is not None:
        pretrained_kwargs["generation_config"] = generation_config
    try:
        return model_cls.from_pretrained(qwen_source, **pretrained_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" not in message and "got an unexpected keyword" not in message:
            raise
        fallback_kwargs = dict(pretrained_kwargs)
        if "generation_config" in message:
            fallback_kwargs.pop("generation_config", None)
        if "dtype" in message:
            fallback_kwargs.pop("dtype", None)
            fallback_kwargs["torch_dtype"] = requested_dtype
        return model_cls.from_pretrained(qwen_source, **fallback_kwargs)


def _enable_gradient_checkpointing(
    model: "Qwen3ASRForConditionalGeneration",
) -> None:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )


def resolve_encoder_layer_indices(config: dict[str, Any], *, total_layers: int) -> list[int] | None:
    """Resolve model.encoder_layer_indices (1-based) into a list of 0-based teacher layer indices.

    Returns None when not configured, meaning the student keeps the first ``encoder_layers`` layers.
    """
    raw_indices = deep_get(config, "model.encoder_layer_indices", None)
    if raw_indices is None:
        return None
    if not isinstance(raw_indices, (list, tuple)) or len(raw_indices) == 0:
        raise ValueError("model.encoder_layer_indices must be a non-empty list of 1-based layer indices")
    resolved: list[int] = []
    seen: set[int] = set()
    for raw_index in raw_indices:
        index_0based = int(raw_index) - 1
        if index_0based < 0 or index_0based >= total_layers:
            raise ValueError(
                f"encoder_layer_index {int(raw_index)} is out of range for audio tower depth {total_layers} "
                "(expected 1-based indices)"
            )
        if index_0based in seen:
            raise ValueError(f"model.encoder_layer_indices must be unique, got duplicate index {int(raw_index)}")
        seen.add(index_0based)
        resolved.append(index_0based)
    return resolved


def truncate_audio_tower(
    model: Qwen3ASRForConditionalGeneration,
    *,
    encoder_layers: int | None = None,
    layer_indices: list[int] | None = None,
) -> None:
    audio_tower = model.thinker.audio_tower
    original_layers = list(audio_tower.layers)
    total_layers = len(original_layers)
    if layer_indices is not None:
        if len(layer_indices) == 0:
            raise ValueError("layer_indices must not be empty")
        seen: set[int] = set()
        selected: list[nn.Module] = []
        for index_0based in layer_indices:
            index_0based = int(index_0based)
            if index_0based < 0 or index_0based >= total_layers:
                raise ValueError(
                    f"layer_index {index_0based} is out of range for audio tower depth {total_layers}"
                )
            if index_0based in seen:
                raise ValueError(f"layer_indices must be unique, got duplicate index {index_0based}")
            seen.add(index_0based)
            selected.append(original_layers[index_0based])
        audio_tower.layers = nn.ModuleList(selected)
        model.config.thinker_config.audio_config.encoder_layers = len(selected)
        return
    if encoder_layers is None or encoder_layers <= 0:
        raise ValueError(f"encoder_layers must be positive, got {encoder_layers}")
    if encoder_layers > total_layers:
        raise ValueError(f"Requested encoder_layers={encoder_layers}, but checkpoint only has {total_layers} layers")
    audio_tower.layers = nn.ModuleList(original_layers[:encoder_layers])
    model.config.thinker_config.audio_config.encoder_layers = int(encoder_layers)


def apply_trainability_config(model: Qwen3ASRForConditionalGeneration, config: dict[str, Any]) -> None:
    trainable_cfg = dict(config.get("trainable") or {})
    audio_trainable = bool(trainable_cfg.get("audio_tower", True))
    audio_encoder_trainable = bool(trainable_cfg.get("audio_encoder", audio_trainable))
    bridge_trainable = bool(trainable_cfg.get("bridge", audio_trainable))
    bridge_ln_post_trainable = bool(trainable_cfg.get("bridge_ln_post", bridge_trainable))
    bridge_proj1_trainable = bool(trainable_cfg.get("bridge_proj1", bridge_trainable))
    bridge_proj2_trainable = bool(trainable_cfg.get("bridge_proj2", bridge_trainable))

    text_trainable = bool(trainable_cfg.get("text_decoder", False))
    decoder_embeddings_trainable = bool(trainable_cfg.get("text_decoder_embeddings", text_trainable))
    decoder_norm_trainable = bool(trainable_cfg.get("text_decoder_norm", text_trainable))
    decoder_trainable_layers = _resolve_decoder_trainable_layer_indices(model, trainable_cfg, text_trainable=text_trainable)
    lm_head_trainable = bool(trainable_cfg.get("lm_head", False))

    for name, parameter in model.named_parameters():
        if name.startswith("thinker.audio_tower."):
            suffix = name[len("thinker.audio_tower."):]
            if suffix.startswith("ln_post."):
                parameter.requires_grad = bridge_ln_post_trainable
            elif suffix.startswith("proj1."):
                parameter.requires_grad = bridge_proj1_trainable
            elif suffix.startswith("proj2."):
                parameter.requires_grad = bridge_proj2_trainable
            else:
                parameter.requires_grad = audio_encoder_trainable
        elif name.startswith("thinker.model."):
            suffix = name[len("thinker.model."):]
            layer_index = _extract_decoder_layer_index(suffix)
            if text_trainable:
                parameter.requires_grad = True
            elif "lora_" in name:


                parameter.requires_grad = True
            elif layer_index is not None:
                parameter.requires_grad = layer_index in decoder_trainable_layers
            elif suffix.startswith("embed_tokens.") or suffix.startswith("model.embed_tokens."):
                parameter.requires_grad = decoder_embeddings_trainable
            elif suffix.startswith("norm.") or suffix.startswith("model.norm."):
                parameter.requires_grad = decoder_norm_trainable
            else:
                parameter.requires_grad = False
        elif name.startswith("thinker.lm_head."):
            parameter.requires_grad = lm_head_trainable
        else:
            parameter.requires_grad = True






    if lm_head_trainable:
        try:
            input_embeddings = model.thinker.get_input_embeddings()
            lm_head_weight = model.thinker.lm_head.weight
        except Exception:
            input_embeddings = None
            lm_head_weight = None
        if (
            input_embeddings is not None
            and lm_head_weight is not None
            and input_embeddings.weight is lm_head_weight
        ):
            input_embeddings.weight.requires_grad = True


def _resolve_decoder_trainable_layer_indices(
    model: Qwen3ASRForConditionalGeneration,
    trainable_cfg: dict[str, Any],
    *,
    text_trainable: bool,
) -> set[int]:
    total_layers = _resolve_decoder_layer_count(model)
    if text_trainable:
        return set(range(total_layers))
    explicit_indices = _parse_trainable_layer_indices(
        trainable_cfg.get("text_decoder_layer_indices"),
        total_layers=total_layers,
        field_name="trainable.text_decoder_layer_indices",
    )
    if explicit_indices is not None:
        return explicit_indices
    top_layers = int(trainable_cfg.get("text_decoder_top_layers", 0) or 0)
    if top_layers < 0:
        raise ValueError(f"trainable.text_decoder_top_layers must be >= 0, got {top_layers}")
    if top_layers == 0:
        return set()
    if total_layers <= 0:
        raise ValueError("Cannot use trainable.text_decoder_top_layers because decoder layer count could not be resolved")
    if top_layers > total_layers:
        raise ValueError(
            f"trainable.text_decoder_top_layers={top_layers} exceeds decoder depth {total_layers}"
        )
    return set(range(total_layers - top_layers, total_layers))


def _resolve_decoder_layer_count(model: Qwen3ASRForConditionalGeneration) -> int:
    decoder = getattr(model.thinker, "model", None)
    if decoder is None:
        return 0
    layer_modules = getattr(decoder, "layers", None)
    if isinstance(layer_modules, nn.ModuleList):
        return len(layer_modules)
    nested_model = getattr(decoder, "model", None)
    nested_layers = getattr(nested_model, "layers", None)
    if isinstance(nested_layers, nn.ModuleList):
        return len(nested_layers)
    return 0


def _parse_trainable_layer_indices(raw_value: Any, *, total_layers: int, field_name: str) -> set[int] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, (list, tuple, set)) or not raw_value:
        raise ValueError(f"{field_name} must be a non-empty list of 1-based layer indices")
    resolved: set[int] = set()
    for raw_index in raw_value:
        layer_index = int(raw_index)
        if layer_index <= 0 or layer_index > total_layers:
            raise ValueError(
                f"{field_name} contains out-of-range index {layer_index}; decoder depth is {total_layers}"
            )
        zero_based = layer_index - 1
        if zero_based in resolved:
            raise ValueError(f"{field_name} must not contain duplicate index {layer_index}")
        resolved.add(zero_based)
    return resolved


def _extract_decoder_layer_index(parameter_suffix: str) -> int | None:
    for prefix in ("layers.", "model.layers."):
        if not parameter_suffix.startswith(prefix):
            continue
        remainder = parameter_suffix[len(prefix):]
        raw_index = remainder.split(".", 1)[0]
        if raw_index.isdigit():
            return int(raw_index)
    return None


class AudioDistillProjection(nn.Module):
    """Project teacher audio representations into the student feature space.

    Used when the teacher and student audio encoders have different output
    dimensions.  A bottleneck MLP with LayerNorm + GELU is preferred over a
    single linear map because it acts as a low-pass filter: empirical work on
    cross-architecture distillation shows that teacher/student shared structure
    is concentrated in a low-rank subspace, and a bottleneck suppresses
    architecture-specific high-frequency noise.
    """

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        *,
        bottleneck_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        teacher_dim = int(teacher_dim)
        student_dim = int(student_dim)
        if bottleneck_dim is None:



            bottleneck_dim = max(64, min(teacher_dim, student_dim) // 4)
        else:
            bottleneck_dim = int(bottleneck_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(teacher_dim, elementwise_affine=True),
            nn.Linear(teacher_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(bottleneck_dim, student_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapped around a ``nn.Linear`` base layer.

    The base layer weights are frozen and the low-rank matrices ``lora_A`` and
    ``lora_B`` are trainable.  This implementation is intentionally lightweight
    and avoids an external ``peft`` dependency, while keeping parameter names
    stable so that existing checkpoint saving/loading code continues to work.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        r: int,
        lora_alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = max(int(r), 1)
        self.lora_alpha = int(lora_alpha)
        self.scaling = self.lora_alpha / self.r

        in_features = int(base_layer.in_features)
        out_features = int(base_layer.out_features)
        self.lora_A = nn.Parameter(torch.empty(in_features, self.r))
        self.lora_B = nn.Parameter(torch.empty(self.r, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()


        base_layer.weight.requires_grad = False
        if base_layer.bias is not None:
            base_layer.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(x)
        if self.r > 0:
            lora_input = self.dropout(x)
            if self.lora_A.dtype != lora_input.dtype or self.lora_A.device != lora_input.device:



                lora_a = self.lora_A.to(device=lora_input.device, dtype=lora_input.dtype)
                lora_b = self.lora_B.to(device=lora_input.device, dtype=lora_input.dtype)
            else:
                lora_a = self.lora_A
                lora_b = self.lora_B
            lora_update = lora_input @ lora_a @ lora_b
            if lora_update.dtype != output.dtype:
                lora_update = lora_update.to(dtype=output.dtype)
            output = output + lora_update * self.scaling
        return output


def _apply_lora_recursive(
    module: nn.Module,
    target_modules: set[str],
    *,
    r: int,
    lora_alpha: int,
    dropout: float,
) -> int:
    """Recursively replace matching ``nn.Linear`` children with ``LoRALinear``."""
    replaced = 0
    for name, child in list(module.named_children()):
        if name in target_modules and isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LoRALinear(
                    child,
                    r=r,
                    lora_alpha=lora_alpha,
                    dropout=dropout,
                ),
            )
            replaced += 1
        else:
            replaced += _apply_lora_recursive(
                child, target_modules, r=r, lora_alpha=lora_alpha, dropout=dropout
            )
    return replaced


def apply_lora_to_decoder(model: Qwen3ASRForConditionalGeneration, config: dict[str, Any]) -> bool:
    """Apply LoRA adapters to the text decoder if ``lora.enabled`` is true."""
    lora_cfg = deep_get(config, "lora", None)
    if not isinstance(lora_cfg, dict) or not bool(lora_cfg.get("enabled", False)):
        return False

    r = int(lora_cfg.get("r", 16))
    lora_alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.05))
    target_modules = set(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))

    decoder = getattr(model.thinker, "model", None)
    if decoder is None:
        raise ValueError("Cannot apply LoRA: model.thinker.model is missing")

    replaced = _apply_lora_recursive(
        decoder,
        target_modules,
        r=r,
        lora_alpha=lora_alpha,
        dropout=dropout,
    )
    if replaced == 0:
        _LOGGER.warning(
            "LoRA enabled but no target modules %s were found in the decoder",
            sorted(target_modules),
        )
    else:
        _LOGGER.info(
            "Applied LoRA to decoder | r=%d alpha=%d dropout=%.3f target_modules=%s replaced_modules=%d",
            r,
            lora_alpha,
            dropout,
            sorted(target_modules),
            replaced,
        )
    return True


@dataclass
class XAuTArtifacts:
    model: "XAuTModel"
    processor: Qwen3ASRProcessor


@dataclass
class XAuTDistillationArtifacts:
    student: "XAuTModel"
    processor: Qwen3ASRProcessor
    teacher: "XAuTModel"
    teacher_audio_tower: nn.Module
    teacher_encoder_layers: int
    teacher_layer_projection: nn.Module | None = None
    teacher_bridge_projection: nn.Module | None = None


class XAuTModel(nn.Module):
    def __init__(
        self,
        model: Qwen3ASRForConditionalGeneration,
        *,
        processor: Qwen3ASRProcessor,
        qwen_source: str,
        encoder_layers: int,
        target_format: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.qwen_source = qwen_source
        self.encoder_layers = int(encoder_layers)
        self.target_format = str(target_format)
        self.audio_token = str(processor.audio_token)
        self.audio_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.audio_token))
        self.pad_token_id = int(
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        )
        self.eos_token_ids = resolve_generation_eos_token_ids(model, processor=processor)

    def _cast_input_features(self, input_features: torch.Tensor) -> torch.Tensor:
        target_dtype = self.model.thinker.audio_tower.conv2d1.weight.dtype
        if input_features.dtype == target_dtype:
            return input_features
        return input_features.to(dtype=target_dtype)

    def _resolve_discrete_input(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> tuple[Any, Any, Any]:
        """Run the (VQ-patched) audio tower and resolve discrete input mode.

        Returns ``(vq_embedding, token_ids, audio_features)`` when the audio
        tower carries a ``vq_bundle.vq_embedding`` (discrete input mode), else
        ``(None, None, None)``.  Detected via duck typing so X-AuT never imports
        x_aut_vq (avoids a circular dependency).
        """
        audio_tower = self.model.thinker.audio_tower
        vq_bundle = getattr(audio_tower, "vq_bundle", None)
        discrete_embedding = getattr(vq_bundle, "vq_embedding", None)
        if discrete_embedding is None:
            return None, None, None



        token_buffer = getattr(audio_tower, "_vq_token_buffer", None)
        if token_buffer is not None:
            token_buffer.clear()
        audio_features = extract_audio_features(
            audio_tower,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        token_ids: torch.Tensor | None = None
        if token_buffer is not None and token_buffer:
            token_ids = torch.cat(list(token_buffer), dim=0)
        if token_ids is None:
            vq_aux = getattr(getattr(vq_bundle, "quantizer", None), "last_aux", None)
            token_ids = getattr(vq_aux, "token_ids", None)
        if token_ids is None or int(token_ids.numel()) == 0:
            raise ValueError(
                "Discrete input mode requires VQ token_ids from the audio tower forward"
            )
        return discrete_embedding, token_ids, audio_features

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_audio_features_only: bool = False,
    ):
        input_features = self._cast_input_features(input_features)
        if return_audio_features_only:
            return self.model.thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
            )
        discrete_embedding, token_ids, audio_features = self._resolve_discrete_input(
            input_features,
            feature_attention_mask,
        )
        if discrete_embedding is not None:
            return thinker_forward_from_audio_features(
                self.model.thinker,
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                discrete_token_ids=token_ids,
                discrete_embedding=discrete_embedding,
                labels=labels,
            )
        return self.model.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_kwargs: dict[str, Any] | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = dict(generation_kwargs or {})
        if not bool(kwargs.get("do_sample", False)):
            kwargs.pop("temperature", None)
        default_eos_token_id = coerce_generation_eos_token_id(self.eos_token_ids)
        if default_eos_token_id is not None and "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = default_eos_token_id
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = self.pad_token_id
        kwargs.setdefault("return_dict_in_generate", False)
        input_ids, attention_mask = normalize_generation_inputs_for_decoder_only(
            input_ids,
            attention_mask,
            pad_token_id=int(kwargs["pad_token_id"]),
        )
        if audio_features is not None:
            return thinker_generate_from_audio_features(
                self.model.thinker,
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                generation_kwargs=kwargs,
            )
        if input_features is None or feature_attention_mask is None:
            raise ValueError("input_features and feature_attention_mask are required when audio_features are not provided")
        input_features = self._cast_input_features(input_features)
        discrete_embedding, token_ids, audio_features = self._resolve_discrete_input(
            input_features,
            feature_attention_mask,
        )
        if discrete_embedding is not None:
            return thinker_generate_from_audio_features(
                self.model.thinker,
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                discrete_token_ids=token_ids,
                discrete_embedding=discrete_embedding,
                generation_kwargs=kwargs,
            )
        return self.model.thinker.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )


def build_processor_from_source(
    *,
    qwen_source: str,
    fix_mistral_regex: bool,
    local_files_only: bool,
) -> Qwen3ASRProcessor:
    _, Qwen3ASRProcessor = _import_qwen_backend()
    return Qwen3ASRProcessor.from_pretrained(
        qwen_source,
        fix_mistral_regex=fix_mistral_regex,
        local_files_only=local_files_only,
    )


def build_processor_from_config(config: dict[str, Any]) -> Qwen3ASRProcessor:
    qwen_source = _resolve_qwen_source(config)
    return build_processor_from_source(
        qwen_source=qwen_source,
        fix_mistral_regex=_resolve_fix_mistral_regex(config),
        local_files_only=_resolve_local_files_only(config),
    )


def _resolve_lm_head_vocab_size(model: "Qwen3ASRForConditionalGeneration") -> int:
    lm_head = getattr(getattr(model, "thinker", None), "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is not None and getattr(weight, "shape", None):
        return int(weight.shape[0])
    out_features = getattr(lm_head, "out_features", None)
    if out_features is not None:
        return int(out_features)
    return int(getattr(getattr(getattr(model, "thinker", None), "config", None), "vocab_size", 0) or 0)


def _resolve_processor_pad_token_id(processor: Qwen3ASRProcessor) -> int | None:
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return None


def _validate_teacher_compatibility(
    *,
    student_model: "Qwen3ASRForConditionalGeneration",
    student_processor: Qwen3ASRProcessor,
    student_qwen_source: str,
    teacher_model: "Qwen3ASRForConditionalGeneration",
    teacher_processor: Qwen3ASRProcessor,
    teacher_qwen_source: str,
) -> None:
    student_vocab_size = _resolve_lm_head_vocab_size(student_model)
    teacher_vocab_size = _resolve_lm_head_vocab_size(teacher_model)
    if student_vocab_size > 0 and teacher_vocab_size > 0 and student_vocab_size != teacher_vocab_size:
        raise ValueError(
            "Teacher/student vocab sizes do not match for logit KD: "
            f"student={student_vocab_size} ({student_qwen_source}), "
            f"teacher={teacher_vocab_size} ({teacher_qwen_source})"
        )

    student_audio_token = str(student_processor.audio_token)
    teacher_audio_token = str(teacher_processor.audio_token)
    if student_audio_token != teacher_audio_token:
        raise ValueError(
            "Teacher/student audio placeholder tokens do not match: "
            f"student={student_audio_token!r} ({student_qwen_source}), "
            f"teacher={teacher_audio_token!r} ({teacher_qwen_source})"
        )

    student_audio_token_id = int(student_processor.tokenizer.convert_tokens_to_ids(student_audio_token))
    teacher_audio_token_id = int(teacher_processor.tokenizer.convert_tokens_to_ids(teacher_audio_token))
    if student_audio_token_id != teacher_audio_token_id:
        raise ValueError(
            "Teacher/student audio placeholder token ids do not match: "
            f"student={student_audio_token_id} ({student_qwen_source}), "
            f"teacher={teacher_audio_token_id} ({teacher_qwen_source})"
        )

    student_pad_token_id = _resolve_processor_pad_token_id(student_processor)
    teacher_pad_token_id = _resolve_processor_pad_token_id(teacher_processor)
    if (
        student_pad_token_id is not None
        and teacher_pad_token_id is not None
        and student_pad_token_id != teacher_pad_token_id
    ):
        raise ValueError(
            "Teacher/student resolved pad token ids do not match: "
            f"student={student_pad_token_id} ({student_qwen_source}), "
            f"teacher={teacher_pad_token_id} ({teacher_qwen_source})"
        )

    student_eos_token_id = getattr(student_processor.tokenizer, "eos_token_id", None)
    teacher_eos_token_id = getattr(teacher_processor.tokenizer, "eos_token_id", None)
    if (
        student_eos_token_id is not None
        and teacher_eos_token_id is not None
        and int(student_eos_token_id) != int(teacher_eos_token_id)
    ):
        raise ValueError(
            "Teacher/student eos token ids do not match: "
            f"student={int(student_eos_token_id)} ({student_qwen_source}), "
            f"teacher={int(teacher_eos_token_id)} ({teacher_qwen_source})"
        )


def _freeze_model_parameters(model: "Qwen3ASRForConditionalGeneration") -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def _maybe_apply_causal_patch(model: Qwen3ASRForConditionalGeneration, config: dict[str, Any]) -> None:
    """Apply the limited-causal audio tower patch when ``model.causal_lookahead.enabled`` is true."""
    causal_cfg = resolve_causal_lookahead_config(config)
    if causal_cfg is None:
        return
    apply_causal_audio_tower_patch(
        model.thinker.audio_tower,
        total_lookahead_frames=causal_cfg["total_lookahead_frames"],
        schedule_mode=causal_cfg["schedule_mode"],
    )


def build_model_from_config(
    config: dict[str, Any],
    *,
    device: torch.device,
    apply_trainability: bool,
    enable_gradient_checkpointing: bool,
) -> XAuTArtifacts:
    qwen_source = _resolve_qwen_source(config)
    processor = build_processor_from_config(config)
    requested_dtype = resolve_model_dtype(deep_get(config, "model.load_dtype", "auto"), device=device)
    attn_implementation = resolve_attn_implementation(config, device=device)
    Qwen3ASRForConditionalGeneration, _ = _import_qwen_backend()
    model = _load_pretrained_qwen_model(
        Qwen3ASRForConditionalGeneration,
        qwen_source=qwen_source,
        local_files_only=_resolve_local_files_only(config),
        requested_dtype=requested_dtype,
        attn_implementation=attn_implementation,
    )
    _apply_loaded_model_runtime_defaults(model, attn_implementation=attn_implementation)
    total_encoder_layers = len(list(model.thinker.audio_tower.layers))
    raw_encoder_layers = deep_get(config, "model.encoder_layers", None)
    encoder_layer_indices = resolve_encoder_layer_indices(
        config, total_layers=total_encoder_layers
    )
    if encoder_layer_indices is not None:
        truncate_audio_tower(model, layer_indices=encoder_layer_indices)
        encoder_layers = len(encoder_layer_indices)
    else:
        encoder_layers = total_encoder_layers if raw_encoder_layers is None else int(raw_encoder_layers)
        truncate_audio_tower(model, encoder_layers=encoder_layers)
    _maybe_apply_causal_patch(model, config)
    if enable_gradient_checkpointing:
        _enable_gradient_checkpointing(model)
        model.config.use_cache = False
        model.thinker.config.use_cache = False
        model.thinker.model.config.use_cache = False
    else:
        model.config.use_cache = True
        model.thinker.config.use_cache = True
        model.thinker.model.config.use_cache = True
    if apply_trainability:
        apply_trainability_config(model, config)
    apply_lora_to_decoder(model, config)
    target_format = str(deep_get(config, "data.target_format", "qwen_asr") or "qwen_asr")
    wrapped = XAuTModel(
        model,
        processor=processor,
        qwen_source=qwen_source,
        encoder_layers=encoder_layers,
        target_format=target_format,
    )
    return XAuTArtifacts(model=wrapped, processor=processor)


def _import_and_patch_processor_12p5hz() -> None:
    """Apply the 12.5 Hz processor patch from ``x_aut.patch_12p5hz``.

    The function is class-level, idempotent, and safe to call multiple times.
    """
    from x_aut.patch_12p5hz import patch_processor_12p5hz

    patch_processor_12p5hz()


def _import_and_apply_frame_align_patch(audio_tower: nn.Module) -> None:
    """Apply the 12.5 Hz frame-align patch from ``x_aut.patch_12p5hz``.

    Used for a continuous teacher when ``distill.align_80ms`` is true but
    ``block_B_teacher`` is null.
    """
    from x_aut.patch_12p5hz import apply_frame_align_patch

    apply_frame_align_patch(audio_tower)


def build_distillation_artifacts_from_config(
    config: dict[str, Any],
    *,
    device: torch.device,
    apply_trainability: bool,
    enable_student_gradient_checkpointing: bool,
) -> XAuTDistillationArtifacts:
    qwen_source = _resolve_qwen_source(config)
    processor = build_processor_from_config(config)
    requested_dtype = resolve_model_dtype(deep_get(config, "model.load_dtype", "auto"), device=device)
    attn_implementation = resolve_attn_implementation(config, device=device)
    Qwen3ASRForConditionalGeneration, _ = _import_qwen_backend()
    model = _load_pretrained_qwen_model(
        Qwen3ASRForConditionalGeneration,
        qwen_source=qwen_source,
        local_files_only=_resolve_local_files_only(config),
        requested_dtype=requested_dtype,
        attn_implementation=attn_implementation,
    )
    _apply_loaded_model_runtime_defaults(model, attn_implementation=attn_implementation)
    student_total_encoder_layers = len(list(model.thinker.audio_tower.layers))





    align_80ms = bool(deep_get(config, "distill.align_80ms", False))
    if align_80ms:
        _import_and_patch_processor_12p5hz()
        from x_aut.distill import set_12p5hz_frame_mode
        set_12p5hz_frame_mode(True)
        _LOGGER.info("distill.align_80ms=true: enabled 12.5 Hz processor + collector frame mode")

    teacher_qwen_source = _resolve_teacher_qwen_source(config)
    teacher_local_files_only = _resolve_teacher_local_files_only(config)
    teacher_fix_mistral_regex = _resolve_teacher_fix_mistral_regex(config)
    teacher_requested_dtype = resolve_model_dtype(
        deep_get(config, "teacher.load_dtype", deep_get(config, "model.load_dtype", "auto")),
        device=device,
    )
    teacher_attn_implementation = _resolve_teacher_attn_implementation(config, device=device)
    teacher_model = _load_pretrained_qwen_model(
        Qwen3ASRForConditionalGeneration,
        qwen_source=teacher_qwen_source,
        local_files_only=teacher_local_files_only,
        requested_dtype=teacher_requested_dtype,
        attn_implementation=teacher_attn_implementation,
    )
    _apply_loaded_model_runtime_defaults(teacher_model, attn_implementation=teacher_attn_implementation)
    if (
        teacher_qwen_source == qwen_source
        and teacher_local_files_only == _resolve_local_files_only(config)
        and teacher_fix_mistral_regex == _resolve_fix_mistral_regex(config)
    ):
        teacher_processor = processor
    else:
        teacher_processor = build_processor_from_source(
            qwen_source=teacher_qwen_source,
            fix_mistral_regex=teacher_fix_mistral_regex,
            local_files_only=teacher_local_files_only,
        )
    _validate_teacher_compatibility(
        student_model=model,
        student_processor=processor,
        student_qwen_source=qwen_source,
        teacher_model=teacher_model,
        teacher_processor=teacher_processor,
        teacher_qwen_source=teacher_qwen_source,
    )
    _freeze_model_parameters(teacher_model)
    teacher_audio_tower = teacher_model.thinker.audio_tower
    teacher_encoder_layers = len(list(teacher_audio_tower.layers))







    teacher_block_B = deep_get(config, "distill.block_B_teacher", None)
    if teacher_block_B is not None:


        _apply_block_mask_patch(teacher_audio_tower, B=int(teacher_block_B), align_80ms=align_80ms)
    elif align_80ms:


        _import_and_apply_frame_align_patch(teacher_audio_tower)







    student_audio_tower = model.thinker.audio_tower
    student_layer_dim = int(getattr(student_audio_tower.config, "d_model", 0) or 0)
    teacher_layer_dim = int(getattr(teacher_audio_tower.config, "d_model", 0) or 0)
    student_bridge_dim = int(getattr(student_audio_tower.proj2, "out_features", 0) or 0)
    teacher_bridge_dim = int(getattr(teacher_audio_tower.proj2, "out_features", 0) or 0)
    projection_cfg = dict(deep_get(config, "distill.teacher_projection", {}) or {})
    projection_enabled_cfg = projection_cfg.get("enabled", "auto")
    needs_projection = (
        (student_layer_dim > 0 and teacher_layer_dim > 0 and student_layer_dim != teacher_layer_dim)
        or (student_bridge_dim > 0 and teacher_bridge_dim > 0 and student_bridge_dim != teacher_bridge_dim)
    )
    if str(projection_enabled_cfg).lower() == "auto":
        use_projection = needs_projection
    else:
        use_projection = bool(projection_enabled_cfg)

    teacher_layer_projection: AudioDistillProjection | None = None
    teacher_bridge_projection: AudioDistillProjection | None = None
    if use_projection:
        if student_layer_dim <= 0 or teacher_layer_dim <= 0:
            raise ValueError(
                "Cannot enable teacher_projection: could not resolve audio layer dims "
                f"(student={student_layer_dim}, teacher={teacher_layer_dim})"
            )
        if student_bridge_dim <= 0 or teacher_bridge_dim <= 0:
            raise ValueError(
                "Cannot enable teacher_projection: could not resolve audio bridge dims "
                f"(student={student_bridge_dim}, teacher={teacher_bridge_dim})"
            )
        bottleneck_dim = projection_cfg.get("bottleneck_dim", None)
        if bottleneck_dim is not None:
            bottleneck_dim = int(bottleneck_dim)
        teacher_layer_projection = AudioDistillProjection(
            teacher_dim=teacher_layer_dim,
            student_dim=student_layer_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=float(projection_cfg.get("dropout", 0.0) or 0.0),
        )
        teacher_bridge_projection = AudioDistillProjection(
            teacher_dim=teacher_bridge_dim,
            student_dim=student_bridge_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=float(projection_cfg.get("dropout", 0.0) or 0.0),
        )
        _LOGGER.info(
            "Attached cross-dimensional teacher projections | "
            "layer=%d->%d bridge=%d->%d bottleneck_dim=%s",
            teacher_layer_dim,
            student_layer_dim,
            teacher_bridge_dim,
            student_bridge_dim,
            bottleneck_dim if bottleneck_dim is not None else "auto",
        )

    raw_encoder_layers = deep_get(config, "model.encoder_layers", None)
    encoder_layer_indices = resolve_encoder_layer_indices(config, total_layers=student_total_encoder_layers)
    if encoder_layer_indices is not None:
        truncate_audio_tower(model, layer_indices=encoder_layer_indices)
        encoder_layers = len(encoder_layer_indices)
    else:
        encoder_layers = student_total_encoder_layers if raw_encoder_layers is None else int(raw_encoder_layers)
        truncate_audio_tower(model, encoder_layers=encoder_layers)


    _maybe_apply_causal_patch(model, config)





    student_block_B = deep_get(config, "distill.block_B_student", None)
    if student_block_B is not None:


        _apply_block_mask_patch(model.thinker.audio_tower, B=int(student_block_B), align_80ms=align_80ms)

    if enable_student_gradient_checkpointing:
        _enable_gradient_checkpointing(model)
        model.config.use_cache = False
        model.thinker.config.use_cache = False
        model.thinker.model.config.use_cache = False
    else:
        model.config.use_cache = True
        model.thinker.config.use_cache = True
        model.thinker.model.config.use_cache = True
    if apply_trainability:
        apply_trainability_config(model, config)
    apply_lora_to_decoder(model, config)
    target_format = str(deep_get(config, "data.target_format", "qwen_asr") or "qwen_asr")
    wrapped = XAuTModel(
        model,
        processor=processor,
        qwen_source=qwen_source,
        encoder_layers=encoder_layers,
        target_format=target_format,
    )
    teacher_wrapped = XAuTModel(
        teacher_model,
        processor=teacher_processor,
        qwen_source=teacher_qwen_source,
        encoder_layers=teacher_encoder_layers,
        target_format=target_format,
    )
    return XAuTDistillationArtifacts(
        student=wrapped,
        processor=processor,
        teacher=teacher_wrapped,
        teacher_audio_tower=teacher_audio_tower,
        teacher_encoder_layers=teacher_encoder_layers,
        teacher_layer_projection=teacher_layer_projection,
        teacher_bridge_projection=teacher_bridge_projection,
    )


def _cast_features_for_audio_tower(audio_tower: nn.Module, input_features: torch.Tensor) -> torch.Tensor:
    target_dtype = audio_tower.conv2d1.weight.dtype
    if input_features.dtype == target_dtype:
        return input_features
    return input_features.to(dtype=target_dtype)


def extract_audio_features(
    audio_tower: nn.Module,
    *,
    input_features: torch.Tensor,
    feature_attention_mask: torch.Tensor,
) -> torch.Tensor:
    input_features = _cast_features_for_audio_tower(audio_tower, input_features)
    feature_lens = torch.sum(feature_attention_mask, dim=1)







    if input_features.dim() == 3:
        packed_samples = [
            input_features[i, :, : int(feature_lens[i])]
            for i in range(input_features.shape[0])
        ]
        input_features = torch.cat(packed_samples, dim=1).contiguous()


    try:
        audio_output = audio_tower(input_features, feature_lens=feature_lens)
        return audio_output.last_hidden_state
    except Exception:




        if not getattr(extract_audio_features, "_warned_batch_fallback", False):
            extract_audio_features._warned_batch_fallback = True
            _LOGGER.warning(
                "Audio tower batch forward failed; falling back to per-sample "
                "(batch>1 re-runs bridge params and breaks DDP trainable bridge)",
                exc_info=True,
            )


    audio_features: list[torch.Tensor] = []
    for input_feature, feature_len in zip(input_features.T, feature_lens):
        input_feature = input_feature[:, : int(feature_len)]
        audio_output = audio_tower(
            input_feature.unsqueeze(0),
            feature_lens=feature_len.unsqueeze(0),
        )
        audio_features.append(audio_output.last_hidden_state)
    return torch.cat(audio_features, dim=0)


def _compose_thinker_inputs_embeds_with_audio_features(
    thinker: nn.Module,
    *,
    input_ids: torch.Tensor,
    audio_features: torch.Tensor,
    discrete_token_ids: torch.Tensor | None = None,
    discrete_embedding: nn.Module | None = None,
) -> torch.Tensor:
    inputs_embeds = thinker.get_input_embeddings()(input_ids)
    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
    audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
    expected_rows = int(audio_mask[..., 0].sum().item())
    if int(audio_features.size(0)) != expected_rows:
        raise ValueError(
            "Audio feature rows do not match decoder placeholder count: "
            f"features={int(audio_features.size(0))}, placeholders={expected_rows}"
        )
    if audio_features.dim() != 2 or int(audio_features.size(-1)) != int(inputs_embeds.size(-1)):
        raise ValueError(
            "Audio feature shape does not match decoder hidden size: "
            f"features={tuple(audio_features.shape)}, hidden_size={int(inputs_embeds.size(-1))}"
        )
    if discrete_token_ids is not None and discrete_embedding is not None:


        audio_embeds = discrete_embedding(discrete_token_ids.to(inputs_embeds.device))
        audio_embeds = audio_embeds.to(inputs_embeds.dtype)
        if tuple(audio_embeds.shape) != tuple(audio_features.shape):
            raise ValueError(
                "Discrete audio embedding shape does not match continuous features: "
                f"embeds={tuple(audio_embeds.shape)}, features={tuple(audio_features.shape)}"
            )
        audio_features = audio_embeds
    return inputs_embeds.masked_scatter(audio_mask, audio_features)


def thinker_forward_from_audio_features(
    thinker: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_features: torch.Tensor,
    labels: torch.Tensor | None = None,
    discrete_token_ids: torch.Tensor | None = None,
    discrete_embedding: nn.Module | None = None,
):
    inputs_embeds = _compose_thinker_inputs_embeds_with_audio_features(
        thinker,
        input_ids=input_ids,
        audio_features=audio_features,
        discrete_token_ids=discrete_token_ids,
        discrete_embedding=discrete_embedding,
    )
    return thinker(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        return_dict=True,
    )


def thinker_generate_from_audio_features(
    thinker: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_features: torch.Tensor,
    generation_kwargs: dict[str, Any] | None = None,
    discrete_token_ids: torch.Tensor | None = None,
    discrete_embedding: nn.Module | None = None,
) -> torch.Tensor:
    inputs_embeds = _compose_thinker_inputs_embeds_with_audio_features(
        thinker,
        input_ids=input_ids,
        audio_features=audio_features,
        discrete_token_ids=discrete_token_ids,
        discrete_embedding=discrete_embedding,
    )
    return thinker.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **dict(generation_kwargs or {}),
    )


def load_model_state(model: XAuTModel, state_dict: dict[str, Any]) -> None:
    model.load_state_dict(state_dict, strict=True)


def _extract_checkpoint_model_payload(
    checkpoint: Any,
    *,
    checkpoint_path: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a mapping: {checkpoint_path}")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint is missing a model state_dict: {checkpoint_path}")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and not isinstance(checkpoint_config, dict):
        checkpoint_config = None
    return state_dict, checkpoint_config


def _resolve_encoder_signature(config: dict[str, Any] | None) -> tuple[int | None, tuple[int, ...] | None]:
    if not isinstance(config, dict):
        return None, None
    raw_indices = deep_get(config, "model.encoder_layer_indices", None)
    if isinstance(raw_indices, (list, tuple)) and raw_indices:
        resolved_indices = tuple(int(value) for value in raw_indices)
        return len(resolved_indices), resolved_indices
    raw_layers = deep_get(config, "model.encoder_layers", None)
    if raw_layers is None:
        return None, None
    return int(raw_layers), None


_AUDIO_TOWER_LAYER_RE = re.compile(r"^(.*audio_tower\.layers\.)(\d+)(\..*)$")


def _remap_audio_tower_layers(
    state_dict: dict[str, torch.Tensor],
    checkpoint_indices: tuple[int, ...],
    current_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Remap ``audio_tower.layers`` keys from checkpoint positions to current model positions.

    Both *checkpoint_indices* and *current_indices* are 1-based original layer indices
    (sorted ascending).  *current_indices* must be a subset of *checkpoint_indices*.
    Layers present in the checkpoint but absent from the current model are dropped.
    """
    ckpt_pos_of = {orig_layer: pos for pos, orig_layer in enumerate(checkpoint_indices)}
    current_pos_of = {orig_layer: pos for pos, orig_layer in enumerate(current_indices)}

    remap: dict[int, int] = {}
    for orig_layer in current_indices:
        ckpt_pos = ckpt_pos_of.get(orig_layer)
        if ckpt_pos is not None:
            remap[ckpt_pos] = current_pos_of[orig_layer]
    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        match = _AUDIO_TOWER_LAYER_RE.match(key)
        if match is not None:
            prefix, pos_str, suffix = match.groups()
            ckpt_pos = int(pos_str)
            current_pos = remap.get(ckpt_pos)
            if current_pos is not None:
                new_state_dict[f"{prefix}{current_pos}{suffix}"] = value

        else:
            new_state_dict[key] = value
    return new_state_dict


def validate_checkpoint_compatibility(
    model: nn.Module,
    *,
    checkpoint_config: dict[str, Any] | None,
    current_config: dict[str, Any] | None,
    checkpoint_path: str,
    allow_layer_subset: bool = False,
) -> None:
    if checkpoint_config is None:
        return
    checkpoint_layers, checkpoint_indices = _resolve_encoder_signature(checkpoint_config)
    current_layers, current_indices = _resolve_encoder_signature(current_config)
    model_core = unwrap_model(model)
    resolved_current_layers = current_layers
    if resolved_current_layers is None:
        encoder_layers = getattr(model_core, "encoder_layers", None)
        resolved_current_layers = None if encoder_layers is None else int(encoder_layers)
    is_subset = (
        allow_layer_subset
        and checkpoint_indices is not None
        and current_indices is not None
        and set(current_indices).issubset(set(checkpoint_indices))
    )
    if (
        checkpoint_layers is not None
        and resolved_current_layers is not None
        and checkpoint_layers != resolved_current_layers
        and not is_subset
    ):
        raise ValueError(
            "Checkpoint encoder depth does not match the current model: "
            f"checkpoint={checkpoint_layers}, current={resolved_current_layers}, path={checkpoint_path}"
        )
    if (
        checkpoint_indices is not None
        and current_indices is not None
        and checkpoint_indices != current_indices
        and not is_subset
    ):
        raise ValueError(
            "Checkpoint encoder layer indices do not match the current model: "
            f"checkpoint={checkpoint_indices}, current={current_indices}, path={checkpoint_path}"
        )


def _checkpoint_contains_lora(state_dict: dict[str, Any]) -> bool:
    return any("lora_" in key for key in state_dict)


def _model_contains_lora(model: nn.Module) -> bool:
    return any("lora_" in name for name, _ in model.named_parameters())


def _remap_lora_base_weights(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Remap bare ``{module}.weight`` keys to ``{module}.base_layer.weight``.

    LoRA wrapping renames e.g. ``q_proj.weight`` to ``q_proj.base_layer.weight``;
    a base (non-LoRA) checkpoint still stores the bare name.  Only keys that
    would otherwise miss the wrapped parameter are renamed, so checkpoints that
    already carry LoRA keys (or plain non-LoRA targets) stay untouched.
    """
    model_param_names = set(model.state_dict().keys())
    remapped: dict[str, torch.Tensor] = {}
    renamed = 0
    for key, value in state_dict.items():
        if key.endswith((".weight", ".bias")):
            stem, leaf = key.rsplit(".", 1)
            base_key = f"{stem}.base_layer.{leaf}"
            if key not in model_param_names and base_key in model_param_names:
                remapped[base_key] = value
                renamed += 1
                continue
        remapped[key] = value
    if renamed:
        _LOGGER.info("Remapped %d keys to LoRA base_layer names", renamed)
    return remapped


def initialize_model_from_checkpoint(
    model: nn.Module,
    *,
    checkpoint_path: str,
    device: torch.device | str,
    current_config: dict[str, Any] | None = None,
    checkpoint_payload: Any | None = None,
) -> dict[str, Any] | None:
    checkpoint = (
        torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint_payload is None
        else checkpoint_payload
    )
    state_dict, checkpoint_config = _extract_checkpoint_model_payload(checkpoint, checkpoint_path=checkpoint_path)
    checkpoint_layers, checkpoint_indices = _resolve_encoder_signature(checkpoint_config)
    current_layers, current_indices = _resolve_encoder_signature(current_config)
    need_remap = (
        checkpoint_indices is not None
        and current_indices is not None
        and checkpoint_indices != current_indices
        and set(current_indices).issubset(set(checkpoint_indices))
    )
    if need_remap:
        state_dict = _remap_audio_tower_layers(state_dict, checkpoint_indices, current_indices)
        _LOGGER.info(
            "Remapping audio_tower layers from checkpoint | checkpoint_indices=%s current_indices=%s path=%s",
            checkpoint_indices,
            current_indices,
            checkpoint_path,
        )
    validate_checkpoint_compatibility(
        model,
        checkpoint_config=checkpoint_config,
        current_config=current_config,
        checkpoint_path=checkpoint_path,
        allow_layer_subset=need_remap,
    )

    model_core = unwrap_model(model)
    model_has_lora = _model_contains_lora(model_core)
    ckpt_has_lora = _checkpoint_contains_lora(state_dict)

    if model_has_lora and not ckpt_has_lora:


        _LOGGER.warning(
            "Loading non-LoRA checkpoint into LoRA model | base weights will be loaded; "
            "LoRA adapters remain randomly initialized | path=%s",
            checkpoint_path,
        )
        missing, unexpected = model_core.load_state_dict(
            _remap_lora_base_weights(state_dict, model_core),
            strict=False,
        )
        non_lora_missing = [key for key in missing if "lora_" not in key]
        if non_lora_missing:
            raise RuntimeError(
                f"Non-LoRA keys missing when loading checkpoint into LoRA model: {non_lora_missing[:20]}"
            )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading checkpoint into LoRA model: {unexpected[:20]}"
            )
        _LOGGER.info(
            "Loaded base weights into LoRA model | lora_weights_missing=%d",
            len(missing),
        )
    else:
        model_core.load_state_dict(state_dict, strict=True)

    return checkpoint_config
