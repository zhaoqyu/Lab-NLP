"""Lazy model loading and architecture adapters for Colab-sized experiments."""

from __future__ import annotations

import gc
import platform
from pathlib import Path

from .config import ProjectConfig


def torch_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(config: ProjectConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        use_fast=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_kwargs(config: ProjectConfig) -> dict:
    import torch
    from transformers import BitsAndBytesConfig

    kwargs = {
        "torch_dtype": torch_dtype(),
        "trust_remote_code": config.model.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if config.model.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit loading requires a CUDA Colab runtime")
        kwargs.update(
            {
                "device_map": {"": torch.cuda.current_device()},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype(),
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    return kwargs


def load_base_model(config: ProjectConfig):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(config.model.id, **_load_kwargs(config))
    model.config.use_cache = False
    return model


def load_adapter_model(config: ProjectConfig, adapter_path: Path):
    from peft import PeftModel

    model = load_base_model(config)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    return model


def _truncate_middle(tokenizer, text: str, max_tokens: int) -> str:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return text
    head = max_tokens // 2
    tail = max_tokens - head
    return tokenizer.decode(token_ids[:head] + token_ids[-tail:], skip_special_tokens=True)


def format_chat_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    *,
    max_user_tokens: int | None = None,
) -> str:
    if max_user_tokens is not None:
        user_prompt = _truncate_middle(tokenizer, user_prompt, max_user_tokens)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


def decoder_layers(model):
    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for chain in candidates:
        value = base
        try:
            for attribute in chain:
                value = getattr(value, attribute)
            if len(value):
                return value
        except (AttributeError, TypeError):
            continue
    raise TypeError(f"Unsupported decoder architecture: {type(base).__name__}")


def resolve_layer_indices(model, specifications: list[float | int]) -> list[int]:
    count = len(decoder_layers(model))
    indices = []
    for spec in specifications:
        if isinstance(spec, float) and 0.0 <= spec <= 1.0:
            index = round(spec * (count - 1))
        else:
            index = int(spec)
            if index < 0:
                index += count
        if not 0 <= index < count:
            raise IndexError(f"Layer {spec} resolves to {index}, outside 0..{count - 1}")
        indices.append(index)
    return list(dict.fromkeys(indices))


def trainable_parameters(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "fraction": trainable / total if total else 0.0}


def hardware_info() -> dict:
    import torch

    info = {"python_platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def release_model(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
