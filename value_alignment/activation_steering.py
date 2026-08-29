"""Architecture-aware hooks for contrastive activation steering."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

import torch


LAYER_PATHS = (
    ("model", "layers"),
    ("transformer", "h"),
    ("transformer", "layers"),
    ("gpt_neox", "layers"),
    ("model", "decoder", "layers"),
)
ATTENTION_NAMES = ("self_attn", "self_attention", "attn", "attention")


def _resolve_path(root: Any, path: tuple[str, ...]) -> Any | None:
    current = root
    for name in path:
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def decoder_layers(model) -> tuple[list, str]:
    roots = [(model, "")]
    if hasattr(model, "get_base_model"):
        roots.append((model.get_base_model(), "base_model."))
    if hasattr(model, "base_model"):
        roots.append((model.base_model, "base_model."))

    checked = set()
    for root, root_name in roots:
        if id(root) in checked:
            continue
        checked.add(id(root))
        for path in LAYER_PATHS:
            layers = _resolve_path(root, path)
            if layers is not None and len(layers) > 0:
                return list(layers), root_name + ".".join(path)
    raise ValueError(
        "Could not locate decoder layers. Supported layouts include model.layers, "
        "transformer.h, transformer.layers, and gpt_neox.layers."
    )


def attention_module(layer):
    for name in ATTENTION_NAMES:
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise ValueError(f"Could not locate an attention module on {type(layer).__name__}.")


def steering_module(model, site: str, layer_index: int):
    layers, layer_path = decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"Layer index {layer_index} is outside 0-{len(layers) - 1}.")
    layer = layers[layer_index]
    if site == "block":
        return layer, f"{layer_path}.{layer_index}", len(layers)
    if site == "attn":
        module, attention_name = attention_module(layer)
        return module, f"{layer_path}.{layer_index}.{attention_name}", len(layers)
    raise ValueError("Steering site must be 'block' or 'attn'.")


def hidden_from_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported hooked module output type: {type(output).__name__}")


def replace_hidden_output(output, hidden: torch.Tensor):
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unsupported hooked module output type: {type(output).__name__}")


class SteeringHook:
    """Add one steering vector from a chosen token position onward."""

    def __init__(
        self,
        model,
        site: str,
        layer_index: int,
        vector: torch.Tensor,
        strength: float = 1.0,
    ) -> None:
        self.module, self.module_name, self.layer_count = steering_module(
            model,
            site,
            layer_index,
        )
        self.vector = vector.detach().flatten().cpu()
        self.strength = float(strength)
        self.start_positions: list[int] | None = None
        self._handle = None

    def set_start_positions(self, positions: list[int] | None) -> None:
        """Set per-row starts; None steers only the final token of every forward call."""
        self.start_positions = positions

    def _hook(self, _module, _inputs, output):
        hidden = hidden_from_output(output)
        if hidden.shape[-1] != self.vector.numel():
            raise ValueError(
                f"Steering vector width {self.vector.numel()} does not match hidden width "
                f"{hidden.shape[-1]} at {self.module_name}."
            )
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype) * self.strength
        modified = hidden.clone()
        if self.start_positions is None:
            modified[:, -1, :] += vector
        else:
            if len(self.start_positions) != hidden.shape[0]:
                raise ValueError("Steering start-position count does not match the batch size.")
            for row_index, start in enumerate(self.start_positions):
                normalized_start = max(0, min(int(start), hidden.shape[1] - 1))
                modified[row_index, normalized_start:, :] += vector
        return replace_hidden_output(output, modified)

    def __enter__(self) -> "SteeringHook":
        if self._handle is not None:
            raise RuntimeError("Steering hook is already active.")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_vector_file(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_steering_selection(path: Path) -> dict:
    selection = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "target_value", "site", "selected_layer", "vector_file"}
    missing = sorted(required - set(selection))
    if missing:
        raise ValueError(f"Steering selection is missing required fields: {missing}")
    vector_path = Path(selection["vector_file"])
    if not vector_path.is_absolute():
        vector_path = (path.parent / vector_path).resolve()
    vectors = load_vector_file(vector_path)
    site = selection["site"]
    layer = int(selection["selected_layer"])
    if site not in {"block", "attn"} or site not in vectors:
        raise ValueError(f"Steering vector file does not contain site {site!r}.")
    if not 0 <= layer < len(vectors[site]):
        raise ValueError(
            f"Selected steering layer {layer} is outside vector file range "
            f"0-{len(vectors[site]) - 1}."
        )
    vector_metadata = vectors.get("metadata", {})
    for field in ("model", "target_value"):
        if vector_metadata.get(field) != selection[field]:
            raise ValueError(
                f"Steering selection {field}={selection[field]!r} does not match "
                f"vector metadata {vector_metadata.get(field)!r}."
            )
    return {
        **selection,
        "vector_file": str(vector_path),
        "vector": vectors[site][layer],
        "site": site,
        "selected_layer": layer,
        "strength": float(selection.get("strength", 1.0)),
    }
