"""Shared mapping from the refined Schwartz values to the ten basic values."""

from __future__ import annotations


BASIC_VALUES = (
    "Self_direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism",
)

FINE_TO_BASIC = {
    "Self_direction_thought": "Self_direction",
    "Self_direction_action": "Self_direction",
    "Stimulation": "Stimulation",
    "Hedonism": "Hedonism",
    "Achievement": "Achievement",
    "Power_dominance": "Power",
    "Power_resources": "Power",
    "Face": "Power",
    "Security_personal": "Security",
    "Security_societal": "Security",
    "Conformity_rules": "Conformity",
    "Conformity_interpersonal": "Conformity",
    "Tradition": "Tradition",
    "Humility": "Tradition",
    "Benevolence_caring": "Benevolence",
    "Benevolence_dependability": "Benevolence",
    "Universalism_concern": "Universalism",
    "Universalism_nature": "Universalism",
    "Universalism_tolerance": "Universalism",
    "Universalism_objectivity": "Universalism",
}


def _normalized(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


_BASIC_LOOKUP = {_normalized(value): value for value in BASIC_VALUES}
_FINE_LOOKUP = {_normalized(value): basic for value, basic in FINE_TO_BASIC.items()}


def canonical_basic_value(value: str) -> str:
    """Return a canonical ten-value label from a basic or refined label."""
    normalized = _normalized(value)
    if normalized in _BASIC_LOOKUP:
        return _BASIC_LOOKUP[normalized]
    if normalized in _FINE_LOOKUP:
        return _FINE_LOOKUP[normalized]
    raise KeyError(f"Unknown Schwartz value: {value!r}")


def basic_value_for_fine(value: str) -> str:
    normalized = _normalized(value)
    if normalized not in _FINE_LOOKUP:
        raise KeyError(f"Unknown refined Schwartz value: {value!r}")
    return _FINE_LOOKUP[normalized]


def selected_basic_values(requested: list[str]) -> list[str]:
    if requested == ["all"]:
        return list(BASIC_VALUES)
    values = [canonical_basic_value(value) for value in requested]
    return list(dict.fromkeys(values))


def value_slug(value: str) -> str:
    return canonical_basic_value(value).lower()
