"""The fixed refined-to-basic Schwartz value mapping used by the benchmark."""

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


def normalize_value(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


_BASIC_LOOKUP = {normalize_value(value): value for value in BASIC_VALUES}
_FINE_LOOKUP = {normalize_value(value): basic for value, basic in FINE_TO_BASIC.items()}


def to_basic(value: str) -> str:
    """Map a refined or already-basic label to its canonical basic value."""
    key = normalize_value(value)
    if key in _BASIC_LOOKUP:
        return _BASIC_LOOKUP[key]
    if key in _FINE_LOOKUP:
        return _FINE_LOOKUP[key]
    raise KeyError(f"Unknown Schwartz value: {value!r}")


def value_slug(value: str) -> str:
    return to_basic(value).lower()
