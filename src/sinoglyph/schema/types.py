from enum import Enum
from typing import TypeVar

EnumT = TypeVar("EnumT", bound="StringEnum")


class StringEnum(str, Enum):
    @classmethod
    def from_value(cls: type[EnumT], value: object, context: str) -> EnumT:
        if not isinstance(value, str):
            raise ValueError(f"{context} expects a string")
        try:
            return cls(value)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"{context} must be one of: {choices}") from exc


class InputType(StringEnum):
    TEXT = "text"
    IMAGE = "image"


class TaskSource(StringEnum):
    TEXT = "text"
    DECOMPOSITION = "decomposition"
    PERTURBATION = "perturbation"


class TaskVariant(StringEnum):
    ORIGINAL = "original"
    ANCHOR_ONLY = "anchor_only"
    NON_ANCHOR_ONLY = "non_anchor_only"
    FULL = "full"


class ModerationLabel(StringEnum):
    HOSTILE = "hostile"
    ABUSIVE = "abusive"
    BENIGN = "benign"
    CONTEXT_DEPENDENT = "context_dependent"
