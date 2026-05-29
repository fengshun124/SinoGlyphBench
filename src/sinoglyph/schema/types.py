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


class Modality(StringEnum):
    TEXT = "text"
    IMAGE = "image"


class ObfuscationType(StringEnum):
    DECOMPOSITION = "decomposition"
    CROSS_SCRIPT = "cross_script"


class ObfuscationScope(StringEnum):
    ORIGINAL = "original"
    ANCHOR_ONLY = "anchor_only"
    BACKGROUND_ONLY = "background_only"
    FULL = "full"


class ModerationLabel(StringEnum):
    HOSTILE = "hostile"
    ABUSIVE = "abusive"
    BENIGN = "benign"
    CONTEXT_DEPENDENT = "context_dependent"
