from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.types import StringEnum

EnumT = TypeVar("EnumT", bound=StringEnum)


def require_mapping(value: object, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{context} expects a mapping")
    return value


def require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} expects a list")
    return value


def require_keys(mapping: JsonObject, keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys).difference(mapping))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{context} is missing required field(s): {names}")


def require_string(value: object, context: str, *, non_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} expects a string")
    if non_empty and not value:
        raise ValueError(f"{context} expects a non-empty string")
    return value


def optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return require_string(value, context)


def require_enum(enum_type: type[EnumT], value: object, context: str) -> EnumT:
    return enum_type.from_value(value, context)


def require_positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} expects a positive integer")
    if value <= 0:
        raise ValueError(f"{context} expects a positive integer")
    return value


def require_non_negative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} expects a non-negative integer")
    if value < 0:
        raise ValueError(f"{context} expects a non-negative integer")
    return value


def require_number(value: object, context: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} expects a number")
    return value


def require_boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} expects a boolean")
    return value


def require_file_path(value: object, context: str) -> Path:
    return Path(require_string(value, context)).expanduser()
