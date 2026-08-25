import json
import os
import re
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeAlias

import tomllib
from dotenv import load_dotenv

PathLike: TypeAlias = str | os.PathLike[str]


def load_json(file_path: PathLike) -> Any:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg} in {file_path}", exc.doc, exc.pos
        ) from exc


def save_json(data: Any, file_path: PathLike, *, warn_overwrite: bool = True) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=4)
    _atomic_write_text(text, file_path, warn_overwrite=warn_overwrite)


def parse_json_response(text: str) -> Any:
    stripped = _strip_json_markdown_fence(text)
    decoder = json.JSONDecoder()
    direct_error: json.JSONDecodeError | None = None

    # Layer 1: Try strict JSON parsing
    try:
        return decoder.decode(stripped)
    except json.JSONDecodeError as exc:
        direct_error = exc

    # Layer 2: Try scanning for JSON objects/arrays and strict decode
    for start, character in enumerate(stripped):
        if character not in "[{":
            continue
        candidate = stripped[start:]
        try:
            parsed, _ = decoder.raw_decode(candidate)
            return parsed
        except json.JSONDecodeError:
            continue

    # Layer 3: Field-by-field extraction for unescaped quotes in values
    parsed = _extract_fields_from_malformed(stripped)
    if parsed is not None:
        return parsed

    snippet = stripped.replace("\n", "\\n")[:160]
    if direct_error is None:
        direct_error = json.JSONDecodeError("Expecting value", stripped, 0)
    raise json.JSONDecodeError(
        f"{direct_error.msg}; no JSON object or array found in response: {snippet!r}",
        direct_error.doc,
        direct_error.pos,
    ) from direct_error


def _strip_json_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        last_line = lines[-1].strip()
        if last_line == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_fields_from_malformed(text: str) -> dict[str, str] | None:
    """
    Extract required fields from JSON with unescaped quotes in values.

    This handles cases where LLMs output valid semantic content but with
    quotes inside string values that aren't properly escaped, e.g.:
    {"interpretation": "呼吁...定义为"正常"。..."}

    Only proceeds if all required fields are found with non-empty values.
    Returns None if structure is too malformed to reliably extract.
    """
    required_fields = ["read_text", "recovered_text", "interpretation", "judge"]
    output: dict[str, str] = {}

    for field in required_fields:
        # Look for pattern: "field_name": "value"
        pattern = rf'"{field}"\s*:\s*"'
        match = re.search(pattern, text)
        if not match:
            return None

        # Start searching for closing quote after the opening quote
        value_start = match.end()
        if value_start >= len(text):
            return None

        # Find the closing quote, handling escaped quotes
        value_chars = []
        i = value_start
        while i < len(text):
            char = text[i]
            if char == "\\":
                # Escaped character - take both \ and next char
                if i + 1 < len(text):
                    decoded_escape, consumed = _decode_json_escape(text, i)
                    value_chars.append(decoded_escape)
                    i += consumed
                else:
                    i += 1
            elif char == '"':
                # Potential end quote - verify it's not followed by another quote
                # (in case of embedded quotes within the value)
                whitespace_start = i + 1
                i += 1
                # Check if next meaningful character is comma or brace (end of field)
                while i < len(text) and text[i] in " \n\t":
                    i += 1
                if i < len(text) and text[i] in ",}]":
                    # This is the real closing quote
                    break
                else:
                    # This was an embedded quote, include it and continue
                    value_chars.append('"')
                    value_chars.append(text[whitespace_start:i])
            else:
                value_chars.append(char)
                i += 1

        value = "".join(value_chars).strip()
        if not value:
            return None
        output[field] = value

    return output if len(output) == len(required_fields) else None


def _decode_json_escape(text: str, slash_index: int) -> tuple[str, int]:
    escape = text[slash_index + 1]
    if escape == "u" and slash_index + 6 <= len(text):
        hex_digits = text[slash_index + 2 : slash_index + 6]
        try:
            return chr(int(hex_digits, 16)), 6
        except ValueError:
            pass
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    return escapes.get(escape, f"\\{escape}"), 2


def load_toml(file_path: PathLike) -> dict[str, Any]:
    with open(file_path, "rb") as f:
        return tomllib.load(f)


def load_env_file(
    file_path: PathLike = ".env",
    *,
    override: bool = False,
    base_dir: PathLike | None = None,
) -> Path:

    env_path = Path(file_path).expanduser()
    if base_dir is not None and not env_path.is_absolute():
        env_path = Path(base_dir) / env_path
    env_path = env_path.resolve(strict=False)
    load_dotenv(dotenv_path=env_path, override=override)
    return env_path


def _atomic_write_text(
    text: str,
    file_path: PathLike,
    *,
    warn_overwrite: bool = True,
) -> None:
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if warn_overwrite and output_path.exists():
        warnings.warn(
            f"File {output_path} already exists and will be overwritten",
            UserWarning,
        )

    with NamedTemporaryFile(
        "w", dir=output_path.parent, encoding="utf-8", delete=False
    ) as f:
        temp_name = f.name
        f.write(text)
        f.write("\n")

    os.replace(temp_name, output_path)
