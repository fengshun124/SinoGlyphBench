from __future__ import annotations

import unicodedata as ud
from typing import cast

from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.corpus import PerturbedCorpusEntry
from sinoglyph.schema.evaluation import EvaluationTask
from sinoglyph.schema.types import TaskSource, TaskVariant
from sinoglyph.schema.utils import require_list, require_mapping, require_string


def wrap_evaluation_text(text: str, render: JsonObject | None) -> str:
    if render is None or not render.get("line_breaks"):
        return text
    max_chars = cast(int, render["line_break_max_chars"])
    return "\n".join(_wrap_line(line, max_chars) for line in text.split("\n"))


def wrap_task_input(
    entry: PerturbedCorpusEntry,
    task: EvaluationTask,
    input_text: str,
    render: JsonObject | None,
) -> str:
    if render is None or not render.get("line_breaks"):
        return input_text
    wrapped_original = wrap_evaluation_text(entry.text, render)
    if input_text == entry.text:
        return wrapped_original
    chunks = _build_task_input_chunks(entry, task, input_text)
    if chunks is None:
        return input_text
    output: list[str] = []
    index = 0
    for character in wrapped_original:
        if character == "\n":
            output.append("\n")
            continue
        output.append(chunks[index])
        index += 1
    return "".join(output)


def _build_task_input_chunks(
    entry: PerturbedCorpusEntry,
    task: EvaluationTask,
    input_text: str,
) -> list[str] | None:
    replacements = _map_task_replacements_by_character(entry, task)
    chunks: list[str] = []
    cursor = 0
    for character in entry.text:
        replacement = replacements.get(character)
        if replacement is not None and input_text.startswith(replacement, cursor):
            chunks.append(replacement)
            cursor += len(replacement)
            continue
        if input_text.startswith(character, cursor):
            chunks.append(character)
            cursor += len(character)
            continue
        return None
    return chunks if cursor == len(input_text) else None


def _map_task_replacements_by_character(
    entry: PerturbedCorpusEntry,
    task: EvaluationTask,
) -> dict[str, str]:
    if task.source not in {TaskSource.DECOMPOSITION, TaskSource.PERTURBATION}:
        return {}
    if task.variant == TaskVariant.ORIGINAL:
        return {}
    groups = {
        TaskVariant.ANCHOR_ONLY: ("anchor",),
        TaskVariant.NON_ANCHOR_ONLY: ("non_anchor",),
        TaskVariant.FULL: ("anchor", "non_anchor"),
    }[task.variant]
    field = task.source.value
    substitutions = require_mapping(entry.substitutions, "entry.substitutions")
    replacements: dict[str, str] = {}
    for group in groups:
        for raw_item in require_list(
            substitutions[group], f"entry.substitutions.{group}"
        ):
            item = require_mapping(raw_item, f"entry.substitutions.{group}[]")
            replacements[
                require_string(item["character"], "substitution.character")
            ] = require_string(
                item[field],
                f"substitution.{field}",
            )
    return replacements


def _wrap_line(text: str, max_chars: int) -> str:
    lines: list[str] = []
    rest = text
    while len(rest) > max_chars:
        break_at = _find_line_break(rest, max_chars)
        if break_at is None:
            break
        end, next_start = break_at
        line = rest[:end].rstrip()
        if line:
            lines.append(line)
        rest = rest[next_start:].lstrip()
        if not rest:
            break
    if rest:
        lines.append(rest)
    return "\n".join(lines)


def _find_line_break(text: str, max_chars: int) -> tuple[int, int] | None:
    candidates = _build_line_break_candidates(text)
    before = [candidate for candidate in candidates if candidate[0] <= max_chars]
    if before:
        return before[-1]
    return next(
        (candidate for candidate in candidates if candidate[0] > max_chars), None
    )


def _build_line_break_candidates(text: str) -> list[tuple[int, int]]:
    return [
        (index, index + 1) if character.isspace() else (index + 1, index + 1)
        for index, character in enumerate(text)
        if character != "\n"
        and (character.isspace() or ud.category(character).startswith("P"))
    ]
