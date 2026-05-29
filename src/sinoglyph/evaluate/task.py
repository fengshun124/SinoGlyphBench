from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, cast

from sinoglyph.evaluate.cache import build_safe_name
from sinoglyph.evaluate.prompt import build_image_message, build_text_message
from sinoglyph.evaluate.text import wrap_task_input
from sinoglyph.io import parse_json_response
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.corpus import ObfuscatedCorpusEntry
from sinoglyph.schema.evaluation import EvaluationTask, validate_response_instance
from sinoglyph.schema.types import Modality, ModerationLabel

if TYPE_CHECKING:
    from sinoglyph.render import TextRenderConfig


@dataclass(frozen=True)
class _EvaluationFailure:
    kind: str
    retryable: bool
    message: str
    status_code: int | None = None
    request_id: str | None = None


def parse_response(raw: str, response_schema: JsonObject) -> JsonObject:
    parsed = parse_json_response(raw)
    try:
        return validate_response_instance(parsed, response_schema)
    except ValueError as exc:
        raise ValueError(f"response JSON does not match schema: {exc}") from exc


def run_task_evaluation(
    entry: JsonObject,
    task: EvaluationTask,
    max_tries: int,
    chat_client: object,
    text_prompt: str,
    image_prompt: str,
    response_schema: JsonObject,
    render_mapping: JsonObject | None,
    render_config: TextRenderConfig | None,
    cache_dir: Path,
) -> JsonObject:
    corpus_entry = ObfuscatedCorpusEntry.parse_mapping(entry)
    raw_input_text = corpus_entry.input_text(task.obfuscation_type.value, task.scope)
    input_text = (
        wrap_task_input(corpus_entry, task, raw_input_text, render_mapping)
        if task.modality == Modality.IMAGE
        else raw_input_text
    )
    obf_density = corpus_entry.obf_density(task.scope)
    raw = ""
    parse_error: str | None = None
    request_error: str | None = None
    failure: _EvaluationFailure | None = None
    parsed = None
    predicted_label = None
    try_count = 0
    image_path = None
    if task.modality == Modality.IMAGE:
        if render_config is None:
            raise ValueError("render config is required for image tasks")
        image_path = _render_task_image(
            cache_dir, corpus_entry, task, input_text, render_config
        )

    for try_index in range(1, max_tries + 1):
        try_count = try_index
        try:
            if task.modality == Modality.TEXT:
                raw = cast(
                    str,
                    chat_client.chat_once(
                        text=build_text_message(text_prompt, input_text)
                    ),
                )
            else:
                if image_path is None:
                    raise AssertionError("image path was not prepared for image task")
                raw = cast(
                    str,
                    chat_client.chat_once(
                        text=build_image_message(image_prompt), images=[str(image_path)]
                    ),
                )
        except Exception as exc:
            failure = _classify_request_error(exc)
            request_error = failure.message
            parse_error = None
            if failure.retryable and try_index < max_tries:
                sleep(1)
                continue
            break

        try:
            parsed = parse_response(raw, response_schema)
        except ValueError as exc:
            failure = _classify_response_error(exc)
            parse_error = failure.message
            request_error = None
            if try_index < max_tries:
                sleep(1)
            continue

        parse_error = None
        request_error = None
        failure = None
        predicted_label = ModerationLabel.from_value(
            parsed["judge"], "response.judge"
        ).value
        break

    if request_error is not None and not raw:
        raw = "<request_error>"
    elif not raw:
        raw = "<empty_response>"

    return {
        "task_name": task.name,
        "input_text": input_text,
        "obf_density": obf_density,
        "response": {"raw": raw, "parsed": parsed, "parse_error": parse_error},
        "predicted_label": predicted_label,
        "label_match": (
            False
            if predicted_label is None
            else corpus_entry.expected_label.value == predicted_label
        ),
        "try_count": try_count,
        "request_error": request_error,
        "failure_kind": None if failure is None else failure.kind,
        "status_code": None if failure is None else failure.status_code,
        "request_id": None if failure is None else failure.request_id,
    }


def _classify_response_error(exc: Exception) -> _EvaluationFailure:
    if isinstance(exc, json.JSONDecodeError):
        return _EvaluationFailure(
            kind="invalid_json",
            retryable=True,
            message=str(exc),
        )
    return _EvaluationFailure(
        kind="schema_mismatch",
        retryable=True,
        message=str(exc),
    )


def _classify_request_error(exc: Exception) -> _EvaluationFailure:
    status_code = _get_exception_int_attr(exc, "status_code")
    request_id = _get_exception_string_attr(
        exc, "request_id"
    ) or _get_exception_string_attr(exc, "_request_id")
    message = _sanitize_error_message(str(exc))
    class_names = _get_exception_class_names(exc)

    if _looks_like_no_quota(exc, message):
        return _EvaluationFailure("no_quota", False, message, status_code, request_id)
    if "APITimeoutError" in class_names or any(
        "Timeout" in name for name in class_names
    ):
        return _EvaluationFailure("timeout", True, message, status_code, request_id)
    if "AuthenticationError" in class_names or status_code == 401:
        return _EvaluationFailure("auth", False, message, status_code, request_id)
    if "PermissionDeniedError" in class_names or status_code == 403:
        return _EvaluationFailure("permission", False, message, status_code, request_id)
    if "RateLimitError" in class_names or status_code == 429:
        return _EvaluationFailure("rate_limit", True, message, status_code, request_id)
    if "APIConnectionError" in class_names:
        return _EvaluationFailure("connection", True, message, status_code, request_id)
    if status_code is not None:
        if status_code >= 500:
            return _EvaluationFailure("server", True, message, status_code, request_id)
        if status_code in {400, 422}:
            return _EvaluationFailure(
                "bad_request", False, message, status_code, request_id
            )
    return _EvaluationFailure("unknown", True, message, status_code, request_id)


def _get_exception_class_names(exc: Exception) -> set[str]:
    return {cls.__name__ for cls in type(exc).mro()}


def _get_exception_int_attr(exc: Exception, name: str) -> int | None:
    value = getattr(exc, name, None)
    return value if isinstance(value, int) else None


def _get_exception_string_attr(exc: Exception, name: str) -> str | None:
    value = getattr(exc, name, None)
    return value if isinstance(value, str) and value else None


def _looks_like_no_quota(exc: Exception, message: str) -> bool:
    haystack = " ".join(
        part
        for part in [
            message,
            str(getattr(exc, "code", "")),
            str(getattr(exc, "type", "")),
            str(getattr(exc, "body", "")),
        ]
        if part
    ).lower()
    return any(
        marker in haystack
        for marker in (
            "insufficient_quota",
            "exceeded your current quota",
            "check your plan and billing",
            "quota exceeded",
            "billing",
        )
    )


def _sanitize_error_message(message: str) -> str:
    cleaned = " ".join(message.split())
    return cleaned[:500] if cleaned else "<request_error>"


def _render_task_image(
    cache_dir: Path,
    entry: ObfuscatedCorpusEntry,
    task: EvaluationTask,
    input_text: str,
    render_config: TextRenderConfig,
) -> Path:
    from sinoglyph.render import TextRenderer

    image_dir = cache_dir / "images" / build_safe_name(entry.id)
    text_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
    image_path = image_dir / f"{text_hash}.png"
    TextRenderer(input_text, render_config).render(image_path)
    return image_path
