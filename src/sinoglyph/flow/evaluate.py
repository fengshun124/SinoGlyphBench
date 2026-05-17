from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import unicodedata as ud
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from typing import TYPE_CHECKING, Iterator, cast

from tqdm.auto import tqdm

from sinoglyph.io import (
    PathLike,
    load_env_file,
    load_json,
    parse_json_response,
    save_json,
)
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.corpus import PerturbedCorpusEntry, load_perturbed_corpus
from sinoglyph.schema.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationTask,
    validate_response_instance,
)
from sinoglyph.schema.types import InputType, ModerationLabel, TaskSource, TaskVariant
from sinoglyph.schema.utils import require_list, require_mapping, require_string

if TYPE_CHECKING:
    from sinoglyph.llm import LLMClient
    from sinoglyph.render import RenderConfig


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    entry: JsonObject


@dataclass(frozen=True)
class EvaluationJobResult:
    index: int
    entry: JsonObject


@dataclass(frozen=True)
class EvaluationFailure:
    kind: str
    retryable: bool
    message: str
    status_code: int | None = None
    request_id: str | None = None


def run_evaluation(
    config_path: PathLike,
    output_path: PathLike | None = None,
    cache_dir: PathLike | None = None,
    n_jobs: int | None = None,
) -> JsonObject:
    load_env_file()
    config = EvaluationConfig.from_toml(config_path)
    settings = config.evaluation
    fingerprint = config.fingerprint()
    meta = config.result_meta()
    resolved_cache_dir = Path(
        cache_dir if cache_dir is not None else settings.cache_dir
    ).expanduser()
    resolved_n_jobs = n_jobs if n_jobs is not None else settings.n_jobs
    if resolved_n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer")
    _prepare_cache_dir(resolved_cache_dir)
    cache = EvaluationCache(resolved_cache_dir, fingerprint)

    corpus = load_perturbed_corpus(settings.corpus_path)
    limited_corpus = [entry.to_mapping() for entry in corpus[: settings.limit]]

    with cache.run_lock():
        llm_config = config.llm_client_config().to_dict()
        max_retries = cast(int, llm_config.get("max_retries", 0))
        max_tries = max_retries + 1
        llm_config["max_retries"] = 0
        tasks = config.tasks
        needs_render = any(task.input_type == InputType.IMAGE for task in tasks)
        render_mapping = config.render if needs_render else None
        render_runtime_mapping = (
            config.render_runtime_mapping() if needs_render else None
        )
        prompt = config.prompt
        response_schema = config.response_schema

        cached_entries: dict[int, JsonObject] = {}
        jobs: list[EvaluationJob] = []
        resumed_entries = 0
        for index, entry in enumerate(limited_corpus):
            entry_id = require_string(entry["id"], "entry.id")
            cached = cache.load_entry(index, entry_id)
            entry = _entry_with_cached_results(entry, cached, meta, tasks)
            if entry.get("results"):
                resumed_entries += 1
            if _entry_has_all_task_results(entry, tasks):
                cached_entries[index] = entry
            else:
                jobs.append(EvaluationJob(index=index, entry=entry))

        if resumed_entries:
            tqdm.write(
                "Resuming evaluation from cache. Use --cache-dir with a new path or "
                f"remove {resolved_cache_dir} for a fresh start."
            )

        completed_entries = dict(cached_entries)
        if jobs:
            try:
                from joblib import Parallel, delayed
            except ImportError as exc:
                raise ImportError(
                    "joblib is required for evaluation. Install dependencies from "
                    "requirements.txt before running the evaluate command."
                ) from exc

            worker = delayed(_evaluate_corpus_entry)
            results = Parallel(n_jobs=resolved_n_jobs, return_as="generator_unordered")(
                worker(
                    job,
                    tasks,
                    max_tries,
                    llm_config,
                    prompt.text_prompt,
                    prompt.image_prompt,
                    response_schema,
                    render_mapping,
                    render_runtime_mapping,
                    resolved_cache_dir,
                    fingerprint,
                )
                for job in jobs
            )
            for result in tqdm(
                results,
                total=len(jobs),
                desc="Evaluating corpus",
                unit="corpus",
            ):
                completed_entries[result.index] = result.entry

        ordered_corpus = [
            completed_entries[index] for index in range(len(limited_corpus))
        ]
        output = {
            "meta": meta,
            "tasks": [task.to_mapping() for task in tasks],
            "corpus": ordered_corpus,
        }
        EvaluationResult.from_mapping(output)

        resolved_output_path = Path(
            output_path
            if output_path is not None
            else Path(settings.output_dir) / f"{settings.name}.json"
        ).expanduser()
        save_json(output, resolved_output_path)
        return output


def parse_evaluation_response(raw: str, response_schema: JsonObject) -> JsonObject:
    parsed = parse_json_response(raw)
    try:
        return validate_response_instance(parsed, response_schema)
    except ValueError as exc:
        raise ValueError(f"response JSON does not match schema: {exc}") from exc


def apply_evaluation_line_breaks(text: str, render: JsonObject | None) -> str:
    if render is None or not render.get("line_breaks"):
        return text
    max_chars = cast(int, render["line_break_max_chars"])
    return "\n".join(
        _wrap_evaluation_line(line, max_chars) for line in text.split("\n")
    )


def apply_task_line_breaks(
    entry: PerturbedCorpusEntry,
    task: EvaluationTask,
    input_text: str,
    render: JsonObject | None,
) -> str:
    if render is None or not render.get("line_breaks"):
        return input_text
    wrapped_original = apply_evaluation_line_breaks(entry.text, render)
    if input_text == entry.text:
        return wrapped_original
    chunks = _task_input_chunks(entry, task, input_text)
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


def _evaluate_corpus_entry(
    job: EvaluationJob,
    tasks: list[EvaluationTask],
    max_tries: int,
    llm_config: dict[str, object],
    text_prompt: str,
    image_prompt: str,
    response_schema: JsonObject,
    render_mapping: JsonObject | None,
    render_runtime_mapping: JsonObject | None,
    cache_dir: Path,
    fingerprint: str,
) -> EvaluationJobResult:
    from sinoglyph.llm import LLMClient, LLMClientConfig
    from sinoglyph.render import RenderConfig

    cache = EvaluationCache(cache_dir, fingerprint)
    render_config = (
        None
        if render_runtime_mapping is None
        else RenderConfig.from_dict(render_runtime_mapping)
    )
    response_contract = _response_contract(response_schema)
    entry = deepcopy(job.entry)
    results_by_task_name = _results_by_task_name(entry, tasks)
    cache_path = _cache_entry_path(
        cache_dir, job.index, require_string(entry["id"], "entry.id")
    )
    for task in tasks:
        if task.name not in results_by_task_name:
            results_by_task_name[task.name] = _evaluate_task(
                entry,
                task,
                max_tries,
                llm_config,
                _prompt_with_response_contract(text_prompt, response_contract),
                _prompt_with_response_contract(image_prompt, response_contract),
                response_schema,
                render_mapping,
                render_config,
                cache_dir,
            )
            entry["results"] = _ordered_results(results_by_task_name, tasks)
            entry_id = require_string(entry["id"], "entry.id")
            with cache.entry_lock(job.index, entry_id):
                cache.save_entry(entry, job.index, cache_path)
    entry["results"] = _ordered_results(results_by_task_name, tasks)
    entry_id = require_string(entry["id"], "entry.id")
    with cache.entry_lock(job.index, entry_id):
        cache.save_entry(entry, job.index, cache_path)
    return EvaluationJobResult(index=job.index, entry=entry)


def _evaluate_task(
    entry: JsonObject,
    task: EvaluationTask,
    max_tries: int,
    llm_config: JsonObject,
    text_prompt: str,
    image_prompt: str,
    response_schema: JsonObject,
    render_mapping: JsonObject | None,
    render_config: RenderConfig | None,
    cache_dir: Path,
) -> JsonObject:
    from sinoglyph.llm import LLMClient, LLMClientConfig

    client = LLMClient(LLMClientConfig.from_dict(llm_config))
    corpus_entry = PerturbedCorpusEntry.from_mapping(entry)
    raw_input_text = corpus_entry.input_text(task.source.value, task.variant)
    input_text = (
        apply_task_line_breaks(corpus_entry, task, raw_input_text, render_mapping)
        if task.input_type == InputType.IMAGE
        else raw_input_text
    )
    substitution_fraction = corpus_entry.substitution_fraction(
        task.source.value, task.variant
    )
    raw = ""
    parse_error: str | None = None
    request_error: str | None = None
    failure: EvaluationFailure | None = None
    parsed = None
    predicted_label = None
    try_count = 0
    image_path = None
    if task.input_type == InputType.IMAGE:
        if render_config is None:
            raise ValueError("render config is required for image tasks")
        image_path = _render_task_image(
            cache_dir, corpus_entry, task, input_text, render_config
        )

    for try_index in range(1, max_tries + 1):
        try_count = try_index
        client.clear(keep_system=True)
        try:
            if task.input_type == InputType.TEXT:
                raw = cast(
                    str, client.chat(text=_text_message(text_prompt, input_text))
                )
            else:
                if image_path is None:
                    raise AssertionError("image path was not prepared for image task")
                raw = cast(
                    str,
                    client.chat(
                        text=_image_message(image_prompt), images=[str(image_path)]
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
            parsed = parse_evaluation_response(raw, response_schema)
        except ValueError as exc:
            failure = _classify_response_error(exc)
            parse_error = failure.message
            request_error = None
            if try_index < max_tries:
                sleep(1)
            continue

        if parsed is None:
            parse_error = "Response schema validation failed"
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
        "substitution_fraction": substitution_fraction,
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


class EvaluationCache:
    def __init__(self, cache_dir: Path, fingerprint: str) -> None:
        self.cache_dir = cache_dir
        self.fingerprint = fingerprint

    def entry_path(self, index: int, entry_id: str) -> Path:
        return _cache_entry_path(self.cache_dir, index, entry_id)

    def load_entry(self, index: int, entry_id: str) -> JsonObject | None:
        cache_path = self.entry_path(index, entry_id)
        if not cache_path.is_file():
            return None
        try:
            cached = load_json(cache_path)
        except Exception:
            return None
        if not isinstance(cached, dict):
            return None
        if cached.get("fingerprint") != self.fingerprint:
            return None
        if cached.get("index") != index:
            return None
        entry = cached.get("entry")
        if not isinstance(entry, dict) or entry.get("id") != entry_id:
            return None
        return entry

    def save_entry(self, entry: JsonObject, index: int, cache_path: Path) -> None:
        save_json(
            {
                "fingerprint": self.fingerprint,
                "index": index,
                "entry": entry,
            },
            cache_path,
            warn_overwrite=False,
        )

    @contextmanager
    def run_lock(self) -> Iterator[None]:
        with _cache_run_lock(self.cache_dir, self.fingerprint):
            yield

    @contextmanager
    def entry_lock(self, index: int, entry_id: str) -> Iterator[None]:
        """Acquire per-entry lock to prevent concurrent writes to the same entry."""
        with _cache_entry_lock(self.cache_dir, index, entry_id):
            yield


def _entry_with_cached_results(
    entry: JsonObject,
    cached_entry: JsonObject | None,
    meta: JsonObject,
    tasks: list[EvaluationTask],
) -> JsonObject:
    resumed = deepcopy(entry)
    if cached_entry is None:
        return resumed
    cached_results = cached_entry.get("results")
    if not isinstance(cached_results, list):
        return resumed
    tasks_by_name = {task.name: task for task in tasks}
    results_by_task_name: dict[str, JsonObject] = {}
    for result in cached_results:
        if not isinstance(result, dict):
            continue
        task_name = result.get("task_name")
        if not isinstance(task_name, str) or task_name in results_by_task_name:
            continue
        task = tasks_by_name.get(task_name)
        if task is None:
            continue
        candidate = cast(JsonObject, deepcopy(result))
        if _cached_result_is_reusable(candidate) and _cached_result_is_valid(
            resumed,
            task,
            candidate,
            meta,
        ):
            results_by_task_name[task_name] = candidate
    if results_by_task_name:
        resumed["results"] = _ordered_results(results_by_task_name, tasks)
    return resumed


def _cached_result_is_reusable(result: JsonObject) -> bool:
    response = result.get("response")
    return (
        isinstance(response, dict)
        and isinstance(response.get("parsed"), dict)
        and response.get("parse_error") is None
        and result.get("request_error") is None
        and result.get("failure_kind") is None
        and isinstance(result.get("predicted_label"), str)
    )


def _cached_result_is_valid(
    entry: JsonObject,
    task: EvaluationTask,
    result: JsonObject,
    meta: JsonObject,
) -> bool:
    candidate_entry = deepcopy(entry)
    candidate_entry["results"] = [result]
    try:
        EvaluationResult.from_mapping(
            {"meta": meta, "tasks": [task.to_mapping()], "corpus": [candidate_entry]}
        )
    except Exception:
        return False
    return True


def _entry_has_all_task_results(entry: JsonObject, tasks: list[EvaluationTask]) -> bool:
    return set(_results_by_task_name(entry, tasks, reusable_only=True)) == {
        task.name for task in tasks
    }


def _results_by_task_name(
    entry: JsonObject,
    tasks: list[EvaluationTask],
    *,
    reusable_only: bool = False,
) -> dict[str, JsonObject]:
    task_names = {task.name for task in tasks}
    results = entry.get("results")
    if not isinstance(results, list):
        return {}
    results_by_task_name: dict[str, JsonObject] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        task_name = result.get("task_name")
        if not isinstance(task_name, str) or task_name not in task_names:
            continue
        if reusable_only and not _cached_result_is_reusable(cast(JsonObject, result)):
            continue
        results_by_task_name.setdefault(task_name, cast(JsonObject, result))
    return results_by_task_name


def _ordered_results(
    results_by_task_name: dict[str, JsonObject],
    tasks: list[EvaluationTask],
) -> list[JsonObject]:
    return [
        results_by_task_name[task.name]
        for task in tasks
        if task.name in results_by_task_name
    ]


def _text_message(prompt: str, input_text: str) -> str:
    return f"{prompt.rstrip()}\n\n<INPUT_TEXT>\n{input_text}\n</INPUT_TEXT>"


def _image_message(prompt: str) -> str:
    return (
        f"{prompt.rstrip()}\n\n<INPUT_IMAGE>\n"
        "The input is attached as a rendered image.\n"
        "</INPUT_IMAGE>"
    )


def _prompt_with_response_contract(prompt: str, response_contract: str) -> str:
    return f"{prompt.rstrip()}\n\n{response_contract}"


def _response_contract(response_schema: JsonObject) -> str:
    properties = require_mapping(
        response_schema.get("properties", {}), "response.schema.properties"
    )
    required_fields = [
        item
        for item in require_list(
            response_schema.get("required", []), "response.schema.required"
        )
        if isinstance(item, str)
    ]
    lines = [
        "Output contract:",
        "Return exactly one JSON object and no markdown.",
        "The JSON object must match this schema:",
    ]
    for field in required_fields:
        field_schema = properties.get(field, {})
        enum = None
        if isinstance(field_schema, dict) and isinstance(
            field_schema.get("enum"), list
        ):
            enum = ", ".join(str(value) for value in field_schema["enum"])
        enum_text = f" one of: {enum}." if enum else ""
        lines.append(f"- {field}: string.{enum_text}")
    if response_schema.get("additionalProperties") is False:
        lines.append("Do not include any other fields.")
    return "\n".join(lines)


def _classify_response_error(exc: Exception) -> EvaluationFailure:
    if isinstance(exc, json.JSONDecodeError):
        return EvaluationFailure(
            kind="invalid_json",
            retryable=True,
            message=str(exc),
        )
    return EvaluationFailure(
        kind="schema_mismatch",
        retryable=True,
        message=str(exc),
    )


def _classify_request_error(exc: Exception) -> EvaluationFailure:
    status_code = _exception_int_attr(exc, "status_code")
    request_id = _exception_string_attr(exc, "request_id") or _exception_string_attr(
        exc, "_request_id"
    )
    message = _sanitize_error_message(str(exc))
    class_names = _exception_class_names(exc)

    if _looks_like_no_quota(exc, message):
        return EvaluationFailure("no_quota", False, message, status_code, request_id)
    if "APITimeoutError" in class_names or any(
        "Timeout" in name for name in class_names
    ):
        return EvaluationFailure("timeout", True, message, status_code, request_id)
    if "AuthenticationError" in class_names or status_code == 401:
        return EvaluationFailure("auth", False, message, status_code, request_id)
    if "PermissionDeniedError" in class_names or status_code == 403:
        return EvaluationFailure("permission", False, message, status_code, request_id)
    if "RateLimitError" in class_names or status_code == 429:
        return EvaluationFailure("rate_limit", True, message, status_code, request_id)
    if "APIConnectionError" in class_names:
        return EvaluationFailure("connection", True, message, status_code, request_id)
    if status_code is not None:
        if status_code >= 500:
            return EvaluationFailure("server", True, message, status_code, request_id)
        if status_code in {400, 422}:
            return EvaluationFailure(
                "bad_request", False, message, status_code, request_id
            )
    return EvaluationFailure("unknown", True, message, status_code, request_id)


def _exception_class_names(exc: Exception) -> set[str]:
    return {cls.__name__ for cls in type(exc).mro()}


def _exception_int_attr(exc: Exception, name: str) -> int | None:
    value = getattr(exc, name, None)
    return value if isinstance(value, int) else None


def _exception_string_attr(exc: Exception, name: str) -> str | None:
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
    entry: PerturbedCorpusEntry,
    task: EvaluationTask,
    input_text: str,
    render_config: RenderConfig,
) -> Path:
    from sinoglyph.render import TextRenderer

    image_dir = cache_dir / "images" / _safe_name(entry.id)
    # Use hash of input_text for content-based deduplication
    text_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
    image_path = image_dir / f"{text_hash}.png"
    TextRenderer(input_text, render_config).render(image_path)
    return image_path


def _task_input_chunks(
    entry: PerturbedCorpusEntry,
    task: EvaluationTask,
    input_text: str,
) -> list[str] | None:
    replacements = _task_replacements_by_character(entry, task)
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


def _task_replacements_by_character(
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


def _wrap_evaluation_line(text: str, max_chars: int) -> str:
    lines: list[str] = []
    rest = text
    while len(rest) > max_chars:
        break_at = _find_evaluation_break(rest, max_chars)
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


def _find_evaluation_break(text: str, max_chars: int) -> tuple[int, int] | None:
    candidates = _evaluation_break_candidates(text)
    before = [candidate for candidate in candidates if candidate[0] <= max_chars]
    if before:
        return before[-1]
    return next(
        (candidate for candidate in candidates if candidate[0] > max_chars), None
    )


def _evaluation_break_candidates(text: str) -> list[tuple[int, int]]:
    return [
        (index, index + 1) if character.isspace() else (index + 1, index + 1)
        for index, character in enumerate(text)
        if character != "\n"
        and (character.isspace() or ud.category(character).startswith("P"))
    ]


def _cache_entry_path(cache_dir: Path, index: int, entry_id: str) -> Path:
    return cache_dir / "entries" / f"{index:06d}-{_safe_name(entry_id)}.json"


def _prepare_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode = cache_dir.stat().st_mode
    if mode & 0o077:
        os.chmod(cache_dir, mode & ~0o077)


@contextmanager
def _cache_run_lock(cache_dir: Path, fingerprint: str):
    lock_path = cache_dir / ".evaluation.lock"
    lock_payload = json.dumps(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "fingerprint": fingerprint,
            "created_at": time(),
        },
        sort_keys=True,
    )
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            if _cache_lock_is_stale(lock_path):
                lock_path.unlink(missing_ok=True)
                continue
            raise RuntimeError(
                f"Cache directory is already in use: {cache_dir}. "
                "Use a different --cache-dir or wait for the other run to finish."
            ) from exc
        break
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(lock_payload)
            f.write("\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


@contextmanager
def _cache_entry_lock(cache_dir: Path, index: int, entry_id: str):
    lock_filename = f".entry-{index}-{entry_id}.lock"
    lock_path = cache_dir / lock_filename
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Open lock file (create if doesn't exist)
    with open(lock_path, "w") as lock_file:
        try:
            # Acquire exclusive lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cache_lock_is_stale(lock_path: Path) -> bool:
    try:
        raw = load_json(lock_path)
    except Exception:
        return _cache_lock_age_seconds(lock_path) > 60
    if not isinstance(raw, dict):
        return _cache_lock_age_seconds(lock_path) > 60
    pid = raw.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return _cache_lock_age_seconds(lock_path) > 60
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _cache_lock_age_seconds(lock_path: Path) -> float:
    try:
        return time() - lock_path.stat().st_mtime
    except OSError:
        return 0


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    safe = "".join(character if character.isalnum() else "-" for character in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{safe[:80]}-{digest}" if safe else digest


def _safe_task_image_name(task_name: str) -> str:
    digest = hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:16]
    phrases = [_safe_task_image_phrase(phrase) for phrase in task_name.split("/")]
    safe = "-".join(phrase for phrase in phrases if phrase)
    return f"{safe[:80]}-{digest}" if safe else digest


def _safe_task_image_phrase(phrase: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in phrase)
    return "_".join(part for part in safe.split("_") if part)
