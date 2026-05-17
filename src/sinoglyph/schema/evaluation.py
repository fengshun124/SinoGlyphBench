from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sinoglyph.io import PathLike, load_toml
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.corpus import PerturbedCorpusEntry
from sinoglyph.schema.types import InputType, ModerationLabel, TaskSource, TaskVariant
from sinoglyph.schema.utils import (
    optional_string,
    require_boolean,
    require_enum,
    require_keys,
    require_list,
    require_mapping,
    require_non_negative_integer,
    require_number,
    require_positive_integer,
    require_string,
)

LLM_DEFAULTS: JsonObject = {"max_retries": 2, "timeout": 120, "temperature": 0}
EVALUATION_DEFAULTS: JsonObject = {"n_jobs": 5}
RENDER_DEFAULTS: JsonObject = {
    "line_breaks": False,
    "line_break_max_chars": 32,
}
RENDER_REQUIRED = {
    "cjk_font",
    "lgc_font",
    "symbol_font",
    "emoji_font",
    "size_px",
    "pad",
    "dpi",
    "fg_color",
    "bg_color",
    "align",
}
DEFAULT_RESPONSE_SCHEMA: JsonObject = {
    "type": "object",
    "additionalProperties": False,
    "required": ["read_text", "recovered_text", "interpretation", "judge"],
    "properties": {
        "read_text": {"type": "string", "minLength": 1},
        "recovered_text": {"type": "string", "minLength": 1},
        "interpretation": {"type": "string", "minLength": 1},
        "judge": {
            "type": "string",
            "enum": [label.value for label in ModerationLabel],
        },
    },
}


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    text_prompt: str
    image_prompt: str

    @classmethod
    def from_mapping(cls, mapping: object, context: str = "prompt") -> "PromptConfig":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"system_prompt", "text_prompt", "image_prompt"}, context)
        return cls(
            system_prompt=require_string(
                raw["system_prompt"], f"{context}.system_prompt"
            ),
            text_prompt=require_string(raw["text_prompt"], f"{context}.text_prompt"),
            image_prompt=require_string(raw["image_prompt"], f"{context}.image_prompt"),
        )

    def to_mapping(self) -> JsonObject:
        return {
            "system_prompt": self.system_prompt,
            "text_prompt": self.text_prompt,
            "image_prompt": self.image_prompt,
        }


@dataclass(frozen=True)
class EvaluationTask:
    input_type: InputType
    source: TaskSource
    variant: TaskVariant
    name: str

    @classmethod
    def from_mapping(cls, mapping: object, context: str = "task") -> "EvaluationTask":
        raw = require_mapping(mapping, context)
        require_keys(raw, {"input_type", "source", "variant"}, context)
        input_type = require_enum(InputType, raw["input_type"], f"{context}.input_type")
        source = require_enum(TaskSource, raw["source"], f"{context}.source")
        variant = require_enum(TaskVariant, raw["variant"], f"{context}.variant")
        if source == TaskSource.TEXT and variant != TaskVariant.ORIGINAL:
            raise ValueError(f"{context}.source='text' requires variant='original'")
        raw_name = raw.get("name")
        name = (
            require_string(raw_name, f"{context}.name")
            if raw_name is not None
            else "/".join((input_type.value, source.value, variant.value))
        )
        return cls(input_type=input_type, source=source, variant=variant, name=name)

    def to_mapping(self) -> JsonObject:
        return {
            "name": self.name,
            "input_type": self.input_type.value,
            "source": self.source.value,
            "variant": self.variant.value,
        }


@dataclass(frozen=True)
class EvaluationSettings:
    name: str
    corpus_path: str
    output_dir: str
    limit: int
    cache_dir: str
    n_jobs: int

    @classmethod
    def from_mapping(
        cls,
        mapping: object,
        context: str = "evaluation",
    ) -> "EvaluationSettings":
        raw = dict(require_mapping(mapping, context))
        if "attempts" in raw:
            raise ValueError(
                f"{context}.attempts has been removed; use llm.max_retries instead"
            )
        require_keys(raw, {"name", "corpus_path", "output_dir", "limit"}, context)
        name = require_string(raw["name"], f"{context}.name")
        return cls(
            name=name,
            corpus_path=require_string(raw["corpus_path"], f"{context}.corpus_path"),
            output_dir=require_string(raw["output_dir"], f"{context}.output_dir"),
            limit=require_positive_integer(raw["limit"], f"{context}.limit"),
            cache_dir=require_string(
                raw.get("cache_dir", f"cache/{name}"),
                f"{context}.cache_dir",
            ),
            n_jobs=require_positive_integer(
                raw.get("n_jobs", EVALUATION_DEFAULTS["n_jobs"]),
                f"{context}.n_jobs",
            ),
        )

    def to_mapping(self) -> JsonObject:
        return {
            "name": self.name,
            "corpus_path": self.corpus_path,
            "output_dir": self.output_dir,
            "limit": self.limit,
            "cache_dir": self.cache_dir,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation: EvaluationSettings
    llm: JsonObject
    prompt: PromptConfig
    response_schema: JsonObject
    tasks: list[EvaluationTask]
    render: JsonObject | None = None

    @classmethod
    def from_mapping(cls, mapping: object) -> "EvaluationConfig":
        raw = require_mapping(mapping, "config")
        require_keys(raw, {"evaluation", "llm", "prompt", "tasks"}, "config")
        evaluation = EvaluationSettings.from_mapping(raw["evaluation"])
        llm = _normalize_llm(raw["llm"])
        prompt = PromptConfig.from_mapping(raw["prompt"])
        response_schema = _normalize_response_schema(raw.get("response"))
        tasks = _normalize_tasks(raw["tasks"])
        render = None
        if raw.get("render") is not None:
            render = _normalize_render(raw["render"])
        if render is None and any(task.input_type == InputType.IMAGE for task in tasks):
            raise ValueError("render is required when any task has input_type='image'")
        return cls(
            evaluation=evaluation,
            llm=llm,
            prompt=prompt,
            response_schema=response_schema,
            tasks=tasks,
            render=render,
        )

    @classmethod
    def from_toml(cls, file_path: PathLike) -> "EvaluationConfig":
        return cls.from_mapping(load_toml(file_path))

    def to_mapping(self) -> JsonObject:
        output: JsonObject = {
            "evaluation": self.evaluation.to_mapping(),
            "llm": dict(self.llm),
            "prompt": self.prompt.to_mapping(),
            "response": {"schema": dict(self.response_schema)},
            "tasks": [task.to_mapping() for task in self.tasks],
        }
        if self.render is not None:
            output["render"] = dict(self.render)
        return output

    def fingerprint(self) -> str:
        render = None
        if self.render is not None:
            render = {
                **{
                    key: value
                    for key, value in self.render.items()
                    if key not in {"cjk_font", "lgc_font", "symbol_font", "emoji_font"}
                },
                "font_sha256": {
                    key.removesuffix("_font"): _file_sha256(str(self.render[key]))
                    for key in ("cjk_font", "lgc_font", "symbol_font", "emoji_font")
                },
            }
        payload: JsonObject = {
            "corpus": {"sha256": _file_sha256(self.evaluation.corpus_path)},
            "limit": self.evaluation.limit,
            "llm": self.resolved_llm_public_fingerprint_config(),
            "prompt": self.prompt.to_mapping(),
            "response": {"schema": self.response_schema},
            "tasks": [task.to_mapping() for task in self.tasks],
        }
        if render is not None:
            payload["render"] = render
        text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def result_meta(self) -> JsonObject:
        meta: JsonObject = {
            "fingerprint": self.fingerprint(),
            "name": self.evaluation.name,
            "corpus_path": self.evaluation.corpus_path,
            "llm": self.resolved_llm_public_config(),
            "prompt": self.prompt.to_mapping(),
            "response": {"schema": self.response_schema},
        }
        if self.render is not None:
            meta["render"] = dict(self.render)
        return meta

    def resolved_llm_public_config(self) -> JsonObject:
        llm = dict(self.llm)
        _resolve_env_value(llm, "base_url", "base_url_env")
        _resolve_env_value(llm, "model", "model_env")
        return {
            key: value
            for key, value in llm.items()
            if key not in {"api_key", "api_key_env", "base_url_env", "model_env"}
        }

    def resolved_llm_public_fingerprint_config(self) -> JsonObject:
        llm = self.resolved_llm_public_config()
        llm.pop("timeout", None)
        return llm

    def resolved_llm_client_config(self) -> JsonObject:
        llm = dict(self.llm)
        _resolve_env_value(llm, "base_url", "base_url_env")
        _resolve_env_value(llm, "model", "model_env")
        _resolve_env_value(llm, "api_key", "api_key_env")
        return llm

    def llm_client_config(self, api_key: str | None = None) -> Any:
        from sinoglyph.llm import LLMClientConfig

        llm = self.resolved_llm_client_config()
        resolved_api_key = (
            api_key
            if api_key is not None
            else require_string(
                llm["api_key"],
                "llm.api_key",
            )
        )
        options = {
            key: value
            for key, value in llm.items()
            if key not in {"api_key", "api_key_env", "base_url_env", "model_env"}
        }
        return LLMClientConfig(
            api_key=resolved_api_key,
            system=self.prompt.system_prompt,
            **options,
        )

    def render_runtime_mapping(self) -> JsonObject | None:
        if self.render is None:
            return None
        return {
            key: value
            for key, value in self.render.items()
            if key not in RENDER_DEFAULTS
        }


@dataclass(frozen=True)
class EvaluationResult:
    meta: JsonObject
    tasks: list[EvaluationTask]
    corpus: list[JsonObject]

    @classmethod
    def from_mapping(cls, mapping: object) -> "EvaluationResult":
        raw = require_mapping(mapping, "result")
        require_keys(raw, {"meta", "tasks", "corpus"}, "result")
        tasks = _normalize_tasks(raw["tasks"])
        task_by_name = {task.name: task for task in tasks}
        meta = _validate_meta(raw["meta"], tasks)
        response_schema = require_mapping(
            require_mapping(meta["response"], "meta.response")["schema"],
            "meta.response.schema",
        )
        corpus = [
            _validate_result_entry(
                entry,
                index,
                task_by_name,
                response_schema,
                require_mapping(meta.get("render"), "meta.render")
                if "render" in meta
                else None,
            )
            for index, entry in enumerate(require_list(raw["corpus"], "corpus"))
        ]
        return cls(meta=meta, tasks=tasks, corpus=corpus)

    def to_mapping(self) -> JsonObject:
        return {
            "meta": dict(self.meta),
            "tasks": [task.to_mapping() for task in self.tasks],
            "corpus": [dict(entry) for entry in self.corpus],
        }


def default_response_schema() -> JsonObject:
    return json.loads(json.dumps(DEFAULT_RESPONSE_SCHEMA))


def validate_response_instance(
    instance: object,
    schema: JsonObject,
    context: str = "response",
) -> JsonObject:
    _validate_response_schema(schema)
    try:
        from jsonschema import validate as validate_jsonschema

        validate_jsonschema(instance=instance, schema=schema)
    except Exception as exc:
        raise ValueError(f"{context} does not match response schema: {exc}") from exc
    return require_mapping(instance, context)


def _normalize_tasks(raw_tasks: object) -> list[EvaluationTask]:
    tasks = [
        EvaluationTask.from_mapping(raw_task, f"tasks[{index}]")
        for index, raw_task in enumerate(require_list(raw_tasks, "tasks"))
    ]
    if not tasks:
        raise ValueError("tasks expects at least one task")
    names = [task.name for task in tasks]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"tasks contains duplicate task name(s): {', '.join(duplicates)}"
        )
    return tasks


def _normalize_llm(raw_llm: object, *, public: bool = False) -> JsonObject:
    llm = dict(require_mapping(raw_llm, "llm"))
    if "retries" in llm:
        raise ValueError("llm.retries has been removed; use llm.max_retries instead")
    require_keys(
        llm,
        {"base_url", "max_tokens", "model"} if public else {"max_tokens"},
        "llm",
    )
    if public:
        llm = {
            key: value
            for key, value in llm.items()
            if key not in {"api_key", "api_key_env", "base_url_env", "model_env"}
        }
        require_string(llm["base_url"], "llm.base_url")
        require_string(llm["model"], "llm.model")
    else:
        _require_exactly_one_string(llm, "api_key", "api_key_env", "llm")
        _require_exactly_one_string(llm, "base_url", "base_url_env", "llm")
        _require_exactly_one_string(llm, "model", "model_env", "llm")
    llm.update({key: llm.get(key, value) for key, value in LLM_DEFAULTS.items()})
    llm["max_tokens"] = require_positive_integer(llm["max_tokens"], "llm.max_tokens")
    llm["max_retries"] = require_non_negative_integer(
        llm["max_retries"], "llm.max_retries"
    )
    timeout = require_number(llm["timeout"], "llm.timeout")
    if timeout <= 0:
        raise ValueError("llm.timeout expects a positive number")
    llm["timeout"] = (
        int(timeout) if isinstance(timeout, float) and timeout.is_integer() else timeout
    )
    temperature = require_number(llm["temperature"], "llm.temperature")
    if temperature < 0 or temperature > 2:
        raise ValueError("llm.temperature expects a number between 0 and 2")
    llm["temperature"] = (
        int(temperature)
        if isinstance(temperature, float) and temperature.is_integer()
        else temperature
    )
    return llm


def _normalize_response_schema(raw_response: object) -> JsonObject:
    if raw_response is None:
        schema = default_response_schema()
    else:
        response = require_mapping(raw_response, "response")
        require_keys(response, {"schema"}, "response")
        schema = dict(require_mapping(response["schema"], "response.schema"))
    _validate_response_schema(schema)
    return schema


def _validate_response_schema(schema: JsonObject) -> None:
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ValueError(f"response.schema is not a valid JSON Schema: {exc}") from exc


def _normalize_render(raw_render: object) -> JsonObject:
    render = dict(require_mapping(raw_render, "render"))
    require_keys(render, RENDER_REQUIRED, "render")
    render.update(
        {key: render.get(key, value) for key, value in RENDER_DEFAULTS.items()}
    )
    if require_string(render["align"], "render.align") not in {
        "left",
        "center",
        "right",
    }:
        raise ValueError("render.align must be one of: left, center, right")
    render["size_px"] = require_positive_integer(render["size_px"], "render.size_px")
    render["pad"] = require_non_negative_integer(render["pad"], "render.pad")
    render["dpi"] = require_positive_integer(render["dpi"], "render.dpi")
    for key in (
        "cjk_font",
        "lgc_font",
        "symbol_font",
        "emoji_font",
        "fg_color",
        "bg_color",
    ):
        require_string(render[key], f"render.{key}")
    require_boolean(render["line_breaks"], "render.line_breaks")
    render["line_break_max_chars"] = require_positive_integer(
        render["line_break_max_chars"], "render.line_break_max_chars"
    )
    return render


def _validate_meta(raw_meta: object, tasks: list[EvaluationTask]) -> JsonObject:
    meta = dict(require_mapping(raw_meta, "meta"))
    require_keys(
        meta,
        {"fingerprint", "name", "corpus_path", "llm", "prompt", "response"},
        "meta",
    )
    require_string(meta["fingerprint"], "meta.fingerprint")
    require_string(meta["name"], "meta.name")
    require_string(meta["corpus_path"], "meta.corpus_path")
    _normalize_llm(meta["llm"], public=True)
    PromptConfig.from_mapping(meta["prompt"], "meta.prompt")
    _normalize_response_schema(meta["response"])
    if "render" in meta:
        _normalize_render(meta["render"])
    elif any(task.input_type == InputType.IMAGE for task in tasks):
        raise ValueError("meta.render is required when any task has input_type='image'")
    return meta


def _validate_result_entry(
    raw_entry: object,
    entry_index: int,
    task_by_name: dict[str, EvaluationTask],
    response_schema: JsonObject,
    render: JsonObject | None,
) -> JsonObject:
    context = f"corpus[{entry_index}]"
    entry = dict(require_mapping(raw_entry, context))
    require_keys(entry, {"id", "text", "expected_label", "results"}, context)
    model = PerturbedCorpusEntry.from_mapping(entry, context)
    results = require_list(entry["results"], f"{context}.results")
    result_task_names: set[str] = set()
    for result_index, raw_result in enumerate(results):
        result_context = f"{context}.results[{result_index}]"
        result = require_mapping(raw_result, result_context)
        _validate_task_result(
            result, result_context, model, task_by_name, response_schema, render
        )
        result_task_names.add(
            require_string(result["task_name"], f"{result_context}.task_name")
        )
    missing = sorted(set(task_by_name).difference(result_task_names))
    if missing:
        raise ValueError(
            f"{context}.results is missing task result(s): {', '.join(missing)}"
        )
    return entry


def _validate_task_result(
    result: JsonObject,
    context: str,
    entry: PerturbedCorpusEntry,
    task_by_name: dict[str, EvaluationTask],
    response_schema: JsonObject,
    render: JsonObject | None,
) -> None:
    require_keys(
        result,
        {
            "task_name",
            "input_text",
            "substitution_fraction",
            "response",
            "predicted_label",
            "label_match",
            "try_count",
            "request_error",
            "failure_kind",
        },
        context,
    )
    task_name = require_string(result["task_name"], f"{context}.task_name")
    if task_name not in task_by_name:
        raise ValueError(f"{context}.task_name references unknown task {task_name!r}")
    task = task_by_name[task_name]
    expected_input = entry.input_text(task.source.value, task.variant)
    if task.input_type == InputType.IMAGE:
        from sinoglyph.flow.evaluate import apply_task_line_breaks

        expected_input = apply_task_line_breaks(entry, task, expected_input, render)
    if result["input_text"] != expected_input:
        raise ValueError(f"{context}.input_text does not match task input")
    expected_fraction = entry.substitution_fraction(task.source.value, task.variant)
    if result["substitution_fraction"] != expected_fraction:
        raise ValueError(f"{context}.substitution_fraction does not match task input")
    predicted_label = None
    if result["predicted_label"] is not None:
        predicted_label = require_enum(
            ModerationLabel,
            result["predicted_label"],
            f"{context}.predicted_label",
        )
    if not isinstance(result["label_match"], bool):
        raise ValueError(f"{context}.label_match expects a boolean")
    if result["label_match"] != (
        False if predicted_label is None else predicted_label == entry.expected_label
    ):
        raise ValueError(f"{context}.label_match does not match labels")
    require_positive_integer(result["try_count"], f"{context}.try_count")
    optional_string(result["request_error"], f"{context}.request_error")
    optional_string(result["failure_kind"], f"{context}.failure_kind")
    if "status_code" in result and result["status_code"] is not None:
        require_positive_integer(result["status_code"], f"{context}.status_code")
    if "request_id" in result:
        optional_string(result["request_id"], f"{context}.request_id")
    response = require_mapping(result["response"], f"{context}.response")
    require_keys(response, {"raw", "parsed", "parse_error"}, f"{context}.response")
    require_string(response["raw"], f"{context}.response.raw")
    optional_string(response["parse_error"], f"{context}.response.parse_error")
    if response["parsed"] is None:
        if predicted_label is not None:
            raise ValueError(
                f"{context}.predicted_label must be null when response.parsed is null"
            )
        return
    parsed = validate_response_instance(
        response["parsed"], response_schema, f"{context}.response.parsed"
    )
    judge = require_enum(
        ModerationLabel, parsed.get("judge"), f"{context}.response.parsed.judge"
    )
    if predicted_label != judge:
        raise ValueError(f"{context}.response.parsed.judge must match predicted_label")


def _require_exactly_one_string(
    mapping: JsonObject,
    direct_key: str,
    env_key: str,
    context: str,
) -> None:
    has_direct = direct_key in mapping
    has_env = env_key in mapping
    if has_direct == has_env:
        raise ValueError(
            f"{context} requires exactly one of {direct_key!r} or {env_key!r}"
        )
    key = direct_key if has_direct else env_key
    require_string(mapping[key], f"{context}.{key}")


def _resolve_env_value(mapping: JsonObject, key: str, env_key: str) -> None:
    env_name = mapping.get(env_key)
    if env_name is None:
        return
    resolved = os.getenv(require_string(env_name, env_key), "")
    if not resolved:
        raise ValueError(f"environment variable {env_name!r} is empty")
    mapping[key] = resolved


def _file_sha256(file_path: str) -> str:
    path = Path(file_path).expanduser()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
