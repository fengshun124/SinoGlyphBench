import hashlib
import json
import os
import unicodedata as ud
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias, cast

from sinoglyph.schema.base import JsonMapping, JsonObject, TomlMapping
from sinoglyph.schema.corpus import PerturbedCorpus

INPUT_TYPES = ("text", "image")
SOURCES = ("text", "decomposition", "perturbation")
VARIANTS = ("original", "anchor_only", "non_anchor_only", "full")
JUDGES = ("hostile", "abusive", "benign", "context_dependent")
CORPUS_LABELS = JUDGES

LLM_DEFAULTS: dict[str, object] = {
    "retries": 3,
    "timeout": 120,
    "temperature": 0,
}
EVALUATION_DEFAULTS: dict[str, object] = {
    "n_jobs": 4,
}
RENDER_DEFAULTS: dict[str, object] = {
    "line_breaks": False,
    "line_break_max_chars": 32,
}
DEFAULT_RESPONSE_SCHEMA: JsonObject = {
    "type": "object",
    "additionalProperties": False,
    "required": ["read_text", "recovered_text", "interpretation", "judge"],
    "properties": {
        "read_text": {
            "type": "string",
            "minLength": 1,
        },
        "recovered_text": {
            "type": "string",
            "minLength": 1,
        },
        "interpretation": {
            "type": "string",
            "minLength": 1,
        },
        "judge": {
            "type": "string",
            "enum": list(JUDGES),
        },
    },
}

TASK_ALIASES = {"input", "stage", "field", "kind"}
LLM_ALIASES = {"api_key_source", "timeout_seconds", "max_output_tokens"}
PROMPT_ALIASES = {"system", "text_user", "image_user"}
RESULT_ALIASES = {
    "task",
    "input",
    "text",
    "perturbation_frac",
    "perturbation_fraction",
    "parsed_label",
}

EVALUATION_REQUIRED = {"name", "corpus_path", "output_dir", "attempts", "limit"}
LLM_REQUIRED = {"max_tokens"}
PROMPT_REQUIRED = {"system_prompt", "text_prompt", "image_prompt"}
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
TASK_REQUIRED = {"input_type", "source", "variant"}
META_REQUIRED = {
    "fingerprint",
    "name",
    "corpus_path",
    "attempts",
    "llm",
    "prompt",
    "response",
}
RESULT_REQUIRED = {
    "task_name",
    "input_text",
    "substitution_fraction",
    "response",
    "predicted_label",
    "label_match",
    "attempt_count",
    "request_error",
}
PARSED_REQUIRED = {"read_text", "recovered_text", "interpretation", "judge"}

TaskConfig: TypeAlias = dict[str, object]


def _require_mapping(value: object, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{context} expects a mapping")
    return value


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} expects a list")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} expects a non-empty string")
    return value


def _require_choice(value: object, choices: tuple[str, ...], context: str) -> str:
    value = _require_string(value, context)
    if value not in choices:
        joined = ", ".join(choices)
        raise ValueError(f"{context} must be one of: {joined}")
    return value


def _require_positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} expects a positive integer")
    if value <= 0:
        raise ValueError(f"{context} expects a positive integer")
    return value


def _require_non_negative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} expects a non-negative integer")
    if value < 0:
        raise ValueError(f"{context} expects a non-negative integer")
    return value


def _require_number(value: object, context: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} expects a number")
    return value


def _require_temperature(value: object, context: str) -> float | int:
    temperature = _require_number(value, context)
    if temperature < 0 or temperature > 2:
        raise ValueError(f"{context} expects a number between 0 and 2")
    return temperature


def _require_boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} expects a boolean")
    return value


def _reject_keys(mapping: JsonObject, aliases: set[str], context: str) -> None:
    present = sorted(aliases.intersection(mapping))
    if present:
        names = ", ".join(present)
        raise ValueError(f"{context} uses unsupported alias field(s): {names}")


def _require_keys(mapping: JsonObject, keys: set[str], context: str) -> None:
    missing = sorted(keys.difference(mapping))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{context} is missing required field(s): {names}")


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, context)


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
    _require_string(mapping[key], f"{context}.{key}")


def _resolve_llm_env_value(
    llm: JsonObject,
    key: str,
    env_key: str,
    context: str,
) -> None:
    env_name = llm.get(env_key)
    if env_name is None:
        return
    resolved = os.getenv(cast(str, env_name), "")
    if not resolved:
        raise ValueError(f"environment variable {env_name!r} is empty")
    llm[key] = resolved


def _file_sha256(file_path: object, context: str) -> str:
    path = Path(_require_string(file_path, context)).expanduser()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_response_schema() -> JsonObject:
    return deepcopy(DEFAULT_RESPONSE_SCHEMA)


def _normalize_response(raw_response: object) -> JsonObject:
    if raw_response is None:
        schema = default_response_schema()
    else:
        response = deepcopy(_require_mapping(raw_response, "response"))
        _require_keys(response, {"schema"}, "response")
        schema = deepcopy(_require_mapping(response["schema"], "response.schema"))

    _validate_response_schema(schema)
    return {"schema": schema}


def _validate_response_schema(schema: JsonObject) -> None:
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ValueError(f"response.schema is not a valid JSON Schema: {exc}") from exc

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
    if not isinstance(instance, dict):
        raise ValueError(f"{context} expects a JSON object")
    return cast(JsonObject, instance)


def _normalize_render(raw_render: object) -> JsonObject:
    render = deepcopy(_require_mapping(raw_render, "render"))
    _require_keys(render, RENDER_REQUIRED, "render")
    render.update(
        {
            key: render.get(key, value)
            for key, value in RENDER_DEFAULTS.items()
        }
    )
    _require_choice(render["align"], ("left", "center", "right"), "render.align")
    _require_positive_integer(render["size_px"], "render.size_px")
    _require_non_negative_integer(render["pad"], "render.pad")
    _require_positive_integer(render["dpi"], "render.dpi")
    _require_string(render["cjk_font"], "render.cjk_font")
    _require_string(render["lgc_font"], "render.lgc_font")
    _require_string(render["symbol_font"], "render.symbol_font")
    _require_string(render["emoji_font"], "render.emoji_font")
    _require_string(render["fg_color"], "render.fg_color")
    _require_string(render["bg_color"], "render.bg_color")
    _require_boolean(render["line_breaks"], "render.line_breaks")
    _require_positive_integer(
        render["line_break_max_chars"],
        "render.line_break_max_chars",
    )
    return render


def render_config_mapping(render: JsonObject) -> JsonObject:
    return {
        key: deepcopy(value)
        for key, value in render.items()
        if key not in RENDER_DEFAULTS
    }


def apply_evaluation_line_breaks(text: str, render: JsonObject | None) -> str:
    if render is None or not render.get("line_breaks"):
        return text
    max_chars = cast(int, render["line_break_max_chars"])
    return "\n".join(
        _wrap_evaluation_line(line, max_chars)
        for line in text.split("\n")
    )


def apply_task_evaluation_line_breaks(
    entry: JsonObject,
    task: TaskConfig,
    input_text: str,
    render: JsonObject | None,
) -> str:
    if render is None or not render.get("line_breaks"):
        return input_text
    original_text = cast(str, entry["text"])
    wrapped_original = apply_evaluation_line_breaks(original_text, render)
    if input_text == original_text:
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


def _task_input_chunks(
    entry: JsonObject,
    task: TaskConfig,
    input_text: str,
) -> list[str] | None:
    original_text = cast(str, entry["text"])
    replacements = _task_replacements_by_character(entry, task)
    chunks: list[str] = []
    cursor = 0
    for character in original_text:
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
    entry: JsonObject,
    task: TaskConfig,
) -> dict[str, str]:
    source = cast(str, task["source"])
    variant = cast(str, task["variant"])
    if source not in {"decomposition", "perturbation"} or variant == "original":
        return {}
    match variant:
        case "anchor_only":
            groups = ("anchor",)
        case "non_anchor_only":
            groups = ("non_anchor",)
        case "full":
            groups = ("anchor", "non_anchor")
        case _:
            raise ValueError(f"Unsupported task variant: {variant}")
    field = "perturbation" if source == "perturbation" else "decomposition"
    substitutions = _require_mapping(entry["substitutions"], "entry.substitutions")
    replacements: dict[str, str] = {}
    for group in groups:
        for raw_item in _require_list(
            substitutions[group],
            f"entry.substitutions.{group}",
        ):
            item = _require_mapping(raw_item, f"entry.substitutions.{group}[]")
            character = _require_string(
                item["character"],
                f"entry.substitutions.{group}.character",
            )
            replacements[character] = _require_string(
                item[field],
                f"entry.substitutions.{group}.{field}",
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


def _find_evaluation_break(
    text: str,
    max_chars: int,
) -> tuple[int, int] | None:
    candidates = _evaluation_break_candidates(text)
    before = [
        candidate
        for candidate in candidates
        if candidate[0] <= max_chars
    ]
    if before:
        return before[-1]
    return next(
        (
            candidate
            for candidate in candidates
            if candidate[0] > max_chars
        ),
        None,
    )


def _evaluation_break_candidates(text: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if character == "\n":
            continue
        if character.isspace():
            candidates.append((index, index + 1))
        elif ud.category(character).startswith("P"):
            candidates.append((index + 1, index + 1))
    return candidates


def _normalize_tasks(raw_tasks: object, context: str = "tasks") -> list[TaskConfig]:
    raw_task_list = _require_list(raw_tasks, context)
    if not raw_task_list:
        raise ValueError(f"{context} expects at least one task")

    tasks: list[TaskConfig] = []
    names: set[str] = set()
    for index, raw_task in enumerate(raw_task_list):
        task_context = f"{context}[{index}]"
        task = deepcopy(_require_mapping(raw_task, task_context))
        _reject_keys(task, TASK_ALIASES, task_context)
        _require_keys(task, TASK_REQUIRED, task_context)

        input_type = _require_choice(
            task["input_type"], INPUT_TYPES, f"{task_context}.input_type"
        )
        source = _require_choice(task["source"], SOURCES, f"{task_context}.source")
        variant = _require_choice(task["variant"], VARIANTS, f"{task_context}.variant")
        if source == "text" and variant != "original":
            raise ValueError(
                f"{task_context}.source='text' requires variant='original'"
            )

        name = task.get("name")
        task["name"] = (
            _require_string(name, f"{task_context}.name")
            if name is not None
            else "/".join((input_type, source, variant))
        )
        if task["name"] in names:
            raise ValueError(f"{task_context}.name duplicates {task['name']!r}")
        names.add(cast(str, task["name"]))
        tasks.append(task)
    return tasks


class EvaluationConfig(TomlMapping):
    def __init__(self, mapping: JsonObject) -> None:
        raw = _require_mapping(mapping, "config")
        _reject_keys(raw, {"task"}, "config")
        _require_keys(raw, {"evaluation", "llm", "prompt", "tasks"}, "config")

        evaluation = deepcopy(_require_mapping(raw["evaluation"], "evaluation"))
        _reject_keys(evaluation, {"corpus"}, "evaluation")
        _require_keys(evaluation, EVALUATION_REQUIRED, "evaluation")
        name = _require_string(evaluation["name"], "evaluation.name")
        _require_string(evaluation["corpus_path"], "evaluation.corpus_path")
        _require_string(evaluation["output_dir"], "evaluation.output_dir")
        _require_positive_integer(evaluation["attempts"], "evaluation.attempts")
        _require_positive_integer(evaluation["limit"], "evaluation.limit")
        evaluation.update(
            {
                key: evaluation.get(key, value)
                for key, value in EVALUATION_DEFAULTS.items()
            }
        )
        evaluation["cache_dir"] = evaluation.get("cache_dir", f"cache/{name}")
        _require_string(evaluation["cache_dir"], "evaluation.cache_dir")
        _require_positive_integer(evaluation["n_jobs"], "evaluation.n_jobs")

        llm = deepcopy(_require_mapping(raw["llm"], "llm"))
        _reject_keys(llm, LLM_ALIASES, "llm")
        _require_keys(llm, LLM_REQUIRED, "llm")
        _require_exactly_one_string(llm, "api_key", "api_key_env", "llm")
        _require_exactly_one_string(llm, "base_url", "base_url_env", "llm")
        _require_exactly_one_string(llm, "model", "model_env", "llm")
        llm.update({key: llm.get(key, value) for key, value in LLM_DEFAULTS.items()})
        _require_positive_integer(llm["max_tokens"], "llm.max_tokens")
        _require_non_negative_integer(llm["retries"], "llm.retries")
        timeout = _require_number(llm["timeout"], "llm.timeout")
        if timeout <= 0:
            raise ValueError("llm.timeout expects a positive number")
        _require_temperature(llm["temperature"], "llm.temperature")

        prompt = deepcopy(_require_mapping(raw["prompt"], "prompt"))
        _reject_keys(prompt, PROMPT_ALIASES, "prompt")
        _require_keys(prompt, PROMPT_REQUIRED, "prompt")
        _require_string(prompt["system_prompt"], "prompt.system_prompt")
        _require_string(prompt["text_prompt"], "prompt.text_prompt")
        _require_string(prompt["image_prompt"], "prompt.image_prompt")

        response = _normalize_response(raw.get("response"))
        tasks = _normalize_tasks(raw["tasks"])
        needs_render = any(task["input_type"] == "image" for task in tasks)
        raw_render = raw.get("render")
        if raw_render is None and needs_render:
            raise ValueError("render is required when any task has input_type='image'")

        render = None
        if raw_render is not None:
            render = _normalize_render(raw_render)

        normalized: JsonObject = {
            "evaluation": evaluation,
            "llm": llm,
            "prompt": prompt,
            "response": response,
            "tasks": tasks,
        }
        if render is not None:
            normalized["render"] = render
        super().__init__(normalized)

    def _validate(self, mapping: JsonObject) -> None:
        _require_mapping(mapping.get("evaluation"), "evaluation")
        _require_mapping(mapping.get("llm"), "llm")
        _require_mapping(mapping.get("prompt"), "prompt")
        _require_mapping(mapping.get("response"), "response")
        _require_list(mapping.get("tasks"), "tasks")

    @property
    def evaluation(self) -> JsonObject:
        return deepcopy(cast(JsonObject, self._mapping["evaluation"]))

    @property
    def llm(self) -> JsonObject:
        return deepcopy(cast(JsonObject, self._mapping["llm"]))

    @property
    def prompt(self) -> JsonObject:
        return deepcopy(cast(JsonObject, self._mapping["prompt"]))

    @property
    def response(self) -> JsonObject:
        return deepcopy(cast(JsonObject, self._mapping["response"]))

    @property
    def response_schema(self) -> JsonObject:
        response = cast(JsonObject, self._mapping["response"])
        return deepcopy(cast(JsonObject, response["schema"]))

    @property
    def render(self) -> JsonObject | None:
        value = self._mapping.get("render")
        return None if value is None else deepcopy(cast(JsonObject, value))

    @property
    def tasks(self) -> list[TaskConfig]:
        return deepcopy(cast(list[TaskConfig], self._mapping["tasks"]))

    def fingerprint(self) -> str:
        evaluation = cast(JsonObject, self._mapping["evaluation"])
        payload: JsonObject = {
            "corpus_sha256": _file_sha256(
                evaluation["corpus_path"],
                "evaluation.corpus_path",
            ),
            "attempts": evaluation["attempts"],
            "limit": evaluation["limit"],
            "llm": self.resolved_llm_public_config(),
            "prompt": cast(JsonObject, self._mapping["prompt"]),
            "response": cast(JsonObject, self._mapping["response"]),
            "tasks": cast(list[TaskConfig], self._mapping["tasks"]),
        }
        if "render" in self._mapping:
            render = deepcopy(cast(JsonObject, self._mapping["render"]))
            for key in ("cjk_font", "lgc_font", "symbol_font", "emoji_font"):
                render[f"{key}_sha256"] = _file_sha256(render.pop(key), f"render.{key}")
            payload["render"] = render
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def result_meta(self) -> JsonObject:
        evaluation = cast(JsonObject, self._mapping["evaluation"])
        meta = {
            "fingerprint": self.fingerprint(),
            "name": evaluation["name"],
            "corpus_path": evaluation["corpus_path"],
            "attempts": evaluation["attempts"],
            "llm": self.resolved_llm_public_config(),
            "prompt": self.prompt,
            "response": self.response,
        }
        if "render" in self._mapping:
            meta["render"] = self.render
        return meta

    def llm_client_config(self, api_key: str | None = None) -> Any:
        from sinoglyph.llm import LLMClientConfig

        llm = self.resolved_llm_client_config()
        prompt = self.prompt
        resolved_api_key = api_key if api_key is not None else cast(str, llm["api_key"])
        if not resolved_api_key:
            raise ValueError("llm.api_key expects a non-empty string")

        options = {
            key: value
            for key, value in llm.items()
            if key not in {"api_key", "api_key_env", "base_url_env", "model_env"}
        }
        return LLMClientConfig(
            api_key=resolved_api_key,
            system=cast(str, prompt["system_prompt"]),
            **options,
        )

    def resolved_llm_public_config(self) -> JsonObject:
        llm = self.llm
        _resolve_llm_env_value(llm, "base_url", "base_url_env", "llm.base_url_env")
        _resolve_llm_env_value(llm, "model", "model_env", "llm.model_env")
        if "api_key" in llm:
            llm["api_key"] = "<redacted>"
        return llm

    def resolved_llm_client_config(self) -> JsonObject:
        llm = self.llm
        _resolve_llm_env_value(llm, "base_url", "base_url_env", "llm.base_url_env")
        _resolve_llm_env_value(llm, "model", "model_env", "llm.model_env")
        _resolve_llm_env_value(llm, "api_key", "api_key_env", "llm.api_key_env")
        return llm

    def render_config(self) -> Any | None:
        from sinoglyph.render import RenderConfig

        render = self.render
        if render is None:
            return None
        return RenderConfig.from_dict(render_config_mapping(render))


class EvaluationResult(JsonMapping):
    def _validate(self, mapping: JsonObject) -> None:
        _reject_keys(mapping, {"task"}, "result")
        _require_keys(mapping, {"meta", "tasks", "corpus"}, "result")

        tasks_by_name = {
            cast(str, task["name"]): task for task in _normalize_tasks(mapping["tasks"])
        }

        meta = _require_mapping(mapping["meta"], "meta")
        _require_keys(meta, META_REQUIRED, "meta")
        _require_string(meta["fingerprint"], "meta.fingerprint")
        _require_string(meta["name"], "meta.name")
        _require_string(meta["corpus_path"], "meta.corpus_path")
        _require_positive_integer(meta["attempts"], "meta.attempts")

        llm = _require_mapping(meta["llm"], "meta.llm")
        _reject_keys(llm, LLM_ALIASES, "meta.llm")
        _require_keys(llm, LLM_REQUIRED.union(LLM_DEFAULTS), "meta.llm")
        _require_string(llm["base_url"], "meta.llm.base_url")
        _require_string(llm["model"], "meta.llm.model")
        _require_exactly_one_string(llm, "api_key", "api_key_env", "meta.llm")
        _require_number(llm["timeout"], "meta.llm.timeout")
        _require_non_negative_integer(llm["retries"], "meta.llm.retries")
        _require_positive_integer(llm["max_tokens"], "meta.llm.max_tokens")
        _require_temperature(llm["temperature"], "meta.llm.temperature")

        prompt = _require_mapping(meta["prompt"], "meta.prompt")
        _reject_keys(prompt, PROMPT_ALIASES, "meta.prompt")
        _require_keys(prompt, PROMPT_REQUIRED, "meta.prompt")
        _require_string(prompt["system_prompt"], "meta.prompt.system_prompt")
        _require_string(prompt["text_prompt"], "meta.prompt.text_prompt")
        _require_string(prompt["image_prompt"], "meta.prompt.image_prompt")

        response = _normalize_response(meta["response"])
        response_schema = cast(JsonObject, response["schema"])
        needs_render = any(task["input_type"] == "image" for task in tasks_by_name.values())
        render = None
        if "render" in meta:
            render = _normalize_render(meta["render"])
        elif needs_render:
            raise ValueError("meta.render is required when any task has input_type='image'")

        for entry_index, raw_entry in enumerate(
            _require_list(mapping["corpus"], "corpus")
        ):
            entry_context = f"corpus[{entry_index}]"
            entry = _require_mapping(raw_entry, entry_context)
            _require_keys(entry, {"id", "text", "label", "results"}, entry_context)
            corpus_entry = {
                key: deepcopy(value) for key, value in entry.items() if key != "results"
            }
            PerturbedCorpus.from_dict([corpus_entry])
            _require_string(entry["id"], f"{entry_context}.id")
            _require_string(entry["text"], f"{entry_context}.text")
            expected_label = _require_choice(
                entry["label"],
                CORPUS_LABELS,
                f"{entry_context}.label",
            )
            result_task_names: set[str] = set()

            for result_index, raw_result in enumerate(
                _require_list(entry["results"], f"{entry_context}.results")
            ):
                result_context = f"{entry_context}.results[{result_index}]"
                result = _require_mapping(raw_result, result_context)
                _reject_keys(result, RESULT_ALIASES, result_context)
                _require_keys(result, RESULT_REQUIRED, result_context)

                task_name = _require_string(
                    result["task_name"],
                    f"{result_context}.task_name",
                )
                if task_name not in tasks_by_name:
                    raise ValueError(
                        f"{result_context}.task_name references unknown task {task_name!r}"
                    )
                if task_name in result_task_names:
                    raise ValueError(
                        f"{result_context}.task_name duplicates {task_name!r}"
                    )
                result_task_names.add(task_name)
                task = tasks_by_name[task_name]
                _require_string(result["input_text"], f"{result_context}.input_text")

                substitution_fraction = _require_number(
                    result["substitution_fraction"],
                    f"{result_context}.substitution_fraction",
                )
                if substitution_fraction < 0 or substitution_fraction > 1:
                    raise ValueError(
                        f"{result_context}.substitution_fraction must be between 0 and 1"
                    )
                self._validate_result_input_matches_task(
                    entry,
                    task,
                    result,
                    render,
                    result_context,
                )

                raw_predicted_label = result["predicted_label"]
                if raw_predicted_label is None:
                    predicted_label = None
                else:
                    predicted_label = _require_choice(
                        raw_predicted_label,
                        JUDGES,
                        f"{result_context}.predicted_label",
                    )
                if not isinstance(result["label_match"], bool):
                    raise ValueError(f"{result_context}.label_match expects a boolean")
                expected_match = (
                    False if predicted_label is None else expected_label == predicted_label
                )
                if result["label_match"] != expected_match:
                    raise ValueError(
                        f"{result_context}.label_match does not match labels"
                    )
                _require_positive_integer(
                    result["attempt_count"],
                    f"{result_context}.attempt_count",
                )
                _optional_string(
                    result["request_error"],
                    f"{result_context}.request_error",
                )

                response_context = f"{result_context}.response"
                response = _require_mapping(result["response"], response_context)
                _require_keys(
                    response,
                    {"raw", "parsed", "parse_error"},
                    response_context,
                )
                _require_string(response["raw"], f"{response_context}.raw")
                _optional_string(
                    response["parse_error"],
                    f"{response_context}.parse_error",
                )
                if response["parsed"] is None:
                    if predicted_label is not None:
                        raise ValueError(
                            f"{result_context}.predicted_label must be null when "
                            "response.parsed is null"
                        )
                    continue
                if predicted_label is None:
                    raise ValueError(
                        f"{result_context}.predicted_label cannot be null when "
                        "response.parsed is present"
                    )

                parsed_context = f"{response_context}.parsed"
                parsed = _require_mapping(response["parsed"], parsed_context)
                try:
                    validate_response_instance(parsed, response_schema, parsed_context)
                except ValueError as exc:
                    raise ValueError(
                        f"{parsed_context} does not match response schema: {exc}"
                    ) from exc
                judge = _require_choice(parsed.get("judge"), JUDGES, f"{parsed_context}.judge")
                if predicted_label != judge:
                    raise ValueError(
                        f"{parsed_context}.judge must match predicted_label"
                    )

            missing_task_names = sorted(set(tasks_by_name).difference(result_task_names))
            if missing_task_names:
                names = ", ".join(missing_task_names)
                raise ValueError(
                    f"{entry_context}.results is missing task result(s): {names}"
                )

    def _validate_result_input_matches_task(
        self,
        entry: JsonObject,
        task: TaskConfig,
        result: JsonObject,
        render: JsonObject | None,
        context: str,
    ) -> None:
        source = cast(str, task["source"])
        variant = cast(str, task["variant"])
        if source == "text":
            expected_input_text = cast(str, entry["text"])
            if task["input_type"] == "image":
                expected_input_text = apply_task_evaluation_line_breaks(
                    entry,
                    task,
                    expected_input_text,
                    render,
                )
            expected_substitution_fraction: float | int = 0
        else:
            source_values = _require_mapping(entry[source], f"{context}.{source}")
            expected_input_text = _require_string(
                source_values[variant],
                f"{context}.{source}.{variant}",
            )
            if task["input_type"] == "image":
                expected_input_text = apply_task_evaluation_line_breaks(
                    entry,
                    task,
                    expected_input_text,
                    render,
                )
            substitutions = _require_mapping(
                entry["substitutions"],
                f"{context}.substitutions",
            )
            fractions = _require_mapping(
                substitutions["fraction"],
                f"{context}.substitutions.fraction",
            )
            expected_substitution_fraction = _require_number(
                fractions[variant],
                f"{context}.substitutions.fraction.{variant}",
            )

        if result["input_text"] != expected_input_text:
            raise ValueError(f"{context}.input_text does not match task input")
        if result["substitution_fraction"] != expected_substitution_fraction:
            raise ValueError(
                f"{context}.substitution_fraction does not match task input"
            )
