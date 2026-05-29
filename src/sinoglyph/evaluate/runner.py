from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tqdm.auto import tqdm

from sinoglyph.evaluate.cache import (
    EvaluationCache,
    build_cache_entry_path,
    prepare_cache_dir,
)
from sinoglyph.evaluate.prompt import append_response_contract, build_response_contract
from sinoglyph.evaluate.task import run_task_evaluation
from sinoglyph.io import PathLike, load_env_file, save_json
from sinoglyph.schema.base import JsonObject
from sinoglyph.schema.corpus import load_obfuscated_corpus
from sinoglyph.schema.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationTask,
)
from sinoglyph.schema.types import Modality
from sinoglyph.schema.utils import require_string


@dataclass(frozen=True)
class _EvaluationJob:
    index: int
    entry: JsonObject


@dataclass(frozen=True)
class _EvaluationJobResult:
    index: int
    entry: JsonObject


@dataclass(frozen=True)
class _CacheMergeResult:
    entry: JsonObject
    reused_results: int = 0
    purged_failures: dict[str, int] | None = None
    cache_changed: bool = False


_PURGEABLE_RESUME_FAILURE_KINDS = frozenset({"no_quota", "rate_limit"})


def run_evaluation(
    config_path: PathLike,
    output_path: PathLike | None = None,
    cache_dir: PathLike | None = None,
    n_jobs: int | None = None,
    env_file: PathLike | None = None,
    env_override: bool = False,
) -> JsonObject:
    load_env_file(
        ".env" if env_file is None else env_file,
        override=env_override,
    )
    config = EvaluationConfig.load_toml(config_path)
    settings = config.evaluation
    fingerprint = config.compute_fingerprint()
    meta = config.build_result_meta()
    resolved_cache_dir = Path(
        cache_dir if cache_dir is not None else settings.cache_dir
    ).expanduser()
    resolved_n_jobs = n_jobs if n_jobs is not None else settings.n_jobs
    if resolved_n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer")
    prepare_cache_dir(resolved_cache_dir)
    cache = EvaluationCache(resolved_cache_dir, fingerprint)

    corpus = load_obfuscated_corpus(settings.corpus_path)
    selected_corpus = corpus if settings.limit is None else corpus[: settings.limit]
    limited_corpus = [entry.export_mapping() for entry in selected_corpus]

    with cache.lock_run():
        chat_config = config.create_chat_config().export_dict()
        max_retries = cast(int, chat_config.get("max_retries", 0))
        max_tries = max_retries + 1
        chat_config["max_retries"] = 0
        tasks = config.tasks
        needs_render = any(task.modality == Modality.IMAGE for task in tasks)
        render_mapping = config.render if needs_render else None
        render_config_mapping = (
            config.build_render_config_mapping() if needs_render else None
        )
        prompt = config.prompt
        response_schema = config.response_schema

        cached_entries: dict[int, JsonObject] = {}
        jobs: list[_EvaluationJob] = []
        resumed_entries = 0
        resumed_results = 0
        purged_failures = {kind: 0 for kind in _PURGEABLE_RESUME_FAILURE_KINDS}
        for index, entry in enumerate(limited_corpus):
            entry_id = require_string(entry["id"], "entry.id")
            cached = cache.load_entry(index, entry_id)
            merge = _merge_cached_results(entry, cached, meta, tasks)
            entry = merge.entry
            resumed_results += merge.reused_results
            if merge.purged_failures is not None:
                for kind, count in merge.purged_failures.items():
                    purged_failures[kind] = purged_failures.get(kind, 0) + count
            if merge.cache_changed:
                cache_path = cache.build_entry_path(index, entry_id)
                with cache.lock_entry(index, entry_id):
                    cache.save_entry(entry, index, cache_path)
            if entry.get("results"):
                resumed_entries += 1
            if _has_all_task_results(entry, tasks):
                cached_entries[index] = entry
            else:
                jobs.append(_EvaluationJob(index=index, entry=entry))

        _write_resume_summary(
            resumed_entries,
            resumed_results,
            purged_failures,
            len(jobs),
            resolved_cache_dir,
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
                    chat_config,
                    prompt.text_prompt,
                    prompt.image_prompt,
                    response_schema,
                    render_mapping,
                    render_config_mapping,
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
            "tasks": [task.export_mapping() for task in tasks],
            "corpus": ordered_corpus,
        }
        EvaluationResult.parse_mapping(output)

        resolved_output_path = Path(
            output_path
            if output_path is not None
            else Path(settings.output_dir) / f"{settings.name}.json"
        ).expanduser()
        save_json(output, resolved_output_path)
        return output


def _evaluate_corpus_entry(
    job: _EvaluationJob,
    tasks: list[EvaluationTask],
    max_tries: int,
    chat_config: dict[str, object],
    text_prompt: str,
    image_prompt: str,
    response_schema: JsonObject,
    render_mapping: JsonObject | None,
    render_config_mapping: JsonObject | None,
    cache_dir: Path,
    fingerprint: str,
) -> _EvaluationJobResult:
    from sinoglyph.llm import ChatClient, ChatClientConfig
    from sinoglyph.render import TextRenderConfig

    cache = EvaluationCache(cache_dir, fingerprint)
    render_config = (
        None
        if render_config_mapping is None
        else TextRenderConfig.parse_dict(render_config_mapping)
    )
    response_contract = build_response_contract(response_schema)
    entry = deepcopy(job.entry)
    results_by_task_name = _map_results_by_task_name(entry, tasks)
    cache_path = build_cache_entry_path(
        cache_dir, job.index, require_string(entry["id"], "entry.id")
    )
    client = ChatClient(ChatClientConfig.parse_dict(chat_config))
    for task in tasks:
        if task.name not in results_by_task_name:
            results_by_task_name[task.name] = run_task_evaluation(
                entry,
                task,
                max_tries,
                client,
                append_response_contract(text_prompt, response_contract),
                append_response_contract(image_prompt, response_contract),
                response_schema,
                render_mapping,
                render_config,
                cache_dir,
            )
            entry["results"] = _order_results(results_by_task_name, tasks)
            entry_id = require_string(entry["id"], "entry.id")
            with cache.lock_entry(job.index, entry_id):
                cache.save_entry(entry, job.index, cache_path)
    entry["results"] = _order_results(results_by_task_name, tasks)
    entry_id = require_string(entry["id"], "entry.id")
    with cache.lock_entry(job.index, entry_id):
        cache.save_entry(entry, job.index, cache_path)
    return _EvaluationJobResult(index=job.index, entry=entry)


def _merge_cached_results(
    entry: JsonObject,
    cached_entry: JsonObject | None,
    meta: JsonObject,
    tasks: list[EvaluationTask],
) -> _CacheMergeResult:
    resumed = deepcopy(entry)
    if cached_entry is None:
        return _CacheMergeResult(entry=resumed)
    cached_results = cached_entry.get("results")
    if not isinstance(cached_results, list):
        return _CacheMergeResult(entry=resumed)
    tasks_by_name = {task.name: task for task in tasks}
    results_by_task_name: dict[str, JsonObject] = {}
    purged_failures = {kind: 0 for kind in _PURGEABLE_RESUME_FAILURE_KINDS}
    for result in cached_results:
        if not isinstance(result, dict):
            continue
        failure_kind = result.get("failure_kind")
        if (
            isinstance(failure_kind, str)
            and failure_kind in _PURGEABLE_RESUME_FAILURE_KINDS
        ):
            purged_failures[failure_kind] += 1
            continue
        task_name = result.get("task_name")
        if not isinstance(task_name, str) or task_name in results_by_task_name:
            continue
        task = tasks_by_name.get(task_name)
        if task is None:
            continue
        candidate = cast(JsonObject, deepcopy(result))
        if _is_reusable_result(candidate) and _is_valid_cached_result(
            resumed,
            task,
            candidate,
            meta,
        ):
            results_by_task_name[task_name] = candidate
    if results_by_task_name:
        resumed["results"] = _order_results(results_by_task_name, tasks)
    total_purged = sum(purged_failures.values())
    return _CacheMergeResult(
        entry=resumed,
        reused_results=len(results_by_task_name),
        purged_failures=purged_failures if total_purged else None,
        cache_changed=total_purged > 0,
    )


def _write_resume_summary(
    resumed_entries: int,
    resumed_results: int,
    purged_failures: dict[str, int],
    pending_jobs: int,
    cache_dir: Path,
) -> None:
    purged_total = sum(purged_failures.values())
    if not resumed_results and not purged_total:
        return
    if resumed_results:
        tqdm.write(
            "Resuming evaluation from cache: reused "
            f"{resumed_results} task {_plural(resumed_results, 'result')} across "
            f"{resumed_entries} corpus {_plural(resumed_entries, 'entry', 'entries')}."
        )
        tqdm.write(
            f"Use --cache-dir with a new path or remove {cache_dir} for a fresh start."
        )
    if purged_total:
        details = ", ".join(
            f"{kind}={count}"
            for kind, count in sorted(purged_failures.items())
            if count
        )
        tqdm.write(
            "Purged "
            f"{purged_total} cached quota/rate-limit {_plural(purged_total, 'leftover')} "
            f"({details}); affected tasks will be retried."
        )
    if pending_jobs:
        tqdm.write(
            f"Evaluation has {pending_jobs} corpus "
            f"{_plural(pending_jobs, 'entry', 'entries')} with pending work."
        )


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def _is_reusable_result(result: JsonObject) -> bool:
    response = result.get("response")
    return (
        isinstance(response, dict)
        and isinstance(response.get("parsed"), dict)
        and response.get("parse_error") is None
        and result.get("request_error") is None
        and result.get("failure_kind") is None
        and isinstance(result.get("predicted_label"), str)
    )


def _is_valid_cached_result(
    entry: JsonObject,
    task: EvaluationTask,
    result: JsonObject,
    meta: JsonObject,
) -> bool:
    candidate_entry = deepcopy(entry)
    candidate_entry["results"] = [result]
    try:
        EvaluationResult.parse_mapping(
            {
                "meta": meta,
                "tasks": [task.export_mapping()],
                "corpus": [candidate_entry],
            }
        )
    except Exception:
        return False
    return True


def _has_all_task_results(entry: JsonObject, tasks: list[EvaluationTask]) -> bool:
    return set(_map_results_by_task_name(entry, tasks, reusable_only=True)) == {
        task.name for task in tasks
    }


def _map_results_by_task_name(
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
        if reusable_only and not _is_reusable_result(cast(JsonObject, result)):
            continue
        results_by_task_name.setdefault(task_name, cast(JsonObject, result))
    return results_by_task_name


def _order_results(
    results_by_task_name: dict[str, JsonObject],
    tasks: list[EvaluationTask],
) -> list[JsonObject]:
    return [
        results_by_task_name[task.name]
        for task in tasks
        if task.name in results_by_task_name
    ]
