from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Iterator

from sinoglyph.io import load_json, save_json
from sinoglyph.schema.base import JsonObject


class EvaluationCache:
    def __init__(self, cache_dir: Path, fingerprint: str) -> None:
        self.cache_dir = cache_dir
        self.fingerprint = fingerprint

    def build_entry_path(self, index: int, entry_id: str) -> Path:
        return build_cache_entry_path(self.cache_dir, index, entry_id)

    def load_entry(self, index: int, entry_id: str) -> JsonObject | None:
        cache_path = self.build_entry_path(index, entry_id)
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
    def lock_run(self) -> Iterator[None]:
        with _lock_cache_run(self.cache_dir, self.fingerprint):
            yield

    @contextmanager
    def lock_entry(self, index: int, entry_id: str) -> Iterator[None]:
        with _lock_cache_entry(self.cache_dir, index, entry_id):
            yield


def prepare_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode = cache_dir.stat().st_mode
    if mode & 0o077:
        os.chmod(cache_dir, mode & ~0o077)


def build_cache_entry_path(cache_dir: Path, index: int, entry_id: str) -> Path:
    return cache_dir / "entries" / f"{index:06d}-{build_safe_name(entry_id)}.json"


def build_safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    safe = "".join(character if character.isalnum() else "-" for character in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{safe[:80]}-{digest}" if safe else digest


@contextmanager
def _lock_cache_run(cache_dir: Path, fingerprint: str) -> Iterator[None]:
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
            if _is_cache_lock_stale(lock_path):
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
def _lock_cache_entry(cache_dir: Path, index: int, entry_id: str) -> Iterator[None]:
    lock_path = cache_dir / f".entry-{index}-{build_safe_name(entry_id)}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_path.unlink(missing_ok=True)


def _is_cache_lock_stale(lock_path: Path) -> bool:
    try:
        raw = load_json(lock_path)
    except Exception:
        return _get_cache_lock_age_seconds(lock_path) > 60
    if not isinstance(raw, dict):
        return _get_cache_lock_age_seconds(lock_path) > 60
    pid = raw.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return _get_cache_lock_age_seconds(lock_path) > 60
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _get_cache_lock_age_seconds(lock_path: Path) -> float:
    try:
        return time() - lock_path.stat().st_mtime
    except OSError:
        return 0
