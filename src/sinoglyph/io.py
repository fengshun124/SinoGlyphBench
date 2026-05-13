import json
import os
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeAlias

import tomllib

PathLike: TypeAlias = str | os.PathLike[str]


def load_json(file_path: PathLike) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: PathLike, *, warn_overwrite: bool = True) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=4)
    _atomic_write_text(text, file_path, warn_overwrite=warn_overwrite)


def load_toml(file_path: PathLike) -> dict[str, Any]:
    with open(file_path, "rb") as f:
        return tomllib.load(f)


def load_env_file(file_path: PathLike = ".env", *, override: bool = False) -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=file_path, override=override)


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
