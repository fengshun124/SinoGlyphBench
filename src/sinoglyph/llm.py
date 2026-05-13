import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sinoglyph.io import PathLike, load_toml


class LLMClientConfig:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        max_retries: int | None = None,
        **request_options: Any,
    ) -> None:
        self._validate_required_string(base_url, "base_url")
        self._validate_required_string(api_key, "api_key")
        self._validate_required_string(model, "model")
        self._validate_optional_string(system, "system")
        self._validate_timeout(timeout)

        resolved_retries = self._resolve_retries(retries, max_retries)
        self._validate_retries(resolved_retries)
        self._validate_request_options(request_options)

        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._system = system
        self._timeout = timeout
        self._retries = resolved_retries
        self._request_options = dict(request_options)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, model={self._model!r})"
        )

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> "LLMClientConfig":
        if not isinstance(mapping, dict):
            raise TypeError("LLMClientConfig.from_dict expects a mapping")
        return cls(**mapping)

    @classmethod
    def from_config(
        cls,
        config: "LLMClientConfig | dict[str, Any] | PathLike",
        *,
        section: str = "llm",
    ) -> "LLMClientConfig":
        if isinstance(config, cls):
            return config
        mapping = cls._load_config_mapping(config, section)
        return cls.from_dict(mapping)

    def export_dict(self) -> dict[str, Any]:
        data = {
            "base_url": self._base_url,
            "api_key": self._api_key,
            "model": self._model,
            **self._request_options,
        }
        if self._system is not None:
            data["system"] = self._system
        if self._timeout is not None:
            data["timeout"] = self._timeout
        if self._retries is not None:
            data["retries"] = self._retries
        return data

    def to_dict(self) -> dict[str, Any]:
        return self.export_dict()

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def system(self) -> str | None:
        return self._system

    @property
    def timeout(self) -> float | None:
        return self._timeout

    @property
    def retries(self) -> int | None:
        return self._retries

    @property
    def request_options(self) -> dict[str, Any]:
        return dict(self._request_options)

    @property
    def defaults(self) -> dict[str, Any]:
        return self.request_options

    @staticmethod
    def _validate_required_string(value: object, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a non-empty string")
        if not value:
            raise ValueError(f"{name} must be a non-empty string")

    @staticmethod
    def _validate_optional_string(value: object, name: str) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a non-empty string")
        if not value:
            raise ValueError(f"{name} must be a non-empty string")

    @staticmethod
    def _validate_timeout(timeout: object) -> None:
        if timeout is None:
            return
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a positive number")
        if timeout <= 0:
            raise ValueError("timeout must be a positive number")

    @staticmethod
    def _resolve_retries(
        retries: int | None,
        max_retries: int | None,
    ) -> int | None:
        if retries is not None and max_retries is not None and retries != max_retries:
            raise ValueError("retries and max_retries cannot disagree")
        return max_retries if max_retries is not None else retries

    @staticmethod
    def _validate_retries(retries: object) -> None:
        if retries is None:
            return
        if isinstance(retries, bool) or not isinstance(retries, int):
            raise TypeError("retries must be a non-negative integer")
        if retries < 0:
            raise ValueError("retries must be a non-negative integer")

    @staticmethod
    def _validate_request_options(request_options: dict[str, Any]) -> None:
        for key in request_options:
            if not isinstance(key, str) or not key:
                raise TypeError("request option keys must be non-empty strings")

    @staticmethod
    def _load_config_mapping(
        config: dict[str, Any] | PathLike,
        section: str,
    ) -> dict[str, Any]:
        if isinstance(config, dict):
            raw = config
        else:
            raw = load_toml(config)

        mapping = raw.get(section, raw)
        if not isinstance(mapping, dict):
            raise TypeError(f"{section!r} config section must be a mapping")
        return mapping


class LLMClient:
    def __init__(
        self,
        base_url: str | LLMClientConfig,
        api_key: str | None = None,
        model: str | None = None,
        system: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        max_retries: int | None = None,
        **request_options: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required. Install it before using LLMClient."
            ) from exc

        if isinstance(base_url, LLMClientConfig):
            if api_key is not None or model is not None or system is not None:
                raise TypeError(
                    "api_key, model, and system cannot be passed with LLMClientConfig"
                )
            if timeout is not None or retries is not None or max_retries is not None:
                raise TypeError(
                    "timeout, retries, and max_retries cannot be passed with "
                    "LLMClientConfig"
                )
            if request_options:
                raise TypeError("request options cannot be passed with LLMClientConfig")
            config = base_url
        else:
            config = LLMClientConfig(
                base_url=base_url,
                api_key="" if api_key is None else api_key,
                model="" if model is None else model,
                system=system,
                timeout=timeout,
                retries=retries,
                max_retries=max_retries,
                **request_options,
            )

        self._config = config
        self._base_url = config.base_url
        self._api_key = config.api_key
        self._model = config.model
        self._request_options = config.request_options
        self._history: list[dict[str, Any]] = []
        client_options: dict[str, Any] = {}
        if config.retries is not None:
            client_options["max_retries"] = config.retries
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            **client_options,
        )
        if config.system:
            self.set_system(config.system)

    @classmethod
    def from_config(
        cls,
        config: LLMClientConfig | dict[str, Any] | PathLike,
        *,
        section: str = "llm",
    ) -> "LLMClient":
        return cls(LLMClientConfig.from_config(config, section=section))

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def request_options(self) -> dict[str, Any]:
        return dict(self._request_options)

    @property
    def history(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self._history]

    def set_system(self, prompt: str) -> None:
        if not prompt:
            raise ValueError("System prompt is required.")
        self._history.append({"role": "system", "content": prompt})

    def clear(self, keep_system: bool = True) -> None:
        if keep_system:
            self._history = [msg for msg in self._history if msg["role"] == "system"]
        else:
            self._history = []

    def chat(
        self,
        text: str | None = None,
        images: list[str] | None = None,
        files: list[str] | None = None,
        model: str | None = None,
        json_schema: dict[str, Any] | None = None,
        attempts: int = 3,
        **kwargs: Any,
    ) -> Any:
        if not text and not images and not files:
            raise ValueError("At least one of text, images, or files is required.")
        if attempts < 1:
            raise ValueError("attempts must be at least 1.")
        if model and model != self._model:
            self._validate_model(model)
            self._model = model

        user_msg = {
            "role": "user",
            "content": {
                "text": text or "",
                "images": list(images or []),
                "files": list(files or []),
            },
        }
        self._history.append(user_msg)

        options = {**self._request_options, **kwargs}
        last_error: Exception | None = None
        jsonschema_validate = None
        jsonschema_validation_error: type[Exception] = ValueError
        if json_schema is not None:
            try:
                from jsonschema import (
                    ValidationError as JsonSchemaValidationError,
                )
                from jsonschema import (
                    validate as jsonschema_validate,
                )
            except ImportError as exc:
                raise ImportError(
                    "The jsonschema package is required for JSON schema validation."
                ) from exc
            jsonschema_validation_error = JsonSchemaValidationError

        for _ in range(attempts):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(exclude_failed_json=json_schema is not None),
                **options,
            )
            reply = self._reply_text(response)

            if json_schema is None:
                self._history.append(
                    {
                        "role": "assistant",
                        "model": self._model,
                        "response": reply,
                        "json_valid": None,
                    }
                )
                return reply

            try:
                parsed = parse_json_response(reply)
                jsonschema_validate(instance=parsed, schema=json_schema)
            except (json.JSONDecodeError, jsonschema_validation_error) as exc:
                last_error = exc
                self._history.append(
                    {
                        "role": "assistant",
                        "model": self._model,
                        "response": reply,
                        "json_valid": False,
                    }
                )
                continue

            self._history.append(
                {
                    "role": "assistant",
                    "model": self._model,
                    "response": reply,
                    "json_valid": True,
                }
            )
            return parsed

        raise ValueError(
            f"JSON response did not match schema after {attempts} attempts."
        ) from last_error

    def _validate_model(self, model: str) -> None:
        if not model:
            raise ValueError("Model is required.")
        models = self._client.models.list()
        ids = {item.id for item in models.data}
        if model not in ids:
            raise ValueError(f"Model '{model}' is not available from {self.base_url}.")

    def _messages(self, exclude_failed_json: bool = False) -> list[dict[str, Any]]:
        messages = []
        for msg in self._history:
            role = msg["role"]
            if role == "system":
                messages.append({"role": "system", "content": msg["content"]})
            elif role == "user":
                messages.append(
                    {"role": "user", "content": self._user_parts(msg["content"])}
                )
            elif role == "assistant":
                if exclude_failed_json and msg.get("json_valid") is False:
                    continue
                messages.append({"role": "assistant", "content": msg["response"]})
        return messages

    def _user_parts(self, content: dict[str, Any]) -> list[dict[str, Any]]:
        parts = []
        text = content.get("text")
        file_text = [self._file_text(item) for item in content.get("files", [])]
        if text or file_text:
            combined = "\n\n".join(part for part in [text, *file_text] if part)
            parts.append({"type": "text", "text": combined})
        for image in content.get("images", []):
            parts.append(
                {"type": "image_url", "image_url": {"url": self._image_url(image)}}
            )
        return parts

    def _image_url(self, value: str) -> str:
        if self._is_url(value):
            return value
        path = Path(value)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def _file_text(self, value: str) -> str:
        if self._is_url(value):
            return f"File URL: {value}"
        path = Path(value)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        try:
            body = data.decode("utf-8")
            return f"File: {path.name}\nMIME: {mime}\n\n{body}"
        except UnicodeDecodeError:
            encoded = base64.b64encode(data).decode("ascii")
            return f"File: {path.name}\nMIME: {mime}\nBase64:\n{encoded}"

    @staticmethod
    def _is_url(value: str) -> bool:
        return urlparse(value).scheme in {"http", "https", "data"}

    @staticmethod
    def _reply_text(response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return "" if content is None else str(content)


def parse_json_response(text: str) -> Any:
    return json.loads(_strip_json_markdown_fence(text))


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
