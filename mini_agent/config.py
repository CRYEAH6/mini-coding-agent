"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from typing import Mapping, Optional

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Validated settings used to connect to an LLM API."""

    api_key: str
    model: str = ""
    base_url: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 2
    max_context_chars: int = 200_000
    summary_model: Optional[str] = None

    @property
    def effective_summary_model(self) -> str:
        """Use the main model unless a dedicated summary model is configured."""
        return self.summary_model or self.model

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        """Create settings from a mapping or the process environment."""
        if environ is None:
            load_dotenv()
            environ = os.environ

        api_key = environ.get("LLM_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("缺少环境变量 LLM_API_KEY。")

        model = environ.get("LLM_MODEL", "").strip()
        if not model:
            raise ConfigurationError("缺少环境变量 LLM_MODEL。")

        base_url = environ.get("LLM_BASE_URL", "").strip().rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ConfigurationError("LLM_BASE_URL 必须是有效的 HTTP(S) 地址。")

        timeout_seconds = _parse_positive_float(
            environ.get("LLM_TIMEOUT_SECONDS", str(cls.timeout_seconds)),
            "LLM_TIMEOUT_SECONDS",
        )
        max_retries = _parse_non_negative_int(
            environ.get("LLM_MAX_RETRIES", str(cls.max_retries)),
            "LLM_MAX_RETRIES",
        )
        max_context_chars = _parse_positive_int(
            environ.get(
                "MINI_AGENT_MAX_CONTEXT_CHARS",
                str(cls.max_context_chars),
            ),
            "MINI_AGENT_MAX_CONTEXT_CHARS",
        )
        summary_model = environ.get("LLM_SUMMARY_MODEL", "").strip()

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_context_chars=max_context_chars,
            summary_model=summary_model or None,
        )


def _parse_positive_float(raw_value: str, name: str) -> float:
    """Parse a positive floating-point environment value."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须是数字。") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} 必须大于 0。")
    return value


def _parse_non_negative_int(raw_value: str, name: str) -> int:
    """Parse a non-negative integer environment value."""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须是整数。") from exc
    if value < 0:
        raise ConfigurationError(f"{name} 不能小于 0。")
    return value


def _parse_positive_int(raw_value: str, name: str) -> int:
    """Parse a positive integer environment value."""
    value = _parse_non_negative_int(raw_value, name)
    if value == 0:
        raise ConfigurationError(f"{name} 必须大于 0。")
    return value
