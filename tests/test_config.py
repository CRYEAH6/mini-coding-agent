"""Tests for environment-based application settings."""

import pytest

from mini_agent.config import ConfigurationError, Settings


def test_settings_load_required_key_and_defaults() -> None:
    settings = Settings.from_env({"DEEPSEEK_API_KEY": "test-key"})

    assert settings.api_key == "test-key"
    assert settings.model == "deepseek-v4-flash"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.timeout_seconds == 60.0
    assert settings.max_retries == 2
    assert settings.max_context_chars == 200_000


def test_settings_load_overrides() -> None:
    settings = Settings.from_env(
        {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_MODEL": "custom-model",
            "DEEPSEEK_BASE_URL": "https://example.com/",
            "DEEPSEEK_TIMEOUT_SECONDS": "30.5",
            "DEEPSEEK_MAX_RETRIES": "4",
            "DEEPSEEK_MAX_CONTEXT_CHARS": "50000",
        }
    )

    assert settings.model == "custom-model"
    assert settings.base_url == "https://example.com"
    assert settings.timeout_seconds == 30.5
    assert settings.max_retries == 4
    assert settings.max_context_chars == 50_000


def test_settings_reject_missing_api_key() -> None:
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        Settings.from_env({})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DEEPSEEK_TIMEOUT_SECONDS", "0"),
        ("DEEPSEEK_TIMEOUT_SECONDS", "not-a-number"),
        ("DEEPSEEK_MAX_RETRIES", "-1"),
        ("DEEPSEEK_MAX_RETRIES", "1.5"),
        ("DEEPSEEK_MAX_CONTEXT_CHARS", "0"),
        ("DEEPSEEK_MAX_CONTEXT_CHARS", "100.5"),
    ],
)
def test_settings_reject_invalid_numeric_values(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({"DEEPSEEK_API_KEY": "test-key", name: value})
