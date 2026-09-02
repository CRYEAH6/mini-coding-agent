"""Tests for environment-based application settings."""

import pytest

from mini_agent.config import ConfigurationError, Settings


def test_settings_load_required_values_and_defaults() -> None:
    settings = Settings.from_env(
        {
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "https://example.com/v1",
        }
    )

    assert settings.api_key == "test-key"
    assert settings.model == "test-model"
    assert settings.base_url == "https://example.com/v1"
    assert settings.timeout_seconds == 60.0
    assert settings.max_retries == 2
    assert settings.max_context_chars == 200_000
    assert settings.effective_summary_model == settings.model


def test_settings_load_overrides() -> None:
    settings = Settings.from_env(
        {
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "custom-model",
            "LLM_BASE_URL": "https://example.com/",
            "LLM_TIMEOUT_SECONDS": "30.5",
            "LLM_MAX_RETRIES": "4",
            "MINI_AGENT_MAX_CONTEXT_CHARS": "50000",
            "LLM_SUMMARY_MODEL": "summary-model",
        }
    )

    assert settings.model == "custom-model"
    assert settings.base_url == "https://example.com"
    assert settings.timeout_seconds == 30.5
    assert settings.max_retries == 4
    assert settings.max_context_chars == 50_000
    assert settings.effective_summary_model == "summary-model"


def test_settings_reject_missing_api_key() -> None:
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        Settings.from_env({})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("LLM_TIMEOUT_SECONDS", "not-a-number"),
        ("LLM_MAX_RETRIES", "-1"),
        ("LLM_MAX_RETRIES", "1.5"),
        ("MINI_AGENT_MAX_CONTEXT_CHARS", "0"),
        ("MINI_AGENT_MAX_CONTEXT_CHARS", "100.5"),
    ],
)
def test_settings_reject_invalid_numeric_values(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env(
            {
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
                "LLM_BASE_URL": "https://example.com/v1",
                name: value,
            }
        )
