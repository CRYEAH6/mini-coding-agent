"""Tests for strict macOS and policy-only command execution plans."""

from pathlib import Path

import pytest

from mini_agent.tools.sandbox import (
    CommandSandbox,
    SandboxError,
    _build_profile,
    _filtered_environment,
    _strict_environment,
)


def test_strict_profile_limits_writes_and_denies_network_by_default(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    temporary = tmp_path / "command-temp"
    workspace.mkdir()
    temporary.mkdir()

    profile = _build_profile(workspace, temporary, allow_network=False)

    assert "(deny default)" in profile
    assert f'(subpath "{workspace}")' in profile
    assert f'(subpath "{temporary}")' in profile
    assert "(allow file-write*" in profile
    assert "(allow network-outbound)" not in profile


def test_strict_profile_opens_network_only_for_approved_command(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    temporary = tmp_path / "command-temp"
    workspace.mkdir()
    temporary.mkdir()

    profile = _build_profile(workspace, temporary, allow_network=True)

    assert "(allow network-outbound)" in profile


def test_strict_environment_uses_isolated_home_cache_and_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    venv_bin = workspace / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    home = tmp_path / "home"
    cache = tmp_path / "cache"
    command_tmp = tmp_path / "tmp"

    environment = _strict_environment(
        workspace,
        home,
        cache,
        command_tmp,
    )

    assert environment["HOME"] == str(home)
    assert environment["TMPDIR"] == str(command_tmp)
    assert environment["PATH"].split(":")[0] == str(venv_bin)
    assert "DEEPSEEK_API_KEY" not in environment


def test_policy_environment_removes_credential_like_variables() -> None:
    filtered = _filtered_environment(
        {
            "PATH": "/usr/bin",
            "DEEPSEEK_API_KEY": "secret",
            "ACCESS_TOKEN": "token",
            "SAFE_VALUE": "visible",
        }
    )

    assert filtered == {"PATH": "/usr/bin", "SAFE_VALUE": "visible"}


def test_policy_mode_uses_non_login_shell(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    sandbox = CommandSandbox(tmp_path, mode="policy")

    with sandbox.prepare("echo hello", allow_network=False) as plan:
        assert plan.arguments == ("/bin/zsh", "-c", "echo hello")
        assert "DEEPSEEK_API_KEY" not in plan.environment


def test_strict_mode_rejects_non_macos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("mini_agent.tools.sandbox.platform.system", lambda: "Linux")
    sandbox = CommandSandbox(tmp_path, mode="strict")

    with pytest.raises(SandboxError, match="只支持 macOS"):
        with sandbox.prepare("echo hello", allow_network=False):
            pass


def test_unknown_sandbox_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未知沙箱模式"):
        CommandSandbox(tmp_path, mode="unknown")
