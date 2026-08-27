"""Tests for bounded shell-command execution."""

from pathlib import Path

from mini_agent.tools.shell import ShellTool


def test_run_command_captures_output_and_working_directory(tmp_path: Path) -> None:
    result = ShellTool(tmp_path).run_command("printf 'hello'; pwd")

    assert result.success
    assert "退出码：0" in result.content
    assert "hello" in result.content
    assert str(tmp_path) in result.content


def test_run_command_reports_nonzero_exit(tmp_path: Path) -> None:
    result = ShellTool(tmp_path).run_command("echo problem >&2; exit 3")

    assert not result.success
    assert "退出码：3" in result.content
    assert "problem" in result.content


def test_run_command_enforces_timeout(tmp_path: Path) -> None:
    result = ShellTool(tmp_path).run_command("sleep 2", timeout_seconds=0.05)

    assert not result.success
    assert "已终止" in result.content


def test_run_command_rejects_invalid_timeout(tmp_path: Path) -> None:
    result = ShellTool(tmp_path).run_command("echo hello", timeout_seconds=121)

    assert not result.success
    assert "必须在" in result.content
