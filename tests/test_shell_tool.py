"""Tests for bounded shell-command execution."""

from pathlib import Path
import time
from unittest.mock import Mock

from mini_agent.tools.shell import MAX_OUTPUT_CHARS, ShellTool


def test_run_command_captures_output_and_working_directory(tmp_path: Path) -> None:
    result = ShellTool(tmp_path, sandbox_mode="policy").run_command(
        "printf 'hello'; pwd"
    )

    assert result.success
    assert "退出码：0" in result.content
    assert "hello" in result.content
    assert str(tmp_path) in result.content


def test_run_command_reports_nonzero_exit(tmp_path: Path) -> None:
    result = ShellTool(tmp_path, sandbox_mode="policy").run_command(
        "echo problem >&2; exit 3"
    )

    assert not result.success
    assert "退出码：3" in result.content
    assert "problem" in result.content


def test_run_command_enforces_timeout(tmp_path: Path) -> None:
    result = ShellTool(tmp_path, sandbox_mode="policy").run_command(
        "sleep 2",
        timeout_seconds=0.05,
    )

    assert not result.success
    assert "已终止" in result.content


def test_run_command_rejects_invalid_timeout(tmp_path: Path) -> None:
    result = ShellTool(tmp_path, sandbox_mode="policy").run_command(
        "echo hello",
        timeout_seconds=121,
    )

    assert not result.success
    assert "必须在" in result.content


def test_run_command_blocks_dangerous_command_before_execution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "keep.txt"
    target.write_text("keep", encoding="utf-8")

    result = ShellTool(tmp_path, sandbox_mode="policy").run_command("rm keep.txt")

    assert not result.success
    assert "安全策略阻止" in result.content
    assert target.exists()


def test_run_command_truncates_oversized_output(tmp_path: Path) -> None:
    command = f"python3 -c 'print(\"x\" * {MAX_OUTPUT_CHARS + 100})'"

    result = ShellTool(tmp_path, sandbox_mode="policy").run_command(command)

    assert result.success
    assert "其余" in result.content
    assert "个字符已省略" in result.content


def test_timeout_stops_child_process_side_effect(tmp_path: Path) -> None:
    command = (
        "python3 -c 'import time; time.sleep(0.4); "
        'open("late.txt", "w").write("late")\''
    )

    result = ShellTool(tmp_path, sandbox_mode="policy").run_command(
        command,
        timeout_seconds=0.05,
    )
    time.sleep(0.6)

    assert not result.success
    assert "已终止" in result.content
    assert not (tmp_path / "late.txt").exists()


def test_explicit_network_access_requires_approval(tmp_path: Path) -> None:
    approval_handler = Mock(return_value=False)
    tool = ShellTool(
        tmp_path,
        approval_handler=approval_handler,
        sandbox_mode="policy",
    )

    result = tool.run_command("python3 -c 'print(42)'", network_access=True)

    assert not result.success
    assert "用户拒绝" in result.content
    request = approval_handler.call_args.args[0]
    assert "开放外部网络" in request.reason


def test_network_access_must_be_boolean(tmp_path: Path) -> None:
    result = ShellTool(tmp_path, sandbox_mode="policy").run_command(
        "echo hello",
        network_access="yes",
    )

    assert not result.success
    assert "必须是布尔值" in result.content


def test_policy_mode_does_not_inherit_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-leak")
    monkeypatch.setenv("SAFE_DEMO_VALUE", "visible")

    result = ShellTool(tmp_path, sandbox_mode="policy").run_command("env")

    assert result.success
    assert "must-not-leak" not in result.content
    assert "SAFE_DEMO_VALUE=visible" in result.content
