"""Tests for bounded shell-command execution."""

from pathlib import Path
import time

from mini_agent.tools.shell import MAX_OUTPUT_CHARS, ShellTool


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


def test_run_command_blocks_dangerous_command_before_execution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "keep.txt"
    target.write_text("keep", encoding="utf-8")

    result = ShellTool(tmp_path).run_command("rm keep.txt")

    assert not result.success
    assert "安全策略阻止" in result.content
    assert target.exists()


def test_run_command_truncates_oversized_output(tmp_path: Path) -> None:
    command = f"python3 -c 'print(\"x\" * {MAX_OUTPUT_CHARS + 100})'"

    result = ShellTool(tmp_path).run_command(command)

    assert result.success
    assert "其余" in result.content
    assert "个字符已省略" in result.content


def test_timeout_stops_child_process_side_effect(tmp_path: Path) -> None:
    command = (
        "python3 -c 'import time; time.sleep(0.4); "
        'open("late.txt", "w").write("late")\''
    )

    result = ShellTool(tmp_path).run_command(command, timeout_seconds=0.05)
    time.sleep(0.6)

    assert not result.success
    assert "已终止" in result.content
    assert not (tmp_path / "late.txt").exists()
