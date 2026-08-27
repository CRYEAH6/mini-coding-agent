"""Tests for the command-line interface."""

from unittest.mock import Mock

from pathlib import Path

from mini_agent.cli import _print_startup, build_parser, main
from mini_agent.config import ConfigurationError, Settings


def test_parser_reads_task_workspace_and_step_limit() -> None:
    args = build_parser().parse_args(
        [
            "Fix the bug",
            "--workspace",
            "example",
            "--max-steps",
            "8",
            "--allow-dangerous-commands",
        ]
    )

    assert args.task == "Fix the bug"
    assert args.workspace == "example"
    assert args.max_steps == 8
    assert args.allow_dangerous_commands


def test_main_reports_missing_api_key(monkeypatch, capsys) -> None:
    missing_key = Mock(
        side_effect=ConfigurationError("缺少环境变量 DEEPSEEK_API_KEY。")
    )
    monkeypatch.setattr(Settings, "from_env", missing_key)

    exit_code = main(["Fix the bug"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "DEEPSEEK_API_KEY" in captured.out
    missing_key.assert_called_once_with()


def test_startup_summary_is_clear_and_secret_free(capsys) -> None:
    _print_startup(
        model="test-model",
        workspace=Path("/tmp/example"),
        max_steps=12,
        max_context_chars=100_000,
        dangerous_commands=False,
    )

    output = capsys.readouterr().out
    assert "模型：test-model" in output
    assert "工作目录：/tmp/example" in output
    assert "最大步骤：12" in output
    assert "上下文预算：100000 字符" in output
    assert "高风险命令默认拦截" in output
