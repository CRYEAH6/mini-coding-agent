"""Tests for the command-line interface."""

from unittest.mock import Mock

from mini_agent.cli import build_parser, main
from mini_agent.config import ConfigurationError, Settings


def test_parser_reads_task_workspace_and_step_limit() -> None:
    args = build_parser().parse_args(
        ["Fix the bug", "--workspace", "example", "--max-steps", "8"]
    )

    assert args.task == "Fix the bug"
    assert args.workspace == "example"
    assert args.max_steps == 8


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
