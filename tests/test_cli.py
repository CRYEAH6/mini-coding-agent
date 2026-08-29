"""Tests for the command-line interface."""

from pathlib import Path
from unittest.mock import Mock

import mini_agent.cli as cli_module
from mini_agent.agent import AgentResult
from mini_agent.cli import _confirm_tool_action, _print_startup, build_parser, main
from mini_agent.config import ConfigurationError, Settings
from mini_agent.tools.approval import ApprovalRequest


def test_parser_reads_workspace_and_step_limit() -> None:
    args = build_parser().parse_args(
        [
            "--workspace",
            "example",
            "--max-steps",
            "8",
            "--allow-dangerous-commands",
        ]
    )

    assert args.workspace == "example"
    assert args.max_steps == 8
    assert args.allow_dangerous_commands


def test_main_reports_missing_api_key(monkeypatch, capsys) -> None:
    missing_key = Mock(
        side_effect=ConfigurationError("缺少环境变量 DEEPSEEK_API_KEY。")
    )
    monkeypatch.setattr(Settings, "from_env", missing_key)

    exit_code = main([])

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


def test_main_reports_api_error(monkeypatch, capsys, tmp_path: Path) -> None:
    class FakeAPIError(Exception):
        pass

    monkeypatch.setattr(cli_module, "APIError", FakeAPIError)
    monkeypatch.setattr(
        Settings,
        "from_env",
        Mock(return_value=Settings(api_key="test-key")),
    )
    monkeypatch.setattr(cli_module, "DeepSeekClient", Mock(return_value=object()))
    monkeypatch.setattr(
        cli_module.CodingAgent,
        "run_turn",
        Mock(side_effect=FakeAPIError("network unavailable")),
    )
    monkeypatch.setattr("builtins.input", Mock(side_effect=["Fix the bug", "/exit"]))

    exit_code = main(["--workspace", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "API 请求失败：network unavailable" in output
    assert "会话已结束" in output


def test_interactive_mode_keeps_accepting_tasks_until_exit(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        Settings,
        "from_env",
        Mock(return_value=Settings(api_key="test-key")),
    )
    monkeypatch.setattr(cli_module, "DeepSeekClient", Mock(return_value=object()))
    run_turn = Mock(
        side_effect=[
            AgentResult("第一轮完成", 1, 0, 0),
            AgentResult("第二轮完成", 1, 0, 0),
        ]
    )
    reset_session = Mock()
    monkeypatch.setattr(cli_module.CodingAgent, "run_turn", run_turn)
    monkeypatch.setattr(cli_module.CodingAgent, "reset_session", reset_session)
    user_input = Mock(
        side_effect=["创建飞机大战", "/new", "增加暂停功能", "/exit"]
    )
    monkeypatch.setattr("builtins.input", user_input)

    exit_code = main(["--workspace", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert run_turn.call_args_list[0].args == ("创建飞机大战",)
    assert run_turn.call_args_list[1].args == ("增加暂停功能",)
    assert reset_session.call_count == 2
    assert "第一轮完成" in output
    assert "第二轮完成" in output
    assert "已清空对话历史" in output
    assert "会话已结束" in output


def test_confirmation_accepts_explicit_yes(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", Mock(return_value="y"))
    request = ApprovalRequest(
        tool_name="run_command",
        action="执行敏感命令",
        details="uv add requests",
        reason="该命令会安装项目依赖。",
    )

    allowed = _confirm_tool_action(request)

    output = capsys.readouterr().out
    assert allowed
    assert "uv add requests" in output
    assert "已允许" in output


def test_confirmation_defaults_to_denial(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", Mock(return_value=""))
    request = ApprovalRequest(
        tool_name="run_command",
        action="执行敏感命令",
        details="git push origin main",
        reason="该命令会与远程仓库交互。",
    )

    allowed = _confirm_tool_action(request)

    output = capsys.readouterr().out
    assert not allowed
    assert "已拒绝" in output
