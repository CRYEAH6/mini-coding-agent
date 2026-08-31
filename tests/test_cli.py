"""Tests for the command-line interface."""

from pathlib import Path
from unittest.mock import Mock

import mini_agent.cli as cli_module
import pytest
from mini_agent.agent import AgentResult
from mini_agent.cli import _confirm_tool_action, _print_startup, build_parser, main
from mini_agent.config import ConfigurationError, Settings
from mini_agent.session import SessionError, SessionStore
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
    assert args.sandbox_mode == "strict"


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
        sandbox_mode="strict",
    )

    output = capsys.readouterr().out
    assert "模型：test-model" in output
    assert "工作目录：/tmp/example" in output
    assert "最大步骤：12" in output
    assert "上下文预算：100000 字符" in output
    assert "摘要模型：test-model" in output
    assert "高风险命令默认拦截" in output
    assert "命令隔离：macOS 系统沙箱" in output


def test_main_reports_api_error(monkeypatch, capsys, tmp_path: Path) -> None:
    class FakeAPIError(Exception):
        pass

    monkeypatch.setattr(cli_module, "APIError", FakeAPIError)
    monkeypatch.setenv("MINI_AGENT_SESSION_DIR", str(tmp_path / "sessions"))
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
    monkeypatch.setenv("MINI_AGENT_SESSION_DIR", str(tmp_path / "sessions"))
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
    assert "已创建新会话" in output
    assert "旧会话和工作目录中的文件保持不变" in output
    assert "会话已结束" in output


def test_interactive_memory_management_commands(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINI_AGENT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("MINI_AGENT_MEMORY_DIR", str(tmp_path / "memories"))
    monkeypatch.setattr(
        Settings,
        "from_env",
        Mock(return_value=Settings(api_key="test-key")),
    )
    monkeypatch.setattr(cli_module, "DeepSeekClient", Mock(return_value=object()))
    monkeypatch.setattr(
        "mini_agent.memory.secrets.token_hex",
        Mock(return_value="a1b2c3d4"),
    )
    monkeypatch.setattr(
        "builtins.input",
        Mock(
            side_effect=[
                "/remember README 使用中文",
                "/memories",
                "/forget mem-a1b2c3d4",
                "/memories",
                "/exit",
            ]
        ),
    )

    exit_code = main(["--workspace", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "已保存长期记忆：mem-a1b2c3d4" in output
    assert "README 使用中文" in output
    assert "已删除长期记忆：mem-a1b2c3d4" in output
    assert "当前工作目录还没有长期记忆" in output


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


def test_main_restores_active_persistent_session(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    store = SessionStore(tmp_path, root=session_root)
    record = store.create()
    saved = store.save(
        record.session_id,
        [
            {"role": "user", "content": "旧需求"},
            {"role": "assistant", "content": "旧回答"},
        ],
        {"summary_lines": [], "omitted_summary_lines": 0},
    )
    monkeypatch.setenv("MINI_AGENT_SESSION_DIR", str(session_root))
    monkeypatch.setattr(
        Settings,
        "from_env",
        Mock(return_value=Settings(api_key="test-key")),
    )
    monkeypatch.setattr(cli_module, "DeepSeekClient", Mock(return_value=object()))
    restore_session = Mock()
    monkeypatch.setattr(cli_module.CodingAgent, "restore_session", restore_session)
    monkeypatch.setattr("builtins.input", Mock(return_value="/exit"))

    exit_code = main(["--workspace", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    restore_session.assert_called_once_with(saved.messages, saved.context_state)
    assert f"当前会话：{saved.session_id}" in output
    assert "已恢复 1 轮历史对话" in output


def test_interactive_session_management_commands(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    store = SessionStore(tmp_path, root=session_root)
    first = store.create()
    store.save(
        first.session_id,
        [
            {"role": "user", "content": "第一个会话"},
            {"role": "assistant", "content": "完成"},
        ],
        {"summary_lines": [], "omitted_summary_lines": 0},
    )
    second = store.create()
    monkeypatch.setenv("MINI_AGENT_SESSION_DIR", str(session_root))
    monkeypatch.setattr(
        Settings,
        "from_env",
        Mock(return_value=Settings(api_key="test-key")),
    )
    monkeypatch.setattr(cli_module, "DeepSeekClient", Mock(return_value=object()))
    monkeypatch.setattr(
        cli_module.CodingAgent,
        "export_session",
        Mock(
            return_value={
                "messages": [],
                "context": {
                    "summary_lines": [],
                    "omitted_summary_lines": 0,
                },
            }
        ),
    )
    user_input = Mock(
        side_effect=[
            "/sessions",
            f"/switch {first.session_id}",
            "/new",
            f"/delete {first.session_id}",
            "y",
            "/exit",
        ]
    )
    monkeypatch.setattr("builtins.input", user_input)

    exit_code = main(["--workspace", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "当前工作目录的会话" in output
    assert f"已切换到会话：{first.session_id}" in output
    assert "已创建新会话" in output
    assert f"已删除会话：{first.session_id}" in output
    with pytest.raises(SessionError, match="不存在"):
        SessionStore(tmp_path, root=session_root).load(first.session_id)
    assert SessionStore(tmp_path, root=session_root).load(second.session_id)


def test_deleting_current_session_creates_replacement(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    store = SessionStore(tmp_path, root=session_root)
    current = store.create()
    monkeypatch.setenv("MINI_AGENT_SESSION_DIR", str(session_root))
    monkeypatch.setattr(
        Settings,
        "from_env",
        Mock(return_value=Settings(api_key="test-key")),
    )
    monkeypatch.setattr(cli_module, "DeepSeekClient", Mock(return_value=object()))
    monkeypatch.setattr(
        cli_module.CodingAgent,
        "export_session",
        Mock(
            return_value={
                "messages": [],
                "context": {
                    "summary_lines": [],
                    "omitted_summary_lines": 0,
                },
            }
        ),
    )
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=[f"/delete {current.session_id}", "y", "/exit"]),
    )

    exit_code = main(["--workspace", str(tmp_path)])

    output = capsys.readouterr().out
    records = SessionStore(tmp_path, root=session_root).list_sessions()
    assert exit_code == 0
    assert "当前会话已删除，已创建新会话" in output
    assert len(records) == 1
    assert records[0].session_id != current.session_id
