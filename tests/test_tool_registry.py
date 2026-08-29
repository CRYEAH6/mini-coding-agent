"""Tests for tool schema exposure and dispatch."""

import json
from pathlib import Path
from unittest.mock import Mock

from mini_agent.tools.registry import ToolRegistry


def test_registry_exposes_all_tool_definitions(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    names = {item["function"]["name"] for item in registry.definitions}

    assert names == {
        "list_files",
        "read_file",
        "write_file",
        "replace_in_file",
        "run_command",
    }


def test_registry_executes_json_arguments(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "write_file",
        json.dumps({"path": "hello.txt", "content": "hello"}),
    )

    assert result.success
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"


def test_registry_rejects_unknown_tool(tmp_path: Path) -> None:
    result = ToolRegistry(tmp_path).execute("unknown", "{}")

    assert not result.success
    assert "未知工具" in result.content


def test_registry_rejects_invalid_json(tmp_path: Path) -> None:
    result = ToolRegistry(tmp_path).execute("list_files", "not-json")

    assert not result.success
    assert "有效 JSON" in result.content


def test_registry_rejects_unexpected_arguments(tmp_path: Path) -> None:
    result = ToolRegistry(tmp_path).execute("list_files", '{"unexpected": true}')

    assert not result.success
    assert "参数不正确" in result.content


def test_tool_result_serializes_as_unicode_json(tmp_path: Path) -> None:
    result = ToolRegistry(tmp_path).execute("list_files", "{}")

    payload = json.loads(result.to_json())

    assert payload == {"success": True, "content": "目录为空。"}


def test_registry_rejects_wrong_argument_type(tmp_path: Path) -> None:
    result = ToolRegistry(tmp_path).execute(
        "write_file",
        '{"path": "example.txt", "content": 42}',
    )

    assert not result.success
    assert "content 必须是字符串" in result.content


def test_registry_executes_sensitive_command_after_approval(tmp_path: Path) -> None:
    approval_handler = Mock(return_value=True)
    registry = ToolRegistry(tmp_path, approval_handler=approval_handler)

    result = registry.execute("run_command", '{"command": "curl --version"}')

    assert result.success
    request = approval_handler.call_args.args[0]
    assert request.tool_name == "run_command"
    assert request.details == "curl --version"
    assert "外部网络" in request.reason


def test_registry_does_not_execute_sensitive_command_when_denied(
    tmp_path: Path,
) -> None:
    approval_handler = Mock(return_value=False)
    registry = ToolRegistry(tmp_path, approval_handler=approval_handler)

    result = registry.execute("run_command", '{"command": "curl --version"}')

    assert not result.success
    assert "用户拒绝" in result.content
    approval_handler.assert_called_once()


def test_registry_reports_missing_approval_entry(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    result = registry.execute("run_command", '{"command": "curl --version"}')

    assert not result.success
    assert "未提供确认入口" in result.content


def test_registry_runs_normal_command_without_approval(tmp_path: Path) -> None:
    approval_handler = Mock(return_value=False)
    registry = ToolRegistry(tmp_path, approval_handler=approval_handler)

    result = registry.execute(
        "run_command",
        '{"command": "python3 -c \\"print(42)\\""}',
    )

    assert result.success
    assert "42" in result.content
    approval_handler.assert_not_called()


def test_dangerous_override_still_requires_approval(tmp_path: Path) -> None:
    approval_handler = Mock(return_value=False)
    registry = ToolRegistry(
        tmp_path,
        allow_dangerous_commands=True,
        approval_handler=approval_handler,
    )

    result = registry.execute("run_command", '{"command": "rm old.txt"}')

    assert not result.success
    assert "用户拒绝" in result.content
    approval_handler.assert_called_once()
