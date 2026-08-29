"""Tests for the core model-tool loop."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from mini_agent.agent import (
    CodingAgent,
    StepLimitError,
    ToolLoopError,
    _describe_tool_call,
)
from mini_agent.tools import ToolRegistry


def _tool_call(name: str, arguments: str, call_id: str = "call-1") -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _message(
    *,
    content: str = "",
    tool_calls: Sequence[Any] = (),
    reasoning_content: str = "",
) -> Any:
    return SimpleNamespace(
        content=content,
        tool_calls=list(tool_calls),
        reasoning_content=reasoning_content,
    )


class FakeClient:
    """Return fixed model messages and record request histories."""

    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[Sequence[Mapping[str, Any]]] = []

    def create_message(self, messages, tools=None, on_text=None):
        self.requests.append(deepcopy(messages))
        response = self._responses.pop(0)
        if on_text is not None and response.content:
            on_text(response.content)
        return response


def test_agent_executes_tool_and_returns_final_response(tmp_path: Path) -> None:
    first = _message(
        tool_calls=[
            _tool_call(
                "write_file",
                '{"path": "answer.txt", "content": "42"}',
            )
        ],
        reasoning_content="I should create the requested file.",
    )
    client = FakeClient([first, _message(content="文件已经创建。")])
    events = []
    agent = CodingAgent(
        client,
        ToolRegistry(tmp_path),
        event_handler=events.append,
    )

    result = agent.run("创建 answer.txt")

    assert result.content == "文件已经创建。"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "42"
    assert any(event.startswith("[工具] write_file") for event in events)

    second_request = client.requests[1]
    assert [message["role"] for message in second_request] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_request[2]["reasoning_content"] == (
        "I should create the requested file."
    )
    assert second_request[3]["tool_call_id"] == "call-1"


def test_agent_stops_after_final_response_without_tools(tmp_path: Path) -> None:
    client = FakeClient([_message(content="无需修改。")])

    result = CodingAgent(client, ToolRegistry(tmp_path)).run("检查项目")

    assert result.content == "无需修改。"
    assert result.steps == 1
    assert result.tool_calls == 0


def test_agent_keeps_history_across_interactive_turns(tmp_path: Path) -> None:
    client = FakeClient(
        [
            _message(content="飞机大战已经创建。"),
            _message(content="暂停功能已经补充。"),
        ]
    )
    agent = CodingAgent(client, ToolRegistry(tmp_path))

    first = agent.run_turn("创建飞机大战")
    second = agent.run_turn("再增加暂停功能")

    assert first.content == "飞机大战已经创建。"
    assert second.content == "暂停功能已经补充。"
    assert [message["role"] for message in client.requests[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert client.requests[1][-2]["content"] == "飞机大战已经创建。"
    assert client.requests[1][-1]["content"] == "再增加暂停功能"


def test_agent_forwards_streamed_text_to_handler(tmp_path: Path) -> None:
    chunks = []
    agent = CodingAgent(
        FakeClient([_message(content="实时回复")]),
        ToolRegistry(tmp_path),
        text_handler=chunks.append,
    )

    result = agent.run_turn("介绍项目")

    assert result.content == "实时回复"
    assert chunks == ["实时回复"]


def test_reset_session_discards_chat_history(tmp_path: Path) -> None:
    client = FakeClient(
        [_message(content="第一轮"), _message(content="新会话")]
    )
    agent = CodingAgent(client, ToolRegistry(tmp_path))
    agent.run_turn("旧任务")

    agent.reset_session()
    agent.run_turn("新任务")

    assert [message["role"] for message in client.requests[1]] == [
        "system",
        "user",
    ]
    assert client.requests[1][-1]["content"] == "新任务"


def test_agent_rejects_empty_task(tmp_path: Path) -> None:
    agent = CodingAgent(FakeClient([]), ToolRegistry(tmp_path))

    with pytest.raises(ValueError, match="不能为空"):
        agent.run("   ")


def test_agent_rejects_empty_model_response(tmp_path: Path) -> None:
    agent = CodingAgent(FakeClient([_message()]), ToolRegistry(tmp_path))

    with pytest.raises(RuntimeError, match="既没有"):
        agent.run("检查项目")


def test_agent_enforces_step_limit(tmp_path: Path) -> None:
    response = _message(tool_calls=[_tool_call("list_files", "{}")])
    agent = CodingAgent(
        FakeClient([response]),
        ToolRegistry(tmp_path),
        max_steps=1,
    )

    with pytest.raises(StepLimitError, match="最大步骤数 1"):
        agent.run("持续检查")


def test_agent_emits_compact_tool_error(tmp_path: Path) -> None:
    first = _message(
        tool_calls=[_tool_call("read_file", '{"path": "missing.txt"}')]
    )
    events = []
    agent = CodingAgent(
        FakeClient([first, _message(content="无法读取文件。")]),
        ToolRegistry(tmp_path),
        event_handler=events.append,
    )

    agent.run("读取文件")

    assert "[结果] 失败" in events
    assert any(event.startswith("[错误] 路径不存在") for event in events)


def test_agent_stops_repeated_identical_tool_calls(tmp_path: Path) -> None:
    responses = [
        _message(tool_calls=[_tool_call("list_files", "{}", f"call-{index}")])
        for index in range(4)
    ]
    client = FakeClient(responses)
    agent = CodingAgent(client, ToolRegistry(tmp_path))

    with pytest.raises(ToolLoopError, match="重复"):
        agent.run("重复查看目录")

    third_request_result = client.requests[3][-1]["content"]
    assert "第 3 次相同工具调用" in third_request_result


def test_agent_stops_after_four_consecutive_failures(tmp_path: Path) -> None:
    responses = [
        _message(
            tool_calls=[
                _tool_call(
                    "read_file",
                    f'{{"path": "missing-{index}.txt"}}',
                    f"call-{index}",
                )
            ]
        )
        for index in range(4)
    ]
    agent = CodingAgent(FakeClient(responses), ToolRegistry(tmp_path))

    with pytest.raises(ToolLoopError, match="连续失败 4 次"):
        agent.run("持续读取不存在的文件")


def test_successful_tool_resets_failure_count(tmp_path: Path) -> None:
    failing_calls = [
        _message(
            tool_calls=[
                _tool_call(
                    "read_file",
                    f'{{"path": "missing-{index}.txt"}}',
                    f"call-{index}",
                )
            ]
        )
        for index in range(3)
    ]
    success = _message(
        tool_calls=[_tool_call("list_files", "{}", "call-success")]
    )
    client = FakeClient(
        [*failing_calls, success, _message(content="已调整方案。")]
    )

    result = CodingAgent(client, ToolRegistry(tmp_path)).run("先失败再恢复")

    assert result.content == "已调整方案。"


def test_agent_compacts_old_tool_rounds(tmp_path: Path) -> None:
    responses = []
    for index in range(3):
        path = f"large-{index}.txt"
        (tmp_path / path).write_text("x" * 700, encoding="utf-8")
        responses.append(
            _message(
                tool_calls=[
                    _tool_call(
                        "read_file",
                        f'{{"path": "{path}"}}',
                        f"call-{index}",
                    )
                ]
            )
        )
    responses.append(_message(content="读取完成。"))
    events = []
    agent = CodingAgent(
        FakeClient(responses),
        ToolRegistry(tmp_path),
        max_context_chars=2_500,
        event_handler=events.append,
    )

    result = agent.run("依次读取三个文件")

    assert result.content == "读取完成。"
    assert result.compacted_rounds >= 1
    assert any(event.startswith("[上下文] 已压缩") for event in events)


def test_tool_description_does_not_echo_sensitive_content() -> None:
    command_description = _describe_tool_call(
        "run_command",
        '{"command": "python3 deploy.py --token private-value"}',
    )
    write_description = _describe_tool_call(
        "write_file",
        '{"path": "config.py", "content": "private-value"}',
    )

    assert "program=python3" in command_description
    assert "private-value" not in command_description
    assert "path=config.py" in write_description
    assert "content_chars=13" in write_description
    assert "private-value" not in write_description


def test_agent_executes_multiple_tool_calls_from_one_response(
    tmp_path: Path,
) -> None:
    first = _message(
        tool_calls=[
            _tool_call(
                "write_file",
                '{"path": "first.txt", "content": "first"}',
                "call-first",
            ),
            _tool_call(
                "write_file",
                '{"path": "second.txt", "content": "second"}',
                "call-second",
            ),
        ]
    )
    client = FakeClient([first, _message(content="两个文件已创建。")])

    result = CodingAgent(client, ToolRegistry(tmp_path)).run("创建两个文件")

    assert result.tool_calls == 2
    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "first"
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "second"
    assert [message["role"] for message in client.requests[1][-3:]] == [
        "assistant",
        "tool",
        "tool",
    ]


def test_agent_recovers_after_invalid_tool_json(tmp_path: Path) -> None:
    first = _message(
        tool_calls=[_tool_call("read_file", "not-json", "call-invalid")]
    )
    client = FakeClient([first, _message(content="参数错误已处理。")])

    result = CodingAgent(client, ToolRegistry(tmp_path)).run("读取文件")

    assert result.content == "参数错误已处理。"
    tool_result = client.requests[1][-1]["content"]
    assert "工具参数不是有效 JSON" in tool_result
