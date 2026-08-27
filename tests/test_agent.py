"""Tests for the core model-tool loop."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from mini_agent.agent import CodingAgent, StepLimitError, ToolLoopError
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

    def create_message(self, messages, tools=None):
        self.requests.append(deepcopy(messages))
        return self._responses.pop(0)


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
    assert "调用工具：write_file" in events

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

    assert "工具结果：失败" in events
    assert any(event.startswith("工具错误：路径不存在") for event in events)


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
    assert any(event.startswith("上下文已压缩") for event in events)
