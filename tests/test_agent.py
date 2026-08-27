"""Tests for the core model-tool loop."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from mini_agent.agent import CodingAgent, StepLimitError
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
