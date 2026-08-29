"""Tests for deterministic conversation-history compaction."""

import json

import pytest

from mini_agent.context import ContextLimitError, ContextManager


def _round(index: int, output_size: int = 600) -> list[dict]:
    call_id = f"call-{index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": f"file-{index}.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {"success": True, "content": "x" * output_size}
            ),
        },
    ]


def test_context_is_unchanged_when_within_budget() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *_round(1, output_size=10),
    ]
    manager = ContextManager(max_chars=10_000)
    manager.start("system")

    result = manager.prepare(messages)

    assert result.messages == messages
    assert result.removed_rounds == 0


def test_context_removes_only_complete_old_rounds() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *_round(1),
        *_round(2),
        *_round(3),
    ]
    manager = ContextManager(max_chars=1_400, keep_recent_rounds=1)
    manager.start("system")

    result = manager.prepare(messages)

    assert result.removed_rounds == 2
    assert result.messages[0]["role"] == "system"
    assert "较早工具历史摘要" in result.messages[0]["content"]
    assert "file-1.py" in result.messages[0]["content"]
    assert "file-2.py" in result.messages[0]["content"]
    assert [message["role"] for message in result.messages[2:]] == [
        "assistant",
        "tool",
    ]
    assert result.messages[-1]["tool_call_id"] == "call-3"
    assert result.estimated_chars <= 1_400


def test_context_summary_records_failed_tool_status() -> None:
    failed_round = _round(1)
    failed_round[1]["content"] = json.dumps(
        {"success": False, "content": "missing"}
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *failed_round,
        *_round(2),
    ]
    manager = ContextManager(max_chars=1_100, keep_recent_rounds=1)
    manager.start("system")

    result = manager.prepare(messages)

    assert "失败" in result.messages[0]["content"]


def test_context_rejects_oversized_required_messages() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "x" * 1_000},
    ]
    manager = ContextManager(max_chars=100)
    manager.start("system")

    with pytest.raises(ContextLimitError, match="超过上下文预算"):
        manager.prepare(messages)


def test_context_rejects_orphan_tool_message() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "tool", "tool_call_id": "orphan", "content": "result"},
    ]
    manager = ContextManager(max_chars=50)
    manager.start("system")

    with pytest.raises(ContextLimitError, match="结构无效"):
        manager.prepare(messages)


def test_context_compacts_complete_old_conversation_turn() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "创建飞机大战"},
        *_round(1, output_size=500),
        {"role": "assistant", "content": "第一轮已完成" + "x" * 400},
        {"role": "user", "content": "增加暂停功能"},
    ]
    manager = ContextManager(max_chars=700, keep_recent_rounds=1)
    manager.start("system")

    result = manager.prepare(messages)

    assert result.removed_turns == 1
    assert result.removed_rounds == 0
    assert "创建飞机大战" in result.messages[0]["content"]
    assert result.messages[-1] == {
        "role": "user",
        "content": "增加暂停功能",
    }
    assert result.estimated_chars <= 700
