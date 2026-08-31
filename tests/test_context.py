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


def test_context_uses_semantic_summary_only_after_budget_is_exceeded() -> None:
    calls = []

    def summarize(existing, removed):
        calls.append((existing, removed))
        return "- 用户目标：创建飞机大战\n- 结果：第一轮已完成"

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "创建飞机大战"},
        *_round(1, output_size=500),
        {"role": "assistant", "content": "第一轮已完成" + "x" * 400},
        {"role": "user", "content": "增加暂停功能"},
    ]
    manager = ContextManager(
        max_chars=700,
        keep_recent_rounds=1,
        summary_generator=summarize,
    )
    manager.start("system")

    result = manager.prepare(messages)

    assert result.semantic_summaries == 1
    assert result.summary_fallbacks == 0
    assert len(calls) == 1
    assert calls[0][0] == ""
    assert calls[0][1][0]["content"] == "创建飞机大战"
    assert "较早对话语义摘要" in result.messages[0]["content"]
    assert "第一轮已完成" in result.messages[0]["content"]
    assert manager.export_state()["semantic_summary"].startswith("- 用户目标")


def test_context_does_not_call_summary_model_within_budget() -> None:
    calls = []
    manager = ContextManager(
        max_chars=10_000,
        summary_generator=lambda existing, removed: calls.append(removed),
    )
    manager.start("system")

    manager.prepare(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "短任务"},
        ]
    )

    assert calls == []


def test_context_falls_back_when_semantic_summary_fails() -> None:
    def fail_summary(existing, removed):
        raise RuntimeError("temporary API failure")

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "创建飞机大战"},
        *_round(1, output_size=500),
        {"role": "assistant", "content": "第一轮已完成" + "x" * 400},
        {"role": "user", "content": "增加暂停功能"},
    ]
    manager = ContextManager(
        max_chars=700,
        keep_recent_rounds=1,
        summary_generator=fail_summary,
    )
    manager.start("system")

    result = manager.prepare(messages)

    assert result.semantic_summaries == 0
    assert result.summary_fallbacks == 1
    assert "较早工具历史摘要" in result.messages[0]["content"]
    assert "创建飞机大战" in result.messages[0]["content"]


def test_context_merges_restored_summary_with_newly_removed_history() -> None:
    received = []

    def summarize(existing, removed):
        received.append((existing, removed))
        return "旧项目已创建；新功能已完成"

    manager = ContextManager(
        max_chars=450,
        summary_generator=summarize,
    )
    manager.restore(
        "system",
        {
            "semantic_summary": "旧项目已创建",
            "summary_lines": [],
            "omitted_summary_lines": 0,
        },
    )
    messages = [
        {"role": "system", "content": manager.system_content},
        {"role": "user", "content": "增加功能"},
        {"role": "assistant", "content": "完成" + "x" * 400},
        {"role": "user", "content": "继续修改"},
    ]

    result = manager.prepare(messages)

    assert result.semantic_summaries == 1
    assert received[0][0] == "旧项目已创建"
    assert manager.export_state()["semantic_summary"] == (
        "旧项目已创建；新功能已完成"
    )


def test_context_compaction_state_can_be_restored() -> None:
    manager = ContextManager()
    manager.restore(
        "system",
        {
            "summary_lines": ["- 用户要求：旧任务"],
            "omitted_summary_lines": 2,
        },
    )
    state = manager.export_state()

    restored = ContextManager()
    restored.restore("new system", state)

    assert "new system" in restored.system_content
    assert "更早的 2 条工具记录已省略" in restored.system_content
    assert "旧任务" in restored.system_content


def test_context_rejects_invalid_persisted_state() -> None:
    manager = ContextManager()

    with pytest.raises(ContextLimitError, match="摘要格式"):
        manager.restore(
            "system",
            {"summary_lines": [42], "omitted_summary_lines": 0},
        )

    with pytest.raises(ContextLimitError, match="语义摘要格式"):
        manager.restore(
            "system",
            {
                "semantic_summary": 42,
                "summary_lines": [],
                "omitted_summary_lines": 0,
            },
        )


def test_runtime_context_is_injected_but_not_persisted() -> None:
    manager = ContextManager(10_000)
    manager.start("system")
    manager.set_runtime_context("- [用户偏好] README 使用中文")

    prepared = manager.prepare(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "编写 README"},
        ]
    )

    assert "README 使用中文" in prepared.messages[0]["content"]
    assert "runtime" not in manager.export_state()
