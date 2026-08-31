"""Tests for LLM-based semantic history summaries."""

from types import SimpleNamespace

import pytest

from mini_agent.summary import (
    MAX_MESSAGE_CONTENT_CHARS,
    MAX_SUMMARY_SOURCE_CHARS,
    SemanticSummarizer,
    SemanticSummaryError,
    _serialize_history,
)


class FakeClient:
    def __init__(self, message) -> None:
        self.message = message
        self.calls = []

    def create_message(self, messages, tools=None, on_text=None, model=None):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "on_text": on_text,
                "model": model,
            }
        )
        return self.message


def _message(content="摘要内容", tool_calls=()):
    return SimpleNamespace(content=content, tool_calls=list(tool_calls))


def test_semantic_summarizer_merges_history_without_tools() -> None:
    client = FakeClient(_message("- 用户目标：修复测试\n- 验证：已通过"))
    events = []
    summarizer = SemanticSummarizer(
        client,
        "summary-model",
        event_handler=events.append,
    )

    summary = summarizer.summarize(
        "- 已有摘要：创建项目",
        [
            {"role": "user", "content": "修复测试"},
            {"role": "assistant", "content": "已经完成"},
        ],
    )

    assert "修复测试" in summary
    assert client.calls[0]["model"] == "summary-model"
    assert client.calls[0]["tools"] is None
    assert client.calls[0]["on_text"] is None
    assert "已有摘要：创建项目" in client.calls[0]["messages"][1]["content"]
    assert events == [
        "[上下文] 历史超限，正在使用 summary-model 生成语义摘要..."
    ]


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (_message(""), "空内容"),
        (_message("x", tool_calls=[object()]), "工具调用"),
        (_message("x" * 101), "超过"),
    ],
)
def test_semantic_summarizer_rejects_invalid_output(message, error) -> None:
    summarizer = SemanticSummarizer(
        FakeClient(message),
        "summary-model",
        max_summary_chars=100,
    )

    with pytest.raises(SemanticSummaryError, match=error):
        summarizer.summarize("", [{"role": "user", "content": "task"}])


def test_semantic_summarizer_redacts_secrets_from_source() -> None:
    client = FakeClient(_message("安全摘要"))
    summarizer = SemanticSummarizer(client, "summary-model")

    summarizer.summarize(
        "",
        [
            {
                "role": "user",
                "content": "DEEPSEEK_API_KEY=sk-private123456",
            }
        ],
    )

    prompt = client.calls[0]["messages"][1]["content"]
    assert "sk-private" not in prompt
    assert "[REDACTED]" in prompt


def test_summary_source_and_each_message_are_bounded() -> None:
    source = _serialize_history(
        [
            {"role": "tool", "content": "x" * 20_000}
            for _ in range(20)
        ]
    )

    assert len(source) <= MAX_SUMMARY_SOURCE_CHARS
    assert "x" * (MAX_MESSAGE_CONTENT_CHARS + 1) not in source
    assert "已截断" in source
