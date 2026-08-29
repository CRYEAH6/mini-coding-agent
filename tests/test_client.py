"""Tests for the DeepSeek API client wrapper."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mini_agent.client import DeepSeekClient
from mini_agent.config import Settings


def _settings() -> Settings:
    return Settings(api_key="test-key", model="test-model")


def test_create_message_sends_messages_and_tools() -> None:
    message = SimpleNamespace(content="done")
    completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    client = DeepSeekClient(_settings(), client=fake_client)
    messages = [{"role": "user", "content": "Fix the bug"}]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    result = client.create_message(messages, tools)

    assert result is message
    completion.assert_called_once_with(
        model="test-model",
        messages=messages,
        tools=tools,
    )


def test_create_message_omits_empty_tools() -> None:
    message = SimpleNamespace(content="done")
    completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    client = DeepSeekClient(_settings(), client=fake_client)
    messages = [{"role": "user", "content": "Hello"}]

    client.create_message(messages)

    completion.assert_called_once_with(model="test-model", messages=messages)


def test_create_message_rejects_empty_choices() -> None:
    completion = Mock(return_value=SimpleNamespace(choices=[]))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    client = DeepSeekClient(_settings(), client=fake_client)

    with pytest.raises(RuntimeError, match="choices"):
        client.create_message([{"role": "user", "content": "Hello"}])


def _chunk(*, content=None, reasoning=None, tool_calls=()):
    delta = SimpleNamespace(content=content, tool_calls=list(tool_calls))
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_delta(index, *, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        type="function" if call_id else None,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_create_message_streams_text_and_assembles_tool_calls() -> None:
    stream = [
        _chunk(reasoning="先分析", content="正在", tool_calls=[
            _tool_delta(
                0,
                call_id="call-1",
                name="write_",
                arguments='{"path":"answer',
            )
        ]),
        _chunk(reasoning="再执行", content="处理", tool_calls=[
            _tool_delta(
                0,
                name="file",
                arguments='.txt","content":"42"}',
            )
        ]),
        SimpleNamespace(choices=[]),
    ]
    completion = Mock(return_value=iter(stream))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    client = DeepSeekClient(_settings(), client=fake_client)
    text_chunks = []

    message = client.create_message(
        [{"role": "user", "content": "创建文件"}],
        on_text=text_chunks.append,
    )

    assert message.content == "正在处理"
    assert message.reasoning_content == "先分析再执行"
    assert text_chunks == ["正在", "处理"]
    assert message.tool_calls[0].id == "call-1"
    assert message.tool_calls[0].function.name == "write_file"
    assert message.tool_calls[0].function.arguments == (
        '{"path":"answer.txt","content":"42"}'
    )
    completion.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "创建文件"}],
        stream=True,
    )


def test_create_message_rejects_empty_stream() -> None:
    completion = Mock(return_value=iter([SimpleNamespace(choices=[])]))
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    client = DeepSeekClient(_settings(), client=fake_client)

    with pytest.raises(RuntimeError, match="空的流式响应"):
        client.create_message(
            [{"role": "user", "content": "Hello"}],
            on_text=lambda _: None,
        )
