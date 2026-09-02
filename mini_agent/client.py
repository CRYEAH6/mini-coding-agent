"""OpenAI-compatible LLM client used by the agent loop."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from openai import OpenAI

from mini_agent.config import Settings


TextHandler = Callable[[str], None]


@dataclass(frozen=True)
class ToolFunction:
    """A fully assembled streamed function call."""

    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCall:
    """A fully assembled streamed tool call."""

    id: str
    type: str
    function: ToolFunction


@dataclass(frozen=True)
class ModelMessage:
    """Assistant message assembled from streamed response chunks."""

    content: Optional[str]
    tool_calls: Sequence[ToolCall]
    reasoning_content: Optional[str] = None


@dataclass
class _ToolCallBuffer:
    """Mutable state used while tool-call JSON arrives in fragments."""

    call_id: str = ""
    call_type: str = "function"
    name: str = ""
    arguments: str = ""


class LLMClient:
    """Send requests through an OpenAI-compatible chat-completions API."""

    def __init__(self, settings: Settings, client: Optional[Any] = None) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def create_message(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        on_text: Optional[TextHandler] = None,
        model: Optional[str] = None,
    ) -> Any:
        """Return one assistant message, optionally streaming text chunks."""
        request: dict[str, Any] = {
            "model": model or self._settings.model,
            "messages": list(messages),
        }
        if tools:
            request["tools"] = list(tools)

        if on_text is not None:
            request["stream"] = True
            return _collect_stream(
                self._client.chat.completions.create(**request),
                on_text,
            )

        response = self._client.chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("模型 API 返回了空的 choices。")
        return response.choices[0].message


def _collect_stream(stream: Any, on_text: TextHandler) -> ModelMessage:
    """Assemble content, reasoning, and fragmented tool calls from SSE chunks."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_buffers: dict[int, _ToolCallBuffer] = {}
    received_delta = False

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        received_delta = True

        content = getattr(delta, "content", None)
        if content:
            content_parts.append(content)
            on_text(content)

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)

        for tool_delta in getattr(delta, "tool_calls", None) or []:
            buffer = tool_buffers.setdefault(tool_delta.index, _ToolCallBuffer())
            if tool_delta.id:
                buffer.call_id += tool_delta.id
            if tool_delta.type:
                buffer.call_type = tool_delta.type
            function = tool_delta.function
            if function is not None:
                if function.name:
                    buffer.name += function.name
                if function.arguments:
                    buffer.arguments += function.arguments

    if not received_delta:
        raise RuntimeError("模型 API 返回了空的流式响应。")

    tool_calls = []
    for index in sorted(tool_buffers):
        buffer = tool_buffers[index]
        if not buffer.call_id or not buffer.name:
            raise RuntimeError("模型 API 返回了不完整的流式工具调用。")
        tool_calls.append(
            ToolCall(
                id=buffer.call_id,
                type=buffer.call_type,
                function=ToolFunction(
                    name=buffer.name,
                    arguments=buffer.arguments,
                ),
            )
        )

    content = "".join(content_parts) or None
    reasoning_content = "".join(reasoning_parts) or None
    return ModelMessage(content, tool_calls, reasoning_content)
