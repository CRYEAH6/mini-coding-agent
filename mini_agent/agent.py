"""Core model-tool loop for the coding agent."""

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Optional

from mini_agent.client import DeepSeekClient
from mini_agent.context import ContextManager
from mini_agent.tools import ToolRegistry, ToolResult


DEFAULT_MAX_STEPS = 20
EVENT_PREVIEW_CHARS = 500
REPEAT_WARNING_THRESHOLD = 3
MAX_CONSECUTIVE_FAILURES = 4

SYSTEM_PROMPT = """你是一个在本地项目中工作的编程智能体。
请先理解任务和现有代码，再选择合适的工具完成修改。
文件路径必须使用相对工作目录的路径。修改后应尽量运行相关测试。
工具失败时请阅读错误信息并调整方案，不要盲目重复相同调用。
完成任务后停止调用工具，用简洁文字总结修改和验证结果。
不要读取、输出或写入 API Key 等敏感凭据。
"""


class StepLimitError(RuntimeError):
    """Raised when the agent exceeds its configured model-step limit."""


class ToolLoopError(RuntimeError):
    """Raised when repeated or failing tool calls form an invalid loop."""


@dataclass(frozen=True)
class AgentResult:
    """Final text and execution statistics for one task."""

    content: str
    steps: int
    tool_calls: int
    compacted_rounds: int


EventHandler = Callable[[str], None]


class CodingAgent:
    """Coordinate model responses and local tool execution."""

    def __init__(
        self,
        client: DeepSeekClient,
        tools: ToolRegistry,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_context_chars: int = 200_000,
        event_handler: Optional[EventHandler] = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0。")
        self._client = client
        self._tools = tools
        self._max_steps = max_steps
        self._context = ContextManager(max_context_chars)
        self._emit = event_handler or (lambda _: None)

    def run(self, task: str) -> AgentResult:
        """Run one task until the model returns a final response."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("任务内容不能为空。")

        messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.strip()},
        ]
        tool_call_count = 0
        compacted_round_count = 0
        consecutive_failures = 0
        previous_fingerprint: Optional[str] = None
        repeated_call_count = 0

        self._context.start(SYSTEM_PROMPT)
        for step in range(1, self._max_steps + 1):
            compacted = self._context.prepare(messages)
            messages = compacted.messages
            if compacted.removed_rounds:
                compacted_round_count += compacted.removed_rounds
                self._emit(
                    "上下文已压缩：移除 "
                    f"{compacted.removed_rounds} 个较早工具轮次。"
                )
            self._emit(f"[{step}/{self._max_steps}] 正在请求模型...")
            message = self._client.create_message(messages, self._tools.definitions)
            assistant_message = _serialize_assistant_message(message)
            messages.append(assistant_message)

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                content = getattr(message, "content", None)
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        "模型既没有返回工具调用，也没有返回最终文字。"
                    )
                return AgentResult(
                    content.strip(),
                    step,
                    tool_call_count,
                    compacted_round_count,
                )

            for tool_call in tool_calls:
                name = tool_call.function.name
                self._emit(f"调用工具：{name}")
                fingerprint = _tool_fingerprint(
                    name,
                    tool_call.function.arguments,
                )
                if fingerprint == previous_fingerprint:
                    repeated_call_count += 1
                else:
                    previous_fingerprint = fingerprint
                    repeated_call_count = 1

                terminate_reason = None
                if repeated_call_count > REPEAT_WARNING_THRESHOLD:
                    result = ToolResult(
                        False,
                        "检测到完全相同的工具调用持续重复，任务已终止。",
                    )
                    terminate_reason = "模型持续重复完全相同的工具调用。"
                elif repeated_call_count == REPEAT_WARNING_THRESHOLD:
                    result = ToolResult(
                        False,
                        "检测到第 3 次相同工具调用，本次未执行。"
                        "请调整参数或方案。",
                    )
                else:
                    result = self._tools.execute(
                        name,
                        tool_call.function.arguments,
                    )

                if result.success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures == MAX_CONSECUTIVE_FAILURES - 1:
                        result = ToolResult(
                            False,
                            f"{result.content}\n警告：工具已连续失败 "
                            f"{consecutive_failures} 次，请调整方案。",
                        )
                    elif consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        terminate_reason = (
                            f"工具连续失败 {consecutive_failures} 次。"
                        )

                status = "成功" if result.success else "失败"
                self._emit(f"工具结果：{status}")
                if not result.success:
                    self._emit(f"工具错误：{_preview(result.content)}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.to_json(),
                    }
                )
                tool_call_count += 1
                if terminate_reason is not None:
                    raise ToolLoopError(terminate_reason)

        raise StepLimitError(
            f"Agent 已达到最大步骤数 {self._max_steps}，任务被终止。"
        )


def _serialize_assistant_message(message: Any) -> Mapping[str, Any]:
    """Preserve the fields required for the next DeepSeek request."""
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None),
    }
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None:
        payload["reasoning_content"] = reasoning_content

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]
    return payload


def _preview(content: str) -> str:
    """Keep terminal error output informative but compact."""
    if len(content) <= EVENT_PREVIEW_CHARS:
        return content
    return f"{content[:EVENT_PREVIEW_CHARS]}..."


def _tool_fingerprint(name: str, arguments: str) -> str:
    """Create a stable identity for semantically equal tool calls."""
    try:
        parsed = json.loads(arguments)
        normalized = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (json.JSONDecodeError, TypeError):
        normalized = arguments
    return f"{name}:{normalized}"
