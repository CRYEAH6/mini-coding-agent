"""Deterministic conversation-history compaction."""

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence


DEFAULT_MAX_CONTEXT_CHARS = 200_000
RECENT_ROUNDS_TO_KEEP = 2
MAX_SUMMARY_LINES = 30
ARGUMENT_PREVIEW_CHARS = 180


class ContextLimitError(RuntimeError):
    """Raised when required recent context exceeds the configured budget."""


@dataclass(frozen=True)
class CompactionResult:
    """Messages prepared for a request and compaction statistics."""

    messages: list[Mapping[str, Any]]
    removed_rounds: int
    estimated_chars: int


class ContextManager:
    """Compact old complete tool rounds without splitting call pairs."""

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        keep_recent_rounds: int = RECENT_ROUNDS_TO_KEEP,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须大于 0。")
        if keep_recent_rounds < 0:
            raise ValueError("keep_recent_rounds 不能小于 0。")
        self._max_chars = max_chars
        self._keep_recent_rounds = keep_recent_rounds
        self._base_system_prompt = ""
        self._summary_lines: list[str] = []
        self._omitted_summary_lines = 0

    def start(self, system_prompt: str) -> None:
        """Reset compaction state for a new user task."""
        self._base_system_prompt = system_prompt
        self._summary_lines.clear()
        self._omitted_summary_lines = 0

    def prepare(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> CompactionResult:
        """Return messages that fit the configured approximate budget."""
        prepared = [dict(message) for message in messages]
        estimated = _estimate_chars(prepared)
        if estimated <= self._max_chars:
            return CompactionResult(prepared, 0, estimated)
        if len(prepared) < 2:
            raise ContextLimitError("上下文缺少必要的 system 或 user 消息。")

        prefix = prepared[:2]
        rounds = _split_rounds(prepared[2:])
        removed = 0
        while len(rounds) > self._keep_recent_rounds:
            oldest = rounds.pop(0)
            self._record_round(oldest)
            removed += 1
            candidate = self._build_messages(prefix, rounds)
            estimated = _estimate_chars(candidate)
            if estimated <= self._max_chars:
                return CompactionResult(candidate, removed, estimated)

        candidate = self._build_messages(prefix, rounds)
        estimated = _estimate_chars(candidate)
        if estimated > self._max_chars:
            raise ContextLimitError(
                "system 指令、用户任务和最近工具轮次已超过上下文预算，"
                "请缩小任务范围或提高 DEEPSEEK_MAX_CONTEXT_CHARS。"
            )
        return CompactionResult(candidate, removed, estimated)

    def _build_messages(
        self,
        prefix: Sequence[Mapping[str, Any]],
        rounds: Sequence[Sequence[Mapping[str, Any]]],
    ) -> list[Mapping[str, Any]]:
        system_message = dict(prefix[0])
        system_message["content"] = self._system_content()
        result: list[Mapping[str, Any]] = [system_message, dict(prefix[1])]
        for round_messages in rounds:
            result.extend(round_messages)
        return result

    def _system_content(self) -> str:
        if not self._summary_lines:
            return self._base_system_prompt
        lines = list(self._summary_lines)
        if self._omitted_summary_lines:
            lines.insert(
                0,
                f"- 更早的 {self._omitted_summary_lines} 条工具记录已省略。",
            )
        summary = "\n".join(lines)
        return f"{self._base_system_prompt}\n[较早工具历史摘要]\n{summary}"

    def _record_round(self, round_messages: Sequence[Mapping[str, Any]]) -> None:
        if not round_messages:
            return
        assistant = round_messages[0]
        results = {
            message.get("tool_call_id"): _tool_status(message.get("content"))
            for message in round_messages[1:]
            if message.get("role") == "tool"
        }
        for tool_call in assistant.get("tool_calls", []):
            function = tool_call.get("function", {})
            name = function.get("name", "unknown")
            arguments = str(function.get("arguments", "{}"))
            preview = _single_line(arguments)[:ARGUMENT_PREVIEW_CHARS]
            status = results.get(tool_call.get("id"), "结果未知")
            self._summary_lines.append(f"- {name}({preview})：{status}")

        while len(self._summary_lines) > MAX_SUMMARY_LINES:
            self._summary_lines.pop(0)
            self._omitted_summary_lines += 1


def _split_rounds(
    history: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    """Group each assistant message with all following tool results."""
    rounds: list[list[Mapping[str, Any]]] = []
    for message in history:
        if message.get("role") == "assistant":
            rounds.append([message])
        elif message.get("role") == "tool" and rounds:
            rounds[-1].append(message)
        else:
            raise ContextLimitError("对话历史结构无效，无法安全压缩。")
    return rounds


def _tool_status(content: Any) -> str:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return "结果格式未知"
    return "成功" if payload.get("success") else "失败"


def _estimate_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def _single_line(value: str) -> str:
    return " ".join(value.split())
