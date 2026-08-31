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
    removed_turns: int = 0


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
        self._runtime_context = ""
        self._summary_lines: list[str] = []
        self._omitted_summary_lines = 0

    def start(self, system_prompt: str) -> None:
        """Reset compaction state for a new user task."""
        self._base_system_prompt = system_prompt
        self._runtime_context = ""
        self._summary_lines.clear()
        self._omitted_summary_lines = 0

    @property
    def system_content(self) -> str:
        """Return the current system prompt including compacted summaries."""
        return self._system_content()

    def set_runtime_context(self, content: str) -> None:
        """Set non-persistent context used only for the current user turn."""
        self._runtime_context = content.strip()

    def export_state(self) -> Mapping[str, Any]:
        """Return JSON-compatible compaction state for session persistence."""
        return {
            "summary_lines": list(self._summary_lines),
            "omitted_summary_lines": self._omitted_summary_lines,
        }

    def restore(self, system_prompt: str, state: Mapping[str, Any]) -> None:
        """Restore validated compaction state for a persisted session."""
        if not isinstance(state, Mapping):
            raise ContextLimitError("会话中的上下文状态必须是对象。")
        summary_lines = state.get("summary_lines", [])
        omitted = state.get("omitted_summary_lines", 0)
        if not isinstance(summary_lines, list) or not all(
            isinstance(line, str) for line in summary_lines
        ):
            raise ContextLimitError("会话中的上下文摘要格式无效。")
        if len(summary_lines) > MAX_SUMMARY_LINES:
            raise ContextLimitError("会话中的上下文摘要条目过多。")
        if isinstance(omitted, bool) or not isinstance(omitted, int) or omitted < 0:
            raise ContextLimitError("会话中的上下文省略计数无效。")

        self._base_system_prompt = system_prompt
        self._runtime_context = ""
        self._summary_lines = list(summary_lines)
        self._omitted_summary_lines = omitted

    def prepare(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> CompactionResult:
        """Return messages that fit the configured approximate budget."""
        prepared = [dict(message) for message in messages]
        if prepared and prepared[0].get("role") == "system":
            prepared[0]["content"] = self._system_content()
        estimated = _estimate_chars(prepared)
        if estimated <= self._max_chars:
            return CompactionResult(prepared, 0, estimated, 0)
        if len(prepared) < 2:
            raise ContextLimitError("上下文缺少必要的 system 或 user 消息。")

        system_message = prepared[0]
        if system_message.get("role") != "system":
            raise ContextLimitError("对话历史必须以 system 消息开始。")
        turns = _split_turns(prepared[1:])
        removed_rounds = 0
        removed_turns = 0

        while len(turns) > 1:
            self._record_turn(turns.pop(0))
            removed_turns += 1
            candidate = self._build_messages(system_message, turns)
            estimated = _estimate_chars(candidate)
            if estimated <= self._max_chars:
                return CompactionResult(
                    candidate,
                    removed_rounds,
                    estimated,
                    removed_turns,
                )

        latest_turn = _split_turn_units(turns[0])
        removable_rounds = [
            unit for unit in latest_turn if _is_tool_round(unit)
        ]
        while len(removable_rounds) > self._keep_recent_rounds:
            oldest = removable_rounds.pop(0)
            latest_turn.remove(oldest)
            self._record_round(oldest)
            removed_rounds += 1
            turns[0] = [message for unit in latest_turn for message in unit]
            candidate = self._build_messages(system_message, turns)
            estimated = _estimate_chars(candidate)
            if estimated <= self._max_chars:
                return CompactionResult(
                    candidate,
                    removed_rounds,
                    estimated,
                    removed_turns,
                )

        candidate = self._build_messages(system_message, turns)
        estimated = _estimate_chars(candidate)
        if estimated > self._max_chars:
            raise ContextLimitError(
                "system 指令、当前用户任务和最近工具轮次已超过上下文预算，"
                "请缩小任务范围或提高 DEEPSEEK_MAX_CONTEXT_CHARS。"
            )
        return CompactionResult(
            candidate,
            removed_rounds,
            estimated,
            removed_turns,
        )

    def _build_messages(
        self,
        system: Mapping[str, Any],
        turns: Sequence[Sequence[Mapping[str, Any]]],
    ) -> list[Mapping[str, Any]]:
        system_message = dict(system)
        system_message["content"] = self._system_content()
        result: list[Mapping[str, Any]] = [system_message]
        for turn in turns:
            result.extend(turn)
        return result

    def _system_content(self) -> str:
        sections = [self._base_system_prompt]
        if self._summary_lines:
            lines = list(self._summary_lines)
            if self._omitted_summary_lines:
                lines.insert(
                    0,
                    f"- 更早的 {self._omitted_summary_lines} 条工具记录已省略。",
                )
            sections.append("[较早工具历史摘要]\n" + "\n".join(lines))
        if self._runtime_context:
            sections.append(
                "[与当前任务相关的长期记忆]\n" + self._runtime_context
            )
        return "\n".join(sections)

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
            self._append_summary(f"- 工具 {name}({preview})：{status}")

    def _record_turn(self, turn: Sequence[Mapping[str, Any]]) -> None:
        """Summarize one complete earlier user turn before removing it."""
        user_content = str(turn[0].get("content", ""))
        self._append_summary(
            f"- 用户要求：{_single_line(user_content)[:ARGUMENT_PREVIEW_CHARS]}"
        )
        for unit in _split_turn_units(turn):
            if _is_tool_round(unit):
                self._record_round(unit)
        final_message = turn[-1]
        if final_message.get("role") == "assistant" and not final_message.get(
            "tool_calls"
        ):
            final_content = str(final_message.get("content", ""))
            self._append_summary(
                f"- 模型答复：{_single_line(final_content)[:ARGUMENT_PREVIEW_CHARS]}"
            )

    def _append_summary(self, line: str) -> None:
        """Append one bounded summary line."""
        self._summary_lines.append(line)

        while len(self._summary_lines) > MAX_SUMMARY_LINES:
            self._summary_lines.pop(0)
            self._omitted_summary_lines += 1


def _split_turns(
    history: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    """Split a session into user turns and validate tool-call pairing."""
    turns: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    pending_call_ids: set[str] = set()

    for message in history:
        role = message.get("role")
        if role == "user":
            if pending_call_ids:
                raise ContextLimitError("对话历史结构无效，工具结果不完整。")
            if current:
                turns.append(current)
            current = [message]
            continue

        if not current:
            raise ContextLimitError("对话历史结构无效，消息缺少用户轮次。")
        if role == "assistant":
            if pending_call_ids:
                raise ContextLimitError("对话历史结构无效，工具结果不完整。")
            current.append(message)
            pending_call_ids = {
                str(tool_call.get("id"))
                for tool_call in message.get("tool_calls", [])
            }
        elif role == "tool":
            call_id = str(message.get("tool_call_id"))
            if call_id not in pending_call_ids:
                raise ContextLimitError("对话历史结构无效，存在孤立工具结果。")
            current.append(message)
            pending_call_ids.remove(call_id)
        else:
            raise ContextLimitError(f"对话历史结构无效，未知角色：{role}")

    if pending_call_ids:
        raise ContextLimitError("对话历史结构无效，工具结果不完整。")
    if current:
        turns.append(current)
    if not turns:
        raise ContextLimitError("上下文缺少必要的 user 消息。")
    return turns


def validate_history(history: Sequence[Mapping[str, Any]]) -> None:
    """Validate persisted non-system messages without modifying them."""
    if not history:
        return
    _split_turns(history)


def _split_turn_units(
    turn: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    """Group a turn into one user message and atomic assistant/tool units."""
    units: list[list[Mapping[str, Any]]] = [[turn[0]]]
    for message in turn[1:]:
        if message.get("role") == "assistant":
            units.append([message])
        elif message.get("role") == "tool" and _is_tool_round(units[-1]):
            units[-1].append(message)
        else:
            raise ContextLimitError("对话历史结构无效，无法安全压缩。")
    return units


def _is_tool_round(unit: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        unit
        and unit[0].get("role") == "assistant"
        and unit[0].get("tool_calls")
    )


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
