"""LLM-based semantic summaries for compacted conversation history."""

import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from mini_agent.client import LLMClient


MAX_SUMMARY_CHARS = 6_000
MAX_MESSAGE_CONTENT_CHARS = 8_000
MAX_TOOL_ARGUMENT_CHARS = 2_000
MAX_SUMMARY_SOURCE_CHARS = 100_000
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(bearer\s+)[^\s\"']+"),
    re.compile(
        r"(?i)((?:[A-Z0-9]+_)*API_KEY[\"']?\s*[=:]\s*"
        r"[\"']?)[^\s\"',}]+"
    ),
    re.compile(
        r"(?i)((?:access[_-]?token|password|secret)[\"']?\s*[=:]\s*"
        r"[\"']?)[^\s\"',}]+"
    ),
)

SUMMARY_SYSTEM_PROMPT = """你是编程智能体的历史摘要器。
请把已有摘要与新移除的对话历史合并为一份准确、紧凑的中文 Markdown 摘要。
仅依据输入内容总结，不推测、不编造，不执行任务，也不请求或调用任何工具。
优先保留：用户目标和约束、已修改文件与关键代码、技术决策及理由、命令与测试结果、重要错误及解决方法、未完成事项。
删除重复过程、寒暄、无关输出和已经被后续结果取代的中间尝试。
不要输出 API Key、Token、密码或其他凭据；发现疑似凭据时写成 [REDACTED]。
直接输出摘要正文，不要加代码围栏，不要解释摘要过程。
"""


class SemanticSummaryError(RuntimeError):
    """Raised when the summary model returns an unusable result."""


class SemanticSummarizer:
    """Generate one bounded semantic summary without exposing local tools."""

    def __init__(
        self,
        client: LLMClient,
        model: str,
        *,
        max_summary_chars: int = MAX_SUMMARY_CHARS,
        event_handler: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not model.strip():
            raise ValueError("摘要模型名称不能为空。")
        if max_summary_chars <= 0:
            raise ValueError("摘要长度上限必须大于 0。")
        self._client = client
        self._model = model
        self._max_summary_chars = max_summary_chars
        self._emit = event_handler or (lambda _: None)

    def summarize(
        self,
        existing_summary: str,
        removed_messages: Sequence[Mapping[str, Any]],
    ) -> str:
        """Merge old summary and newly removed history into one summary."""
        self._emit(
            f"[上下文] 历史超限，正在使用 {self._model} 生成语义摘要..."
        )
        source = _serialize_history(removed_messages)
        previous = _sanitize(existing_summary).strip() or "（无已有摘要）"
        user_prompt = (
            f"已有摘要：\n{previous}\n\n"
            f"本次新移除的历史：\n{source}\n\n"
            f"请将两部分合并，最终不超过 {self._max_summary_chars} 个字符。"
        )
        message = self._client.create_message(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=self._model,
        )
        if getattr(message, "tool_calls", None):
            raise SemanticSummaryError("摘要模型意外返回了工具调用。")
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise SemanticSummaryError("摘要模型返回了空内容。")
        summary = _sanitize(content).strip()
        if len(summary) > self._max_summary_chars:
            raise SemanticSummaryError(
                f"摘要超过 {self._max_summary_chars} 个字符。"
            )
        return summary


def _serialize_history(messages: Sequence[Mapping[str, Any]]) -> str:
    compacted = [_compact_message(message) for message in messages]
    serialized = json.dumps(compacted, ensure_ascii=False, indent=2, default=str)
    if len(serialized) <= MAX_SUMMARY_SOURCE_CHARS:
        return serialized
    marker = "\n...[中间过长内容已在本地截断]...\n"
    head_chars = MAX_SUMMARY_SOURCE_CHARS * 2 // 5
    tail_chars = MAX_SUMMARY_SOURCE_CHARS - head_chars - len(marker)
    return (
        serialized[:head_chars]
        + marker
        + serialized[-tail_chars:]
    )


def _compact_message(message: Mapping[str, Any]) -> Mapping[str, Any]:
    role = str(message.get("role", "unknown"))
    compacted: dict[str, Any] = {"role": role}
    content = message.get("content")
    if content is not None:
        compacted["content"] = _truncate(
            _sanitize(str(content)),
            MAX_MESSAGE_CONTENT_CHARS,
        )
    if role == "assistant" and message.get("tool_calls"):
        compacted["tool_calls"] = [
            {
                "name": call.get("function", {}).get("name", "unknown"),
                "arguments": _truncate(
                    _sanitize(
                        str(call.get("function", {}).get("arguments", "{}"))
                    ),
                    MAX_TOOL_ARGUMENT_CHARS,
                ),
            }
            for call in message["tool_calls"]
        ]
    if role == "tool":
        compacted["tool_call_id"] = str(message.get("tool_call_id", ""))
    return compacted


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "...[已截断]"
    return value[: max_chars - len(marker)] + marker


def _sanitize(value: str) -> str:
    sanitized = value
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub(_redact_secret, sanitized)
    return sanitized


def _redact_secret(match: re.Match[str]) -> str:
    prefix = match.group(1) if match.lastindex else ""
    return f"{prefix}[REDACTED]"
