"""Shared tool result type."""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ToolResult:
    """A serializable result returned by every local tool."""

    success: bool
    content: str

    def to_json(self) -> str:
        """Serialize the result for a model tool message."""
        return json.dumps(
            {"success": self.success, "content": self.content},
            ensure_ascii=False,
        )
