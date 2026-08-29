"""Shared types for user approval before sensitive tool actions."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ApprovalRequest:
    """Describe one sensitive action that requires a user decision."""

    tool_name: str
    action: str
    details: str
    reason: str


ApprovalHandler = Callable[[ApprovalRequest], bool]
