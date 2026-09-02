"""Session-scoped approval for projects outside the primary workspace."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from mini_agent.tools.approval import ApprovalHandler, ApprovalRequest
from mini_agent.tools.result import ToolResult


READ_ACCESS = "read"
READ_WRITE_ACCESS = "read_write"
ACCESS_MODES = (READ_ACCESS, READ_WRITE_ACCESS)


@dataclass(frozen=True)
class WorkspaceGrant:
    """One approved external directory and its maximum access level."""

    root: Path
    writable: bool


class WorkspaceAccessManager:
    """Keep external-directory grants in memory for one Agent process."""

    def __init__(
        self,
        workspace: Union[str, Path],
        approval_handler: Optional[ApprovalHandler] = None,
    ) -> None:
        resolved = Path(workspace).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"工作目录不存在或不是目录：{resolved}")
        self._workspace = resolved
        self._request_approval = approval_handler
        self._grants: dict[Path, WorkspaceGrant] = {}

    @property
    def workspace(self) -> Path:
        """Return the always-authorized primary workspace."""
        return self._workspace

    @property
    def readable_roots(self) -> tuple[Path, ...]:
        """Return external roots approved for at least reading."""
        return tuple(grant.root for grant in self._grants.values())

    @property
    def writable_roots(self) -> tuple[Path, ...]:
        """Return external roots approved for writing."""
        return tuple(
            grant.root for grant in self._grants.values() if grant.writable
        )

    def request_access(
        self,
        path: str,
        access: str = READ_ACCESS,
        reason: str = "完成当前编程任务需要访问该项目。",
    ) -> ToolResult:
        """Ask the user to grant temporary access to an external directory."""
        if not isinstance(path, str) or not path.strip():
            return ToolResult(False, "path 必须是非空字符串。")
        if access not in ACCESS_MODES:
            return ToolResult(
                False,
                f"access 必须是 {READ_ACCESS} 或 {READ_WRITE_ACCESS}。",
            )
        if not isinstance(reason, str) or not reason.strip():
            return ToolResult(False, "reason 必须是非空字符串。")

        supplied = Path(path).expanduser()
        if not supplied.is_absolute():
            return ToolResult(False, "外部项目必须使用绝对路径。")
        target = supplied.resolve(strict=False)
        if not target.exists():
            return ToolResult(False, f"外部项目目录不存在：{target}")
        if not target.is_dir():
            return ToolResult(False, f"外部项目路径不是目录：{target}")
        if target == Path(target.anchor):
            return ToolResult(False, "不能将文件系统根目录授权为外部项目。")
        if self._contains(self._workspace, target):
            return ToolResult(True, "该目录位于主工作区内，无需额外授权。")

        writable = access == READ_WRITE_ACCESS
        if self.can_access(target, write=writable):
            return ToolResult(True, "该外部项目已经获得所需权限。")
        if self._request_approval is None:
            return ToolResult(
                False,
                "访问外部项目需要用户确认，但当前运行方式未提供确认入口。",
            )

        access_label = "读取和修改" if writable else "只读"
        request = ApprovalRequest(
            tool_name="request_workspace_access",
            action="访问主工作区外的项目",
            details=(
                f"目标目录：{target}\n"
                f"访问权限：{access_label}\n"
                "授权范围：仅本次 Agent 运行期间"
            ),
            reason=reason.strip(),
        )
        if not self._request_approval(request):
            return ToolResult(False, "用户拒绝访问该外部项目。")

        current = self._grants.get(target)
        self._grants[target] = WorkspaceGrant(
            target,
            writable=writable or (current.writable if current else False),
        )
        return ToolResult(
            True,
            f"已临时授权{access_label}外部项目：{target}",
        )

    def can_access(self, path: Path, *, write: bool) -> bool:
        """Return whether one resolved path is covered by current grants."""
        target = path.resolve(strict=False)
        if self._contains(self._workspace, target):
            return True
        return any(
            self._contains(grant.root, target)
            and (grant.writable or not write)
            for grant in self._grants.values()
        )

    def require_access(self, path: Path, *, write: bool) -> None:
        """Reject an external path that lacks the required grant."""
        if self.can_access(path, write=write):
            return
        access = READ_WRITE_ACCESS if write else READ_ACCESS
        raise ValueError(
            "路径位于主工作区之外。请先调用 request_workspace_access，"
            f"为外部项目申请 {access} 权限。"
        )

    @staticmethod
    def _contains(root: Path, target: Path) -> bool:
        try:
            target.relative_to(root)
        except ValueError:
            return False
        return True
