"""Non-destructive Git inspection and internal checkpoint tools."""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Mapping, Optional, Sequence, Union

from mini_agent.tools.approval import ApprovalHandler, ApprovalRequest
from mini_agent.tools.result import ToolResult


GIT_TIMEOUT_SECONDS = 30.0
MAX_GIT_OUTPUT_CHARS = 50_000
MAX_APPROVAL_DETAILS_CHARS = 5_000
MAX_CHECKPOINT_MESSAGE_CHARS = 200
CHECKPOINT_REF_PREFIX = "refs/mini-agent/checkpoints"
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9_./@{}^~:+-]+$")


class GitToolError(RuntimeError):
    """Raised when a bounded Git operation cannot be completed."""


class GitTools:
    """Inspect a repository and create refs without changing its worktree."""

    def __init__(
        self,
        workspace: Union[str, Path],
        *,
        approval_handler: Optional[ApprovalHandler] = None,
    ) -> None:
        resolved = Path(workspace).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"工作目录不存在或不是目录：{resolved}")
        self._workspace = resolved
        self._request_approval = approval_handler

    def git_status(self) -> ToolResult:
        """Return branch state, changed files, and recent checkpoints."""
        try:
            self._require_repository_root()
            status = self._run_checked(("status", "--short", "--branch"))
            checkpoints = self._run_checked(
                (
                    "for-each-ref",
                    "--sort=-creatordate",
                    "--count=10",
                    "--format=%(refname:short) %(objectname:short) %(subject)",
                    CHECKPOINT_REF_PREFIX,
                )
            )
            sections = [status.strip() or "工作区干净。"]
            if checkpoints.strip():
                sections.append(f"最近的 Agent 检查点：\n{checkpoints.strip()}")
            else:
                sections.append("尚未创建 Agent 检查点。")
            return ToolResult(True, self._truncate("\n\n".join(sections)))
        except GitToolError as exc:
            return ToolResult(False, str(exc))

    def git_diff(self, base: str = "HEAD", path: str = ".") -> ToolResult:
        """Compare the complete current worktree snapshot with a Git revision."""
        try:
            self._require_repository_root()
            revision = self._validate_revision(base)
            relative_path = self._validate_path(path)
            base_tree = self._resolve_tree(revision)
            current_tree = self._write_worktree_tree()
            output = self._run_checked(
                (
                    "diff-tree",
                    "-r",
                    "--patch",
                    "--stat",
                    "--no-ext-diff",
                    base_tree,
                    current_tree,
                    "--",
                    relative_path,
                )
            )
            if not output.strip():
                return ToolResult(True, f"当前工作区与 {revision} 没有差异。")
            return ToolResult(True, self._truncate(output.rstrip()))
        except GitToolError as exc:
            return ToolResult(False, str(exc))

    def git_checkpoint(self, message: str) -> ToolResult:
        """Save the current worktree as an internal Git ref after approval."""
        if not isinstance(message, str) or not message.strip():
            return ToolResult(False, "message 必须是非空字符串。")
        message = message.strip()
        if len(message) > MAX_CHECKPOINT_MESSAGE_CHARS:
            return ToolResult(
                False,
                f"message 不能超过 {MAX_CHECKPOINT_MESSAGE_CHARS} 个字符。",
            )

        try:
            self._require_repository_root()
            status = self._run_checked(
                ("status", "--short", "--untracked-files=all")
            )
            if not status.strip():
                return ToolResult(False, "工作区没有需要保存的修改。")

            request = ApprovalRequest(
                tool_name="git_checkpoint",
                action="创建本地 Git 检查点",
                details=self._checkpoint_details(message, status),
                reason=(
                    "该操作会把当前工作区快照保存为本地 Git 引用，"
                    "但不会修改当前分支、暂存区或工作文件。"
                ),
            )
            if self._request_approval is None:
                return ToolResult(
                    False,
                    "创建检查点需要用户确认，"
                    "但当前运行方式未提供确认入口。",
                )
            if not self._request_approval(request):
                return ToolResult(False, "用户拒绝创建 Git 检查点。")

            tree = self._write_worktree_tree()
            head = self._try_resolve_commit("HEAD")
            commit_arguments = ["commit-tree", tree, "-m", message]
            if head is not None:
                commit_arguments[2:2] = ["-p", head]
            identity = {
                "GIT_AUTHOR_NAME": "Mini Coding Agent",
                "GIT_AUTHOR_EMAIL": "mini-agent@local",
                "GIT_COMMITTER_NAME": "Mini Coding Agent",
                "GIT_COMMITTER_EMAIL": "mini-agent@local",
            }
            commit = self._run_checked(commit_arguments, extra_env=identity).strip()
            ref_name = self._next_checkpoint_ref()
            self._run_checked(("update-ref", ref_name, commit))
            short_ref = ref_name.removeprefix("refs/")
            return ToolResult(
                True,
                "已创建本地 Git 检查点。\n"
                f"引用：{short_ref}\n"
                f"提交：{commit}\n"
                "当前分支、暂存区和工作文件均未改变。",
            )
        except GitToolError as exc:
            return ToolResult(False, str(exc))

    def _require_repository_root(self) -> None:
        root = self._run_checked(("rev-parse", "--show-toplevel")).strip()
        if Path(root).resolve() != self._workspace:
            raise GitToolError(
                "Git 工具要求工作目录是仓库根目录，"
                f"当前仓库根目录为：{root}"
            )

    def _write_worktree_tree(self) -> str:
        """Write a tree through a temporary index without touching the real one."""
        with tempfile.TemporaryDirectory(prefix="mini-agent-git-") as directory:
            index_path = Path(directory) / "index"
            environment = {"GIT_INDEX_FILE": str(index_path)}
            head = self._try_resolve_commit("HEAD")
            if head is None:
                self._run_checked(("read-tree", "--empty"), extra_env=environment)
            else:
                self._run_checked(("read-tree", head), extra_env=environment)
            self._run_checked(("add", "-A", "--", "."), extra_env=environment)
            return self._run_checked(
                ("write-tree",),
                extra_env=environment,
            ).strip()

    def _resolve_tree(self, revision: str) -> str:
        try:
            return self._run_checked(
                ("rev-parse", "--verify", f"{revision}^{{tree}}")
            ).strip()
        except GitToolError as exc:
            raise GitToolError(f"找不到 Git 基准：{revision}") from exc

    def _try_resolve_commit(self, revision: str) -> Optional[str]:
        completed = self._run(("rev-parse", "--verify", f"{revision}^{{commit}}"))
        return completed.stdout.strip() if completed.returncode == 0 else None

    def _validate_revision(self, revision: str) -> str:
        if not isinstance(revision, str) or not revision.strip():
            raise GitToolError("base 必须是非空字符串。")
        revision = revision.strip()
        if revision.startswith("-") or not REVISION_PATTERN.fullmatch(revision):
            raise GitToolError("base 不是安全、有效的 Git 引用格式。")
        return revision

    def _validate_path(self, path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise GitToolError("path 必须是非空字符串。")
        supplied = Path(path)
        if supplied.is_absolute():
            raise GitToolError("只允许比较工作目录内的相对路径。")
        candidate = (self._workspace / supplied).resolve(strict=False)
        if os.path.commonpath((self._workspace, candidate)) != str(self._workspace):
            raise GitToolError("路径超出了允许的工作目录。")
        return supplied.as_posix()

    def _next_checkpoint_ref(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{CHECKPOINT_REF_PREFIX}/{timestamp}"

    @staticmethod
    def _checkpoint_details(message: str, status: str) -> str:
        details = f"说明：{message}\n当前修改：\n{status.strip()}"
        if len(details) <= MAX_APPROVAL_DETAILS_CHARS:
            return details
        omitted = len(details) - MAX_APPROVAL_DETAILS_CHARS
        return f"{details[:MAX_APPROVAL_DETAILS_CHARS]}\n... 其余 {omitted} 个字符已省略"

    def _run_checked(
        self,
        arguments: Sequence[str],
        *,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> str:
        completed = self._run(arguments, extra_env=extra_env)
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise GitToolError(f"Git 操作失败：{details or '未知错误'}")
        return completed.stdout

    def _run(
        self,
        arguments: Sequence[str],
        *,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        if extra_env:
            environment.update(extra_env)
        try:
            return subprocess.run(
                ("git", *arguments),
                cwd=self._workspace,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitToolError("未找到 git 命令。") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitToolError("Git 操作超时，已终止等待。") from exc

    @staticmethod
    def _truncate(content: str) -> str:
        if len(content) <= MAX_GIT_OUTPUT_CHARS:
            return content
        omitted = len(content) - MAX_GIT_OUTPUT_CHARS
        return f"{content[:MAX_GIT_OUTPUT_CHARS]}\n... 其余 {omitted} 个字符已省略"
