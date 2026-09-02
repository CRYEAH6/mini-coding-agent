"""Bounded shell-command execution inside the workspace."""

import os
from pathlib import Path
import signal
import subprocess
from typing import Optional, Union

from mini_agent.tools.access import WorkspaceAccessManager
from mini_agent.tools.approval import ApprovalHandler, ApprovalRequest
from mini_agent.tools.result import ToolResult
from mini_agent.tools.sandbox import CommandSandbox, SandboxError, STRICT_MODE
from mini_agent.tools.security import CommandPolicy


DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_CHARS = 30_000


class ShellTool:
    """Execute shell commands with time and output limits."""

    def __init__(
        self,
        workspace: Union[str, Path],
        *,
        allow_dangerous_commands: bool = False,
        approval_handler: Optional[ApprovalHandler] = None,
        sandbox_mode: str = STRICT_MODE,
        access_manager: Optional[WorkspaceAccessManager] = None,
    ) -> None:
        resolved = Path(workspace).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"工作目录不存在或不是目录：{resolved}")
        self._workspace = resolved
        self._policy = CommandPolicy(allow_dangerous_commands)
        self._request_approval = approval_handler
        self._sandbox = CommandSandbox(resolved, sandbox_mode)
        self._access = access_manager or WorkspaceAccessManager(
            resolved,
            approval_handler,
        )

    def run_command(
        self,
        command: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        network_access: bool = False,
    ) -> ToolResult:
        """Run one command in zsh and return its exit status and output."""
        if not isinstance(command, str) or not command.strip():
            return ToolResult(False, "command 必须是非空字符串。")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            return ToolResult(False, "timeout_seconds 必须是数字。")
        if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
            return ToolResult(
                False,
                f"timeout_seconds 必须在 0 到 {MAX_TIMEOUT_SECONDS} 之间。",
            )
        if not isinstance(network_access, bool):
            return ToolResult(False, "network_access 必须是布尔值。")

        decision = self._policy.check(command)
        if not decision.allowed:
            return ToolResult(False, f"命令被安全策略阻止：{decision.reason}")
        allow_network = decision.network_access or network_access
        requires_approval = decision.requires_approval or network_access
        if requires_approval:
            reason = decision.reason
            if network_access and not decision.network_access:
                reason = "模型请求为该命令临时开放外部网络。"
            request = ApprovalRequest(
                tool_name="run_command",
                action="执行需授权命令",
                details=command,
                reason=reason,
            )
            if self._request_approval is None:
                return ToolResult(
                    False,
                    "该命令需要用户确认，但当前运行方式未提供确认入口。",
                )
            if not self._request_approval(request):
                return ToolResult(False, "用户拒绝执行该命令。")

        try:
            with self._sandbox.prepare(
                command,
                allow_network=allow_network,
                external_read_paths=self._access.readable_roots,
                external_write_paths=self._access.writable_roots,
            ) as plan:
                process = subprocess.Popen(
                    plan.arguments,
                    cwd=self._workspace,
                    env=plan.environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    stdout, stderr = process.communicate()
                    details = self._format_output(stdout, stderr)
                    return ToolResult(
                        False,
                        f"命令执行超过 {timeout_seconds:g} 秒，"
                        f"已终止。\n{details}",
                    )
        except SandboxError as exc:
            return ToolResult(False, f"无法启用系统沙箱：{exc}")
        except OSError as exc:
            return ToolResult(False, f"无法启动命令进程：{exc}")

        if (
            self._sandbox.mode == STRICT_MODE
            and process.returncode == 71
            and "sandbox_apply" in stderr
        ):
            return ToolResult(False, f"macOS 系统沙箱启动失败：{stderr.strip()}")

        details = self._format_output(stdout, stderr)
        return ToolResult(
            process.returncode == 0,
            f"退出码：{process.returncode}\n{details}",
        )

    @staticmethod
    def _format_output(stdout: str, stderr: str) -> str:
        """Format and truncate captured output for the model context."""
        sections = []
        if stdout:
            sections.append(f"标准输出：\n{stdout.rstrip()}")
        if stderr:
            sections.append(f"标准错误：\n{stderr.rstrip()}")
        content = "\n".join(sections) or "命令没有产生输出。"
        if len(content) > MAX_OUTPUT_CHARS:
            omitted = len(content) - MAX_OUTPUT_CHARS
            return f"{content[:MAX_OUTPUT_CHARS]}\n... 其余 {omitted} 个字符已省略"
        return content
