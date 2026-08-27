"""Bounded shell-command execution inside the workspace."""

import os
from pathlib import Path
import signal
import subprocess
from typing import Union

from mini_agent.tools.result import ToolResult
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
    ) -> None:
        resolved = Path(workspace).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"工作目录不存在或不是目录：{resolved}")
        self._workspace = resolved
        self._policy = CommandPolicy(allow_dangerous_commands)

    def run_command(
        self,
        command: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
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

        decision = self._policy.check(command)
        if not decision.allowed:
            return ToolResult(False, f"命令被安全策略阻止：{decision.reason}")

        process = subprocess.Popen(
            ["/bin/zsh", "-lc", command],
            cwd=self._workspace,
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
                f"命令执行超过 {timeout_seconds:g} 秒，已终止。\n{details}",
            )

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
