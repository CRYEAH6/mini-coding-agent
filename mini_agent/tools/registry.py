"""Tool definitions and dispatch for model-requested actions."""

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from mini_agent.tools.approval import ApprovalHandler
from mini_agent.tools.filesystem import FileTools
from mini_agent.tools.git import GitTools
from mini_agent.tools.result import ToolResult
from mini_agent.tools.shell import ShellTool


ToolHandler = Callable[..., ToolResult]


class ToolRegistry:
    """Expose tool schemas and dispatch validated JSON arguments."""

    def __init__(
        self,
        workspace: Union[str, Path],
        *,
        allow_dangerous_commands: bool = False,
        approval_handler: Optional[ApprovalHandler] = None,
    ) -> None:
        file_tools = FileTools(workspace)
        git_tools = GitTools(workspace, approval_handler=approval_handler)
        shell_tool = ShellTool(
            workspace,
            allow_dangerous_commands=allow_dangerous_commands,
            approval_handler=approval_handler,
        )
        self._handlers: dict[str, ToolHandler] = {
            "list_files": file_tools.list_files,
            "read_file": file_tools.read_file,
            "write_file": file_tools.write_file,
            "replace_in_file": file_tools.replace_in_file,
            "run_command": shell_tool.run_command,
            "git_status": git_tools.git_status,
            "git_diff": git_tools.git_diff,
            "git_checkpoint": git_tools.git_checkpoint,
        }

    @property
    def definitions(self) -> Sequence[Mapping[str, Any]]:
        """Return OpenAI-compatible function tool definitions."""
        return TOOL_DEFINITIONS

    def execute(self, name: str, arguments: str) -> ToolResult:
        """Parse a model tool call and execute its registered handler."""
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(False, f"未知工具：{name}")

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return ToolResult(False, f"工具参数不是有效 JSON：{exc.msg}")
        if not isinstance(parsed, dict):
            return ToolResult(False, "工具参数必须是 JSON 对象。")

        try:
            return handler(**parsed)
        except TypeError as exc:
            return ToolResult(False, f"工具参数不正确：{exc}")
        except Exception as exc:  # Keep one faulty tool call from crashing the agent.
            return ToolResult(False, f"工具执行出现未预期错误：{exc}")


TOOL_DEFINITIONS: Sequence[Mapping[str, Any]] = (
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出工作目录中指定目录的直接子项。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对工作目录的路径，默认为当前目录。",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作目录内的 UTF-8 文本文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对工作目录的文件路径。",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "创建或完整写入工作目录内的 UTF-8 文本文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对工作目录的文件路径。",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文件内容。",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "精确替换文本文件中的一段内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对工作目录的文件路径。",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "需要被替换的原始文本。",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "替换后的新文本。",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "是否替换全部匹配，默认为 false。",
                    },
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "在工作目录中执行一条 zsh 命令并返回退出码和输出；"
                "高风险命令默认会被阻止。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "需要执行的命令。"},
                    "timeout_seconds": {
                        "type": "number",
                        "description": "超时秒数，默认 30，最大 120。",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": (
                "读取 Git 分支、工作区修改和最近的 Agent 检查点；"
                "不会改变仓库。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": (
                "比较当前完整工作区与 HEAD 或指定 Agent 检查点，"
                "包含已暂存、未暂存和未跟踪文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {
                        "type": "string",
                        "description": "Git 基准，默认为 HEAD。",
                    },
                    "path": {
                        "type": "string",
                        "description": "工作目录内相对路径，默认为整个项目。",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkpoint",
            "description": (
                "经用户确认后将当前工作区保存为本地 Git 检查点；"
                "不修改当前分支、暂存区或工作文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "简短说明创建检查点的原因。",
                    }
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
)
