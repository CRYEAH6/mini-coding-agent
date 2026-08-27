"""Workspace-scoped file inspection and editing tools."""

import os
from pathlib import Path
from typing import Union

from mini_agent.tools.result import ToolResult


MAX_FILE_CHARS = 50_000
MAX_DIRECTORY_ENTRIES = 200
IGNORED_NAMES = {".git", ".venv", "__pycache__"}


class FileTools:
    """Perform text-file operations inside one workspace."""

    def __init__(self, workspace: Union[str, Path]) -> None:
        resolved = Path(workspace).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"工作目录不存在或不是目录：{resolved}")
        self._workspace = resolved

    def list_files(self, path: str = ".") -> ToolResult:
        """List direct children of a workspace directory."""
        if isinstance(path, str) and not path.strip():
            path = "."
        try:
            target = self._resolve_path(path, must_exist=True)
            if not target.is_dir():
                return ToolResult(False, f"目标不是目录：{path}")

            entries = [item for item in target.iterdir() if item.name not in IGNORED_NAMES]
            entries.sort(key=lambda item: (not item.is_dir(), item.name.lower()))
            lines = []
            for item in entries[:MAX_DIRECTORY_ENTRIES]:
                relative = item.relative_to(self._workspace).as_posix()
                lines.append(f"{relative}/" if item.is_dir() else relative)

            if len(entries) > MAX_DIRECTORY_ENTRIES:
                lines.append(f"... 其余 {len(entries) - MAX_DIRECTORY_ENTRIES} 项已省略")
            return ToolResult(True, "\n".join(lines) or "目录为空。")
        except (OSError, ValueError) as exc:
            return ToolResult(False, str(exc))

    def read_file(self, path: str) -> ToolResult:
        """Read a UTF-8 text file with a bounded result size."""
        try:
            target = self._resolve_path(path, must_exist=True)
            if not target.is_file():
                return ToolResult(False, f"目标不是文件：{path}")

            content = target.read_text(encoding="utf-8")
            if len(content) > MAX_FILE_CHARS:
                omitted = len(content) - MAX_FILE_CHARS
                content = f"{content[:MAX_FILE_CHARS]}\n... 其余 {omitted} 个字符已省略"
            return ToolResult(True, content)
        except UnicodeDecodeError:
            return ToolResult(False, f"文件不是有效的 UTF-8 文本：{path}")
        except (OSError, ValueError) as exc:
            return ToolResult(False, str(exc))

    def write_file(self, path: str, content: str) -> ToolResult:
        """Create or overwrite a UTF-8 text file."""
        if not isinstance(content, str):
            return ToolResult(False, "content 必须是字符串。")

        try:
            target = self._resolve_path(path, must_exist=False)
            if target.exists() and target.is_dir():
                return ToolResult(False, f"目标是目录，不能写入：{path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            relative = target.relative_to(self._workspace).as_posix()
            return ToolResult(True, f"已写入 {relative}（{len(content)} 个字符）。")
        except (OSError, ValueError) as exc:
            return ToolResult(False, str(exc))

    def replace_in_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> ToolResult:
        """Replace one exact text block, or all matches when requested."""
        if not all(isinstance(value, str) for value in (old_text, new_text)):
            return ToolResult(False, "old_text 和 new_text 必须是字符串。")
        if not isinstance(replace_all, bool):
            return ToolResult(False, "replace_all 必须是布尔值。")
        if not old_text:
            return ToolResult(False, "old_text 不能为空。")

        try:
            target = self._resolve_path(path, must_exist=True)
            if not target.is_file():
                return ToolResult(False, f"目标不是文件：{path}")

            content = target.read_text(encoding="utf-8")
            match_count = content.count(old_text)
            if match_count == 0:
                return ToolResult(False, "未找到要替换的旧文本。")
            if match_count > 1 and not replace_all:
                return ToolResult(
                    False,
                    f"旧文本出现 {match_count} 次，请提供更精确的内容。",
                )

            count = -1 if replace_all else 1
            target.write_text(content.replace(old_text, new_text, count), encoding="utf-8")
            replaced = match_count if replace_all else 1
            return ToolResult(True, f"已完成 {replaced} 处替换。")
        except UnicodeDecodeError:
            return ToolResult(False, f"文件不是有效的 UTF-8 文本：{path}")
        except (OSError, ValueError) as exc:
            return ToolResult(False, str(exc))

    def _resolve_path(self, path: str, *, must_exist: bool) -> Path:
        """Resolve a relative path and reject workspace escapes."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path 必须是非空字符串。")

        supplied = Path(path).expanduser()
        if supplied.is_absolute():
            raise ValueError("只允许使用工作目录内的相对路径。")

        candidate = (self._workspace / supplied).resolve(strict=False)
        if os.path.commonpath((self._workspace, candidate)) != str(self._workspace):
            raise ValueError("路径超出了允许的工作目录。")
        if must_exist and not candidate.exists():
            raise ValueError(f"路径不存在：{path}")
        return candidate
