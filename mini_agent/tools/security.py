"""Conservative policy checks for high-risk shell commands."""

from dataclasses import dataclass
import re
from typing import Pattern, Sequence, Tuple


@dataclass(frozen=True)
class CommandDecision:
    """Whether a command may run and the reason when it is blocked."""

    allowed: bool
    reason: str = ""


Rule = Tuple[Pattern[str], str]
COMMAND_START = r"(?:^|[;&|(\n]\s*)"

DANGEROUS_RULES: Sequence[Rule] = (
    (
        re.compile(rf"{COMMAND_START}(?:\S*/)?sudo(?:\s|$)", re.IGNORECASE),
        "禁止使用 sudo 提升权限。",
    ),
    (
        re.compile(
            rf"{COMMAND_START}(?:(?:command|env)\s+)*(?:\S*/)?rm(?:\s|$)",
            re.IGNORECASE,
        ),
        "默认禁止使用 rm 删除文件。",
    ),
    (
        re.compile(
            rf"{COMMAND_START}(?:shutdown|reboot|halt|poweroff)(?:\s|$)",
            re.IGNORECASE,
        ),
        "禁止执行关机或重启命令。",
    ),
    (
        re.compile(
            rf"{COMMAND_START}(?:\S*/)?(?:mkfs(?:\.\w+)?|diskutil|fdisk)(?:\s|$)",
            re.IGNORECASE,
        ),
        "禁止执行磁盘管理或格式化命令。",
    ),
    (
        re.compile(rf"{COMMAND_START}(?:\S*/)?dd(?:\s|$)", re.IGNORECASE),
        "禁止执行原始块复制命令 dd。",
    ),
    (
        re.compile(
            rf"{COMMAND_START}(?:\S*/)?git\b[^;&|\n]*\breset\b"
            r"[^;&|\n]*--hard(?:\s|$)",
            re.IGNORECASE,
        ),
        "禁止执行 git reset --hard。",
    ),
    (
        re.compile(
            rf"{COMMAND_START}(?:\S*/)?git\b[^;&|\n]*\bclean\b[^;&|\n]*"
            r"(?:-[A-Za-z]*f[A-Za-z]*|--force)(?:\s|$)",
            re.IGNORECASE,
        ),
        "禁止强制清理 Git 工作区。",
    ),
    (
        re.compile(
            rf"{COMMAND_START}(?:\S*/)?git\b[^;&|\n]*\bpush\b[^;&|\n]*"
            r"(?:--force(?:-with-lease)?|-f)(?:\s|$)",
            re.IGNORECASE,
        ),
        "禁止强制推送 Git 历史。",
    ),
    (
        re.compile(
            rf"{COMMAND_START}find\b[^;&|\n]*\s-delete(?:\s|$)",
            re.IGNORECASE,
        ),
        "禁止使用 find -delete 批量删除文件。",
    ),
    (
        re.compile(
            r"(?:curl|wget)\b[^|\n]*\|\s*(?:\S*/)?(?:sh|bash|zsh)\b",
            re.IGNORECASE,
        ),
        "禁止下载内容后直接交给 shell 执行。",
    ),
    (
        re.compile(r">\s*/dev/(?:disk\w*|sd\w*|nvme\w*)", re.IGNORECASE),
        "禁止重定向写入磁盘设备。",
    ),
    (
        re.compile(r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:", re.DOTALL),
        "禁止执行 fork bomb。",
    ),
)


class CommandPolicy:
    """Block recognizable high-risk commands unless explicitly disabled."""

    def __init__(self, allow_dangerous_commands: bool = False) -> None:
        self._allow_dangerous_commands = allow_dangerous_commands

    def check(self, command: str) -> CommandDecision:
        """Return the first matching policy decision for a command."""
        if self._allow_dangerous_commands:
            return CommandDecision(True)
        for pattern, reason in DANGEROUS_RULES:
            if pattern.search(command):
                return CommandDecision(False, reason)
        return CommandDecision(True)
