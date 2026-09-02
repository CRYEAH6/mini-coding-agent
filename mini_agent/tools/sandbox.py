"""macOS system-sandbox plans for untrusted shell commands."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import tempfile
from typing import Iterator, Mapping, Sequence


STRICT_MODE = "strict"
POLICY_MODE = "policy"
SANDBOX_MODES = (STRICT_MODE, POLICY_MODE)
SANDBOX_EXECUTABLE = "/usr/bin/sandbox-exec"
SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "PASSWORD",
    "SECRET",
    "AUTHORIZATION",
    "CREDENTIAL",
)
SYSTEM_READ_PATHS = (
    "/System",
    "/usr",
    "/bin",
    "/sbin",
    "/Library",
    "/opt/homebrew",
    "/usr/local",
    "/private/etc",
    "/private/var/db",
    "/Applications/Xcode.app",
)
DEFAULT_PATH_ENTRIES = (
    "/Library/Developer/CommandLineTools/usr/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)


class SandboxError(RuntimeError):
    """Raised when a requested command sandbox cannot be prepared."""


@dataclass(frozen=True)
class ExecutionPlan:
    """Executable arguments and environment for one command."""

    arguments: Sequence[str]
    environment: Mapping[str, str]


class CommandSandbox:
    """Build strict macOS or policy-only command execution plans."""

    def __init__(self, workspace: Path, mode: str = STRICT_MODE) -> None:
        if mode not in SANDBOX_MODES:
            raise ValueError(f"未知沙箱模式：{mode}")
        self._workspace = workspace
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @contextmanager
    def prepare(
        self,
        command: str,
        *,
        allow_network: bool,
        external_read_paths: Sequence[Path] = (),
        external_write_paths: Sequence[Path] = (),
    ) -> Iterator[ExecutionPlan]:
        """Yield a command plan with only approved filesystem roots."""
        if self._mode == POLICY_MODE:
            yield ExecutionPlan(
                ("/bin/zsh", "-c", command),
                _filtered_environment(os.environ),
            )
            return

        self._require_macos_sandbox()
        with tempfile.TemporaryDirectory(
            prefix="mini-agent-command-"
        ) as temporary:
            temporary_path = Path(temporary).resolve()
            home = temporary_path / "home"
            cache = temporary_path / "cache"
            command_tmp = temporary_path / "tmp"
            for directory in (home, cache, command_tmp):
                directory.mkdir()
            profile = _build_profile(
                self._workspace,
                temporary_path,
                allow_network=allow_network,
                external_read_paths=external_read_paths,
                external_write_paths=external_write_paths,
            )
            environment = _strict_environment(
                self._workspace,
                home,
                cache,
                command_tmp,
            )
            yield ExecutionPlan(
                (
                    SANDBOX_EXECUTABLE,
                    "-p",
                    profile,
                    "/bin/zsh",
                    "-c",
                    command,
                ),
                environment,
            )

    @staticmethod
    def _require_macos_sandbox() -> None:
        if platform.system() != "Darwin":
            raise SandboxError("strict 模式目前只支持 macOS。")
        executable = Path(SANDBOX_EXECUTABLE)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise SandboxError("系统未提供可执行的 sandbox-exec。")


def _build_profile(
    workspace: Path,
    temporary: Path,
    *,
    allow_network: bool,
    external_read_paths: Sequence[Path] = (),
    external_write_paths: Sequence[Path] = (),
) -> str:
    readable_paths = [
        *SYSTEM_READ_PATHS,
        str(workspace),
        str(temporary),
        *(str(path) for path in external_read_paths),
    ]
    readable = " ".join(
        f'(subpath "{_escape_profile_path(path)}")'
        for path in readable_paths
        if Path(path).exists()
    )
    writable = " ".join(
        f'(subpath "{_escape_profile_path(path)}")'
        for path in (
            str(workspace),
            str(temporary),
            *(str(path) for path in external_write_paths),
        )
    )
    network_rule = "(allow network-outbound)" if allow_network else ""
    return " ".join(
        (
            "(version 1)",
            "(deny default)",
            '(import "system.sb")',
            "(allow process*)",
            "(allow file-read-metadata)",
            f"(allow file-read* {readable})",
            f"(allow file-write* {writable})",
            network_rule,
        )
    ).strip()


def _strict_environment(
    workspace: Path,
    home: Path,
    cache: Path,
    command_tmp: Path,
) -> Mapping[str, str]:
    path_entries = []
    workspace_venv = workspace / ".venv" / "bin"
    if workspace_venv.is_dir():
        path_entries.append(str(workspace_venv))
    path_entries.extend(
        entry for entry in DEFAULT_PATH_ENTRIES if Path(entry).is_dir()
    )
    return {
        "HOME": str(home),
        "TMPDIR": str(command_tmp),
        "XDG_CACHE_HOME": str(cache),
        "UV_CACHE_DIR": str(cache / "uv"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "npm_config_cache": str(cache / "npm"),
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
        "PATH": os.pathsep.join(path_entries),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "CI": "1",
        "NO_COLOR": "1",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
    }


def _filtered_environment(environment: Mapping[str, str]) -> Mapping[str, str]:
    """Remove credential-like variables even in policy-only mode."""
    return {
        key: value
        for key, value in environment.items()
        if not any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS)
    }


def _escape_profile_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')
