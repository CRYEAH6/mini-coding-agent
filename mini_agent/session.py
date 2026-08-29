"""Persistent, workspace-isolated conversation session storage."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Mapping, Optional, Sequence, Union

from mini_agent.context import ContextLimitError, ContextManager, validate_history


SESSION_VERSION = 1
MAX_SESSION_FILE_BYTES = 10_000_000
MAX_SESSION_TITLE_CHARS = 50
SESSION_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}$")
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?token|password|secret|authorization)",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(bearer\s+)[^\s\"']+"),
    re.compile(
        r"(?i)((?:DEEPSEEK_)?API_KEY[\"']?\s*[=:]\s*"
        r"[\"']?)[^\s\"',}]+"
    ),
    re.compile(
        r"(?i)((?:access[_-]?token|password|secret)[\"']?\s*[=:]\s*"
        r"[\"']?)[^\s\"',}]+"
    ),
)


class SessionError(RuntimeError):
    """Raised when a persisted session cannot be safely handled."""


@dataclass(frozen=True)
class SessionRecord:
    """One validated conversation session."""

    session_id: str
    workspace: str
    title: str
    created_at: str
    updated_at: str
    messages: list[Mapping[str, Any]]
    context_state: Mapping[str, Any]

    @property
    def turn_count(self) -> int:
        """Return the number of persisted user turns."""
        return sum(message.get("role") == "user" for message in self.messages)


@dataclass(frozen=True)
class OpenedSession:
    """Describe startup recovery and any non-fatal storage warning."""

    record: SessionRecord
    resumed: bool
    warning: Optional[str] = None


class SessionStore:
    """Persist sessions below one workspace-specific data directory."""

    def __init__(
        self,
        workspace: Union[str, Path],
        *,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        resolved = Path(workspace).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"工作目录不存在或不是目录：{resolved}")
        self._workspace = resolved
        storage_root = Path(root).expanduser() if root is not None else _default_root()
        workspace_hash = hashlib.sha256(
            str(resolved).encode("utf-8")
        ).hexdigest()[:16]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved.name) or "workspace"
        self._directory = storage_root.resolve() / f"{safe_name}-{workspace_hash}"
        self._active_path = self._directory / "active.json"

    @property
    def directory(self) -> Path:
        """Return the workspace-specific session directory."""
        return self._directory

    def open_active(self) -> OpenedSession:
        """Resume the active session, fall back safely, or create a new one."""
        warning: Optional[str] = None
        active_id: Optional[str] = None
        if self._active_path.exists():
            try:
                payload = self._read_json(self._active_path)
                active_id = self._validate_session_id(payload.get("session_id"))
            except SessionError as exc:
                warning = f"活动会话记录无法读取，已尝试恢复其他会话：{exc}"

        if active_id is not None:
            try:
                return OpenedSession(self.load(active_id), True, warning)
            except SessionError as exc:
                warning = f"上次会话无法恢复，已尝试恢复其他会话：{exc}"

        records = self.list_sessions()
        if records:
            record = records[0]
            self.set_active(record.session_id)
            return OpenedSession(record, True, warning)

        return OpenedSession(self.create(), False, warning)

    def create(self) -> SessionRecord:
        """Create, persist, and activate an empty conversation."""
        self._ensure_directory()
        while True:
            session_id = _new_session_id()
            if not self._session_path(session_id).exists():
                break
        timestamp = _utc_now()
        record = SessionRecord(
            session_id=session_id,
            workspace=str(self._workspace),
            title="新会话",
            created_at=timestamp,
            updated_at=timestamp,
            messages=[],
            context_state=_empty_context_state(),
        )
        self._write_record(record)
        self.set_active(session_id)
        return record

    def save(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        context_state: Mapping[str, Any],
    ) -> SessionRecord:
        """Atomically persist validated messages and context state."""
        current = self.load(session_id)
        copied_messages = _json_copy(messages)
        copied_context = _json_copy(context_state)
        try:
            validate_history(copied_messages)
            _validate_context_state(copied_context)
        except ContextLimitError as exc:
            raise SessionError(f"会话历史结构无效：{exc}") from exc

        sanitized_messages = _sanitize_value(copied_messages)
        sanitized_context = _sanitize_value(copied_context)
        record = SessionRecord(
            session_id=current.session_id,
            workspace=current.workspace,
            title=_derive_title(sanitized_messages),
            created_at=current.created_at,
            updated_at=_utc_now(),
            messages=sanitized_messages,
            context_state=sanitized_context,
        )
        self._write_record(record)
        return record

    def load(self, session_id: str) -> SessionRecord:
        """Load and validate one session belonging to this workspace."""
        path = self._session_path(self._validate_session_id(session_id))
        if not path.is_file():
            raise SessionError(f"会话不存在：{session_id}")
        return self._record_from_payload(self._read_json(path))

    def list_sessions(self) -> list[SessionRecord]:
        """Return valid sessions ordered from most recently updated."""
        if not self._directory.is_dir():
            return []
        records = []
        for path in self._directory.glob("*.json"):
            if path == self._active_path:
                continue
            try:
                records.append(self._record_from_payload(self._read_json(path)))
            except SessionError:
                continue
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def set_active(self, session_id: str) -> None:
        """Atomically select an existing session for the next startup."""
        session_id = self._validate_session_id(session_id)
        if not self._session_path(session_id).is_file():
            raise SessionError(f"会话不存在：{session_id}")
        self._ensure_directory()
        self._atomic_write_json(
            self._active_path,
            {"version": SESSION_VERSION, "session_id": session_id},
        )

    def delete(self, session_id: str) -> None:
        """Delete one exact session file and clear its active pointer."""
        session_id = self._validate_session_id(session_id)
        path = self._session_path(session_id)
        if not path.is_file():
            raise SessionError(f"会话不存在：{session_id}")
        try:
            path.unlink()
            if self._active_session_id() == session_id:
                self._active_path.unlink(missing_ok=True)
        except OSError as exc:
            raise SessionError(f"无法删除会话：{exc}") from exc

    def _record_from_payload(self, payload: Mapping[str, Any]) -> SessionRecord:
        if payload.get("version") != SESSION_VERSION:
            raise SessionError("会话文件版本不受支持。")
        session_id = self._validate_session_id(payload.get("session_id"))
        workspace = payload.get("workspace")
        if workspace != str(self._workspace):
            raise SessionError("会话不属于当前工作目录。")
        title = payload.get("title")
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        messages = payload.get("messages")
        context_state = payload.get("context")
        if not isinstance(title, str) or not title:
            raise SessionError("会话标题无效。")
        if not all(isinstance(value, str) for value in (created_at, updated_at)):
            raise SessionError("会话时间字段无效。")
        if not isinstance(messages, list) or not all(
            isinstance(message, Mapping) for message in messages
        ):
            raise SessionError("会话消息格式无效。")
        if not isinstance(context_state, Mapping):
            raise SessionError("会话上下文状态无效。")
        try:
            validate_history(messages)
            _validate_context_state(context_state)
        except ContextLimitError as exc:
            raise SessionError(f"会话历史结构无效：{exc}") from exc
        return SessionRecord(
            session_id=session_id,
            workspace=workspace,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            messages=list(messages),
            context_state=dict(context_state),
        )

    def _session_path(self, session_id: str) -> Path:
        return self._directory / f"{session_id}.json"

    def _active_session_id(self) -> Optional[str]:
        if not self._active_path.is_file():
            return None
        try:
            payload = self._read_json(self._active_path)
            return self._validate_session_id(payload.get("session_id"))
        except SessionError:
            return None

    def _read_json(self, path: Path) -> Mapping[str, Any]:
        try:
            if path.stat().st_size > MAX_SESSION_FILE_BYTES:
                raise SessionError(f"会话文件过大：{path.name}")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except SessionError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionError(f"无法读取会话文件 {path.name}：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise SessionError(f"会话文件不是 JSON 对象：{path.name}")
        return payload

    def _write_record(self, record: SessionRecord) -> None:
        payload = {
            "version": SESSION_VERSION,
            "session_id": record.session_id,
            "workspace": record.workspace,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "messages": record.messages,
            "context": record.context_state,
        }
        self._ensure_directory()
        self._atomic_write_json(self._session_path(record.session_id), payload)

    def _ensure_directory(self) -> None:
        try:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._directory.chmod(0o700)
        except OSError as exc:
            raise SessionError(f"无法创建会话目录：{exc}") from exc

    def _atomic_write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(serialized)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise SessionError(f"无法写入会话文件：{exc}") from exc

    @staticmethod
    def _validate_session_id(session_id: Any) -> str:
        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(
            session_id
        ):
            raise SessionError("会话 ID 格式无效。")
        return session_id


def _default_root() -> Path:
    configured = os.environ.get("MINI_AGENT_SESSION_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".mini-coding-agent" / "sessions"


def _new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _empty_context_state() -> Mapping[str, Any]:
    return {"summary_lines": [], "omitted_summary_lines": 0}


def _validate_context_state(state: Mapping[str, Any]) -> None:
    manager = ContextManager()
    manager.restore("", state)


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise SessionError(f"会话内容无法序列化：{exc}") from exc


def _derive_title(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            title = " ".join(content.split())
            if title:
                if len(title) > MAX_SESSION_TITLE_CHARS:
                    return f"{title[:MAX_SESSION_TITLE_CHARS]}…"
                return title
    return "新会话"


def _sanitize_value(value: Any, key: str = "") -> Any:
    if key and SENSITIVE_KEY_PATTERN.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        sanitized = value
        for pattern in SECRET_PATTERNS:
            if pattern.groups:
                sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
            else:
                sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_value(item_value, str(item_key))
            for item_key, item_value in value.items()
        }
    return value
