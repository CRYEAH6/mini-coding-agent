"""Tests for persistent workspace-isolated conversation sessions."""

import json
from pathlib import Path
import stat

import pytest

from mini_agent.session import SessionError, SessionStore


EMPTY_CONTEXT = {"summary_lines": [], "omitted_summary_lines": 0}


def _history(user: str, assistant: str = "完成") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def test_store_creates_and_resumes_active_session(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    root = tmp_path / "sessions"
    store = SessionStore(workspace, root=root)

    opened = store.open_active()
    saved = store.save(
        opened.record.session_id,
        _history("实现飞机大战"),
        EMPTY_CONTEXT,
    )
    reopened = SessionStore(workspace, root=root).open_active()

    assert not opened.resumed
    assert reopened.resumed
    assert reopened.record.session_id == saved.session_id
    assert reopened.record.title == "实现飞机大战"
    assert reopened.record.turn_count == 1


def test_sessions_are_isolated_by_workspace(tmp_path: Path) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    root = tmp_path / "sessions"

    first = SessionStore(first_workspace, root=root)
    second = SessionStore(second_workspace, root=root)
    first_record = first.create()
    second_record = second.create()

    assert first.directory != second.directory
    assert first.list_sessions() == [first_record]
    assert second.list_sessions() == [second_record]


def test_session_files_use_restricted_permissions(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")

    record = store.create()
    session_path = store.directory / f"{record.session_id}.json"

    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_path.stat().st_mode) == 0o600
    assert not list(store.directory.glob("*.tmp"))


def test_save_redacts_common_secrets(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    record = store.create()
    messages = _history(
        "LLM_API_KEY=secret-value sk-1234567890",
        "Bearer private-token",
    )
    context = {
        "summary_lines": [],
        "omitted_summary_lines": 0,
        "LLM_API_KEY": "nested-secret",
    }

    saved = store.save(record.session_id, messages, context)
    session_path = store.directory / f"{record.session_id}.json"
    serialized = session_path.read_text(encoding="utf-8")

    assert "secret-value" not in serialized
    assert "sk-1234567890" not in serialized
    assert "private-token" not in serialized
    assert "nested-secret" not in serialized
    assert "[REDACTED]" in serialized
    assert "[REDACTED]" in saved.title


def test_save_redacts_secret_inside_tool_argument_string(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    record = store.create()
    messages = [
        {"role": "user", "content": "执行操作"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "example",
                        "arguments": '{"api_key":"plain-secret"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"success":true,"content":"done"}',
        },
        {"role": "assistant", "content": "完成"},
    ]

    store.save(record.session_id, messages, EMPTY_CONTEXT)
    serialized = (
        store.directory / f"{record.session_id}.json"
    ).read_text(encoding="utf-8")

    assert "plain-secret" not in serialized
    assert "[REDACTED]" in serialized


def test_save_rejects_incomplete_tool_history(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    record = store.create()
    messages = [
        {"role": "user", "content": "读取文件"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
    ]

    with pytest.raises(SessionError, match="工具结果不完整"):
        store.save(record.session_id, messages, EMPTY_CONTEXT)


def test_corrupt_active_session_falls_back_to_valid_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    record = store.create()
    store.save(record.session_id, _history("保留会话"), EMPTY_CONTEXT)
    (store.directory / "active.json").write_text(
        '{"session_id": "invalid"}',
        encoding="utf-8",
    )

    opened = store.open_active()

    assert opened.resumed
    assert opened.record.session_id == record.session_id
    assert opened.warning is not None
    assert "无法读取" in opened.warning


def test_list_skips_corrupt_session_file(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    valid = store.create()
    (store.directory / "broken.json").write_text("not-json", encoding="utf-8")

    records = store.list_sessions()

    assert [record.session_id for record in records] == [valid.session_id]


def test_sessions_are_ordered_by_recent_update(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    first = store.create()
    second = store.create()
    updated_first = store.save(
        first.session_id,
        _history("最近更新"),
        EMPTY_CONTEXT,
    )

    records = store.list_sessions()

    assert records[0].session_id == updated_first.session_id
    assert records[1].session_id == second.session_id


def test_delete_removes_only_selected_session(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    first = store.create()
    second = store.create()

    store.delete(first.session_id)

    assert [record.session_id for record in store.list_sessions()] == [
        second.session_id
    ]
    with pytest.raises(SessionError, match="不存在"):
        store.load(first.session_id)


def test_session_id_cannot_escape_storage_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")

    with pytest.raises(SessionError, match="ID 格式"):
        store.load("../../outside")


def test_session_payload_records_workspace_and_version(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    store = SessionStore(workspace, root=tmp_path / "sessions")
    record = store.create()

    payload = json.loads(
        (store.directory / f"{record.session_id}.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["version"] == 1
    assert payload["workspace"] == str(workspace.resolve())
    assert "system" not in {message.get("role") for message in payload["messages"]}
