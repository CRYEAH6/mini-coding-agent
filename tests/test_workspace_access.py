"""Tests for temporary access to projects outside the main workspace."""

from pathlib import Path
from unittest.mock import Mock

from mini_agent.tools.access import WorkspaceAccessManager
from mini_agent.tools.filesystem import FileTools


def _directories(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "main"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    return workspace, external


def test_external_access_denial_keeps_project_blocked(tmp_path: Path) -> None:
    workspace, external = _directories(tmp_path)
    approval_handler = Mock(return_value=False)
    manager = WorkspaceAccessManager(workspace, approval_handler)

    result = manager.request_access(
        str(external),
        "read",
        "需要读取共享配置。",
    )

    assert not result.success
    assert "用户拒绝" in result.content
    assert not manager.can_access(external / "config.json", write=False)


def test_read_grant_is_cached_but_does_not_allow_writes(tmp_path: Path) -> None:
    workspace, external = _directories(tmp_path)
    approval_handler = Mock(return_value=True)
    manager = WorkspaceAccessManager(workspace, approval_handler)

    first = manager.request_access(str(external), "read", "读取配置。")
    second = manager.request_access(str(external), "read", "再次读取配置。")

    assert first.success
    assert second.success
    assert manager.can_access(external / "config.json", write=False)
    assert not manager.can_access(external / "config.json", write=True)
    approval_handler.assert_called_once()


def test_read_grant_can_be_upgraded_to_read_write(tmp_path: Path) -> None:
    workspace, external = _directories(tmp_path)
    approval_handler = Mock(return_value=True)
    manager = WorkspaceAccessManager(workspace, approval_handler)

    manager.request_access(str(external), "read", "读取配置。")
    upgraded = manager.request_access(
        str(external),
        "read_write",
        "修改共享配置。",
    )

    assert upgraded.success
    assert manager.can_access(external / "config.json", write=True)
    assert approval_handler.call_count == 2
    assert "读取和修改" in approval_handler.call_args.args[0].details


def test_file_tools_use_approved_external_directory(tmp_path: Path) -> None:
    workspace, external = _directories(tmp_path)
    external_file = external / "shared.txt"
    external_file.write_text("before", encoding="utf-8")
    manager = WorkspaceAccessManager(workspace, Mock(return_value=True))
    tools = FileTools(workspace, manager)

    manager.request_access(str(external), "read", "读取共享文件。")
    read = tools.read_file(str(external_file))
    denied_write = tools.write_file(str(external_file), "blocked")
    manager.request_access(str(external), "read_write", "修改共享文件。")
    allowed_write = tools.write_file(str(external_file), "after")

    assert read.success
    assert read.content == "before"
    assert not denied_write.success
    assert "read_write" in denied_write.content
    assert allowed_write.success
    assert external_file.read_text(encoding="utf-8") == "after"


def test_relative_symlink_cannot_bypass_primary_workspace(
    tmp_path: Path,
) -> None:
    workspace, external = _directories(tmp_path)
    external_file = external / "secret.txt"
    external_file.write_text("secret", encoding="utf-8")
    manager = WorkspaceAccessManager(workspace, Mock(return_value=True))
    manager.request_access(str(external), "read", "读取外部项目。")
    (workspace / "link.txt").symlink_to(external_file)

    result = FileTools(workspace, manager).read_file("link.txt")

    assert not result.success
    assert "超出了" in result.content


def test_filesystem_root_cannot_be_granted(tmp_path: Path) -> None:
    workspace = tmp_path / "main"
    workspace.mkdir()
    approval_handler = Mock(return_value=True)
    manager = WorkspaceAccessManager(workspace, approval_handler)

    result = manager.request_access("/", "read", "读取所有文件。")

    assert not result.success
    assert "根目录" in result.content
    approval_handler.assert_not_called()
