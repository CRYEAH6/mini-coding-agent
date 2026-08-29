"""Tests for non-destructive Git inspection and checkpoint tools."""

from pathlib import Path
import subprocess
from unittest.mock import Mock

from mini_agent.tools.git import GitTools


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _initialize_repository(repository: Path) -> None:
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "app.py").write_text("value = 1\n", encoding="utf-8")
    _git(repository, "add", "app.py")
    _git(repository, "commit", "-q", "-m", "initial")


def test_git_status_reports_branch_and_clean_worktree(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)

    result = GitTools(tmp_path).git_status()

    assert result.success
    assert "##" in result.content
    assert "尚未创建 Agent 检查点" in result.content


def test_git_tools_reject_non_repository(tmp_path: Path) -> None:
    result = GitTools(tmp_path).git_status()

    assert not result.success
    assert "Git 操作失败" in result.content


def test_git_tools_require_workspace_to_be_repository_root(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    child = tmp_path / "src"
    child.mkdir()

    result = GitTools(child).git_status()

    assert not result.success
    assert "仓库根目录" in result.content


def test_git_diff_includes_staged_unstaged_and_untracked_files(
    tmp_path: Path,
) -> None:
    _initialize_repository(tmp_path)
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    (tmp_path / "app.py").write_text("value = 3\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("created = True\n", encoding="utf-8")

    result = GitTools(tmp_path).git_diff()

    assert result.success
    assert "+value = 3" in result.content
    assert "new.py" in result.content
    assert "+created = True" in result.content


def test_git_diff_rejects_workspace_escape(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)

    result = GitTools(tmp_path).git_diff(path="../outside")

    assert not result.success
    assert "超出了" in result.content


def test_git_diff_rejects_invalid_revision(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)

    result = GitTools(tmp_path).git_diff(base="--output=/tmp/example")

    assert not result.success
    assert "引用格式" in result.content


def test_checkpoint_requires_user_approval(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")

    result = GitTools(tmp_path).git_checkpoint("before refactor")

    assert not result.success
    assert "未提供确认入口" in result.content


def test_denied_checkpoint_does_not_create_ref(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    approval_handler = Mock(return_value=False)

    result = GitTools(
        tmp_path,
        approval_handler=approval_handler,
    ).git_checkpoint("before refactor")

    assert not result.success
    assert "用户拒绝" in result.content
    assert _git(tmp_path, "for-each-ref", "refs/mini-agent/checkpoints") == ""
    request = approval_handler.call_args.args[0]
    assert request.tool_name == "git_checkpoint"
    assert "app.py" in request.details


def test_checkpoint_preserves_head_index_and_worktree(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    (tmp_path / "app.py").write_text("value = 3\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("created = True\n", encoding="utf-8")
    head_before = _git(tmp_path, "rev-parse", "HEAD")
    staged_before = _git(tmp_path, "diff", "--cached")
    unstaged_before = _git(tmp_path, "diff")
    approval_handler = Mock(return_value=True)

    result = GitTools(
        tmp_path,
        approval_handler=approval_handler,
    ).git_checkpoint("before refactor")

    assert result.success
    assert "当前分支、暂存区和工作文件均未改变" in result.content
    assert _git(tmp_path, "rev-parse", "HEAD") == head_before
    assert _git(tmp_path, "diff", "--cached") == staged_before
    assert _git(tmp_path, "diff") == unstaged_before
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "value = 3\n"
    checkpoint = next(
        line.removeprefix("引用：")
        for line in result.content.splitlines()
        if line.startswith("引用：")
    )
    assert _git(tmp_path, "show", f"{checkpoint}:app.py") == "value = 3"
    assert _git(tmp_path, "show", f"{checkpoint}:new.py") == "created = True"


def test_git_diff_can_compare_with_checkpoint(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    tools = GitTools(tmp_path, approval_handler=Mock(return_value=True))
    checkpoint_result = tools.git_checkpoint("working version")
    checkpoint = next(
        line.removeprefix("引用：")
        for line in checkpoint_result.content.splitlines()
        if line.startswith("引用：")
    )
    (tmp_path / "app.py").write_text("value = 3\n", encoding="utf-8")

    diff_result = tools.git_diff(base=checkpoint)
    status_result = tools.git_status()

    assert diff_result.success
    assert "-value = 2" in diff_result.content
    assert "+value = 3" in diff_result.content
    assert checkpoint in status_result.content


def test_checkpoint_rejects_clean_worktree_without_prompt(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    approval_handler = Mock(return_value=True)

    result = GitTools(
        tmp_path,
        approval_handler=approval_handler,
    ).git_checkpoint("nothing changed")

    assert not result.success
    assert "没有需要保存" in result.content
    approval_handler.assert_not_called()


def test_checkpoint_supports_repository_without_initial_commit(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "first.py").write_text("ready = True\n", encoding="utf-8")
    tools = GitTools(tmp_path, approval_handler=Mock(return_value=True))

    result = tools.git_checkpoint("initial workspace")

    assert result.success
    checkpoint = next(
        line.removeprefix("引用：")
        for line in result.content.splitlines()
        if line.startswith("引用：")
    )
    assert _git(tmp_path, "show", f"{checkpoint}:first.py") == "ready = True"
