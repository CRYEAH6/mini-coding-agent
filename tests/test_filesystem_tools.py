"""Tests for workspace-scoped file tools."""

from pathlib import Path

from mini_agent.tools.filesystem import FileTools, MAX_FILE_CHARS


def test_write_read_and_list_files(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)

    write_result = tools.write_file("src/example.py", "answer = 42\n")
    read_result = tools.read_file("src/example.py")
    list_result = tools.list_files("src")

    assert write_result.success
    assert read_result.success
    assert read_result.content == "answer = 42\n"
    assert list_result.success
    assert list_result.content == "src/example.py"


def test_list_files_hides_internal_directories(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "visible").mkdir()

    result = FileTools(tmp_path).list_files()

    assert result.success
    assert result.content == "visible/"


def test_list_files_treats_empty_path_as_workspace(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content", encoding="utf-8")

    result = FileTools(tmp_path).list_files("")

    assert result.success
    assert result.content == "example.txt"


def test_replace_in_file_requires_unambiguous_match(tmp_path: Path) -> None:
    target = tmp_path / "values.txt"
    target.write_text("old\nold\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    ambiguous = tools.replace_in_file("values.txt", "old", "new")
    replaced = tools.replace_in_file("values.txt", "old", "new", replace_all=True)

    assert not ambiguous.success
    assert "出现 2 次" in ambiguous.content
    assert replaced.success
    assert target.read_text(encoding="utf-8") == "new\nnew\n"


def test_replace_in_file_rejects_missing_text(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("original", encoding="utf-8")

    result = FileTools(tmp_path).replace_in_file("example.txt", "missing", "new")

    assert not result.success
    assert "未找到" in result.content


def test_file_tools_reject_parent_directory_escape(tmp_path: Path) -> None:
    result = FileTools(tmp_path).read_file("../outside.txt")

    assert not result.success
    assert "超出了" in result.content


def test_file_tools_require_approval_for_external_absolute_paths(
    tmp_path: Path,
) -> None:
    result = FileTools(tmp_path).read_file("/etc/hosts")

    assert not result.success
    assert "request_workspace_access" in result.content


def test_file_tools_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-agent-file.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)

    result = FileTools(tmp_path).read_file("link.txt")

    assert not result.success
    assert "超出了" in result.content


def test_read_file_rejects_non_utf8_content(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")

    result = FileTools(tmp_path).read_file("binary.bin")

    assert not result.success
    assert "UTF-8" in result.content


def test_read_file_truncates_oversized_content(tmp_path: Path) -> None:
    content = "x" * (MAX_FILE_CHARS + 25)
    (tmp_path / "large.txt").write_text(content, encoding="utf-8")

    result = FileTools(tmp_path).read_file("large.txt")

    assert result.success
    assert result.content.startswith("x" * MAX_FILE_CHARS)
    assert "其余 25 个字符已省略" in result.content
