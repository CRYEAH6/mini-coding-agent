"""Tests for workspace-scoped long-term memory."""

import json
from pathlib import Path

import pytest

from mini_agent.memory import MemoryError, MemoryStore


def test_manual_memory_persists_and_is_workspace_isolated(tmp_path: Path) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    root = tmp_path / "memories"

    first = MemoryStore(first_workspace, root=root)
    record, created = first.remember("README 使用中文")

    assert created
    assert MemoryStore(first_workspace, root=root).list_memories() == [record]
    assert MemoryStore(second_workspace, root=root).list_memories() == []
    assert first.path.stat().st_mode & 0o777 == 0o600


def test_duplicate_memory_is_refreshed_instead_of_repeated(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")

    first, first_created = store.remember("代码必须保持简洁")
    second, second_created = store.remember("  代码必须保持简洁  ")

    assert first_created
    assert not second_created
    assert second.memory_id == first.memory_id
    assert len(store.list_memories()) == 1


def test_automatic_learning_keeps_only_explicit_durable_information(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")

    learned = store.learn_from_turn(
        "请先查看项目。README要写中文；项目使用 Python 3.9。"
        "只需确认，不要调用工具；README应该使用什么语言？"
    )

    assert [item.category for item in learned] == ["preference", "decision"]
    assert "请先查看项目" not in [item.content for item in learned]
    assert "只需确认，不要调用工具" not in [
        item.content for item in learned
    ]
    assert "README应该使用什么语言" not in [
        item.content for item in learned
    ]


def test_remember_prefix_is_removed_from_automatic_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")

    learned = store.learn_from_turn("请记住：README要写中文。")

    assert len(learned) == 1
    assert learned[0].content == "README要写中文"

    arbitrary = store.learn_from_turn("请记住：负责人代号 Aurora。")

    assert len(arbitrary) == 1
    assert arbitrary[0].category == "fact"
    assert arbitrary[0].content == "负责人代号 Aurora"


def test_retrieval_finds_chinese_and_technical_terms(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")
    store.remember("README要写中文", category="preference")
    store.remember("数据库使用 SQLite", category="decision")

    readme_matches = store.retrieve("README应该用什么语言")
    database_matches = store.retrieve("修改数据库存储")

    assert readme_matches[0].record.content == "README要写中文"
    assert database_matches[0].record.content == "数据库使用 SQLite"


def test_context_is_bounded_and_explains_precedence(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")
    store.remember("代码必须保持简洁", category="preference")

    context, count = store.build_context("请增加一个新功能", max_chars=100)

    assert count == 1
    assert len(context) <= 100
    assert "当前用户要求" in context
    assert "代码必须保持简洁" in context


def test_forget_removes_only_the_selected_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")
    first, _ = store.remember("第一条长期记忆")
    second, _ = store.remember("第二条长期记忆")

    removed = store.forget(first.memory_id)

    assert removed == first
    assert store.list_memories() == [second]
    with pytest.raises(MemoryError, match="不存在"):
        store.forget(first.memory_id)


def test_secret_like_values_are_redacted_before_storage(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")

    record, _ = store.remember("API_KEY=sk-example123456")

    assert "sk-example" not in record.content
    assert "[REDACTED]" in record.content


def test_corrupted_or_cross_workspace_file_is_rejected(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, root=tmp_path / "memories")
    store.remember("有效记忆")
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["workspace"] = str(tmp_path / "other")
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MemoryError, match="不属于当前工作目录"):
        store.list_memories()
