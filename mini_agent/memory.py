"""Workspace-scoped long-term memory with local similarity retrieval."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
from typing import Any, Mapping, Optional, Sequence, Union


MEMORY_VERSION = 1
MAX_MEMORY_FILE_BYTES = 2_000_000
MAX_MEMORY_ITEMS = 500
MAX_MEMORY_CHARS = 500
DEFAULT_RETRIEVAL_LIMIT = 5
DEFAULT_CONTEXT_CHARS = 2_000
MEMORY_ID_PATTERN = re.compile(r"^mem-[0-9a-f]{8}$")

PREFERENCE_PATTERN = re.compile(
    r"(?:我(?:希望|喜欢|偏好|习惯)|请(?:始终|一直|默认)|"
    r"以后(?:都|请)|默认(?:使用|采用)|代码(?:始终|必须|要)|"
    r"README\s*(?:始终|必须|要))",
    re.IGNORECASE,
)
DECISION_PATTERN = re.compile(
    r"(?:(?:项目|本项目|技术栈|运行环境).{0,30}"
    r"(?:使用|采用|选择|要求|版本|是|为)|"
    r"(?:Python|模型|README|测试命令|启动命令)\s*"
    r"(?:使用|采用|选择|要求|版本|是|为))",
    re.IGNORECASE,
)
FACT_PATTERN = re.compile(
    r"(?:入口文件|工作目录|仓库|测试命令|启动命令|API|"
    r"模型|主模块)"
    r".{0,50}(?:是|为|使用|位于|放在)",
    re.IGNORECASE,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?；;\n]+")
QUESTION_PATTERN = re.compile(r"(?:什么|是否|怎么|如何|哪(?:个|种|些)?|为何|为什么)")
REMEMBER_PREFIX_PATTERN = re.compile(r"^请记住\s*[:：]?\s*")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:\.\d+)*")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
SPACE_PATTERN = re.compile(r"\s+")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(bearer\s+)[^\s\"']+"),
    re.compile(
        r"(?i)((?:[A-Z0-9]+_)*API_KEY[\"']?\s*[=:]\s*"
        r"[\"']?)[^\s\"',}]+"
    ),
    re.compile(
        r"(?i)((?:access[_-]?token|password|secret)[\"']?\s*[=:]\s*"
        r"[\"']?)[^\s\"',}]+"
    ),
)

CATEGORY_LABELS = {
    "preference": "用户偏好",
    "decision": "项目决策",
    "fact": "项目事实",
    "manual": "手动记忆",
}


class MemoryError(RuntimeError):
    """Raised when long-term memory cannot be safely persisted or loaded."""


@dataclass(frozen=True)
class MemoryRecord:
    """One durable memory item."""

    memory_id: str
    category: str
    content: str
    source: str
    created_at: str
    updated_at: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class MemoryMatch:
    """One retrieved memory and its relevance score."""

    record: MemoryRecord
    score: float


class MemoryStore:
    """Persist and retrieve long-term memories for one workspace."""

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
        storage_root = (
            Path(root).expanduser() if root is not None else _default_root()
        )
        workspace_hash = hashlib.sha256(
            str(resolved).encode("utf-8")
        ).hexdigest()[:16]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved.name) or "workspace"
        self._directory = storage_root.resolve() / f"{safe_name}-{workspace_hash}"
        self._path = self._directory / "memory.json"

    @property
    def path(self) -> Path:
        """Return the memory file path for diagnostics and tests."""
        return self._path

    def list_memories(self) -> list[MemoryRecord]:
        """Return newest memories first."""
        return sorted(
            self._load(),
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def remember(
        self,
        content: str,
        *,
        category: str = "manual",
        source: str = "manual",
    ) -> tuple[MemoryRecord, bool]:
        """Store one sanitized memory or refresh its existing duplicate."""
        normalized = _normalize_content(content)
        if category not in CATEGORY_LABELS:
            raise ValueError(f"未知记忆类别：{category}")
        if source not in {"manual", "automatic"}:
            raise ValueError(f"未知记忆来源：{source}")
        records = self._load()
        timestamp = _utc_now()

        for index, existing in enumerate(records):
            if _is_duplicate(normalized, existing.content):
                refreshed_category = (
                    category if source == "manual" else existing.category
                )
                refreshed_source = (
                    "manual" if source == "manual" else existing.source
                )
                refreshed = MemoryRecord(
                    memory_id=existing.memory_id,
                    category=refreshed_category,
                    content=existing.content,
                    source=refreshed_source,
                    created_at=existing.created_at,
                    updated_at=timestamp,
                    keywords=existing.keywords,
                )
                records[index] = refreshed
                self._save(records)
                return refreshed, False

        if len(records) >= MAX_MEMORY_ITEMS:
            records.sort(key=lambda item: item.updated_at)
            records.pop(0)
        record = MemoryRecord(
            memory_id=f"mem-{secrets.token_hex(4)}",
            category=category,
            content=normalized,
            source=source,
            created_at=timestamp,
            updated_at=timestamp,
            keywords=tuple(sorted(_tokenize(normalized))),
        )
        records.append(record)
        self._save(records)
        return record, True

    def learn_from_turn(self, user_message: str) -> list[MemoryRecord]:
        """Extract explicit durable facts from a successfully completed turn."""
        learned = []
        for category, content in _extract_candidates(user_message):
            record, created = self.remember(
                content,
                category=category,
                source="automatic",
            )
            if created:
                learned.append(record)
        return learned

    def retrieve(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ) -> list[MemoryMatch]:
        """Return relevant memories using lexical-semantic similarity."""
        if limit <= 0:
            return []
        query_tokens = _tokenize(query)
        matches = []
        for record in self._load():
            score = _relevance(query, query_tokens, record)
            if score >= _minimum_score(record.category):
                matches.append(MemoryMatch(record, score))
        matches.sort(
            key=lambda item: (item.score, item.record.updated_at),
            reverse=True,
        )
        return matches[:limit]

    def build_context(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        max_chars: int = DEFAULT_CONTEXT_CHARS,
    ) -> tuple[str, int]:
        """Format relevant memories as bounded, unambiguous model context."""
        if max_chars <= 0:
            return "", 0
        prefix = (
            "以下是从本工作目录长期记忆中检索出的背景。它们可能过期；"
            "当前用户要求和真实文件内容始终优先。"
            "不要把记忆当作已验证的执行结果。\n"
        )
        if len(prefix) >= max_chars:
            return "", 0
        lines = []
        for match in self.retrieve(query, limit=limit):
            label = CATEGORY_LABELS[match.record.category]
            line = f"- [{label}] {match.record.content}"
            candidate = prefix + "\n".join([*lines, line])
            if len(candidate) > max_chars:
                break
            lines.append(line)
        if not lines:
            return "", 0
        context = prefix + "\n".join(lines)
        return context, len(lines)

    def forget(self, memory_id: str) -> MemoryRecord:
        """Delete one exact memory by ID."""
        if not MEMORY_ID_PATTERN.fullmatch(memory_id):
            raise MemoryError("记忆 ID 格式无效。")
        records = self._load()
        for index, record in enumerate(records):
            if record.memory_id == memory_id:
                removed = records.pop(index)
                self._save(records)
                return removed
        raise MemoryError(f"记忆不存在：{memory_id}")

    def _load(self) -> list[MemoryRecord]:
        if not self._path.exists():
            return []
        try:
            if self._path.stat().st_size > MAX_MEMORY_FILE_BYTES:
                raise MemoryError("长期记忆文件过大。")
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except MemoryError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MemoryError(f"无法读取长期记忆：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise MemoryError("长期记忆文件不是 JSON 对象。")
        if payload.get("version") != MEMORY_VERSION:
            raise MemoryError("长期记忆文件版本不受支持。")
        if payload.get("workspace") != str(self._workspace):
            raise MemoryError("长期记忆不属于当前工作目录。")
        items = payload.get("memories")
        if not isinstance(items, list) or len(items) > MAX_MEMORY_ITEMS:
            raise MemoryError("长期记忆条目格式无效。")
        return [_record_from_payload(item) for item in items]

    def _save(self, records: Sequence[MemoryRecord]) -> None:
        payload = {
            "version": MEMORY_VERSION,
            "workspace": str(self._workspace),
            "memories": [_record_to_payload(record) for record in records],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if len(serialized.encode("utf-8")) > MAX_MEMORY_FILE_BYTES:
            raise MemoryError("长期记忆文件超过大小限制。")
        temporary = self._path.with_name(
            f".{self._path.name}.{secrets.token_hex(4)}.tmp"
        )
        try:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._directory.chmod(0o700)
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(serialized)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise MemoryError(f"无法写入长期记忆：{exc}") from exc


def _extract_candidates(user_message: str) -> list[tuple[str, str]]:
    candidates = []
    seen = set()
    for raw_sentence in SENTENCE_SPLIT_PATTERN.split(user_message):
        sentence = raw_sentence.strip(" \t-—•")
        explicitly_requested = bool(REMEMBER_PREFIX_PATTERN.match(sentence))
        sentence = REMEMBER_PREFIX_PATTERN.sub("", sentence)
        if not 4 <= len(sentence) <= MAX_MEMORY_CHARS:
            continue
        category = _classify(sentence)
        if category is None and explicitly_requested:
            category = "fact"
        normalized = _normalize_content(sentence)
        identity = normalized.casefold()
        if category is not None and identity not in seen:
            candidates.append((category, normalized))
            seen.add(identity)
        if len(candidates) == 3:
            break
    return candidates


def _classify(sentence: str) -> Optional[str]:
    if QUESTION_PATTERN.search(sentence):
        return None
    if PREFERENCE_PATTERN.search(sentence):
        return "preference"
    if DECISION_PATTERN.search(sentence):
        return "decision"
    if FACT_PATTERN.search(sentence):
        return "fact"
    return None


def _normalize_content(content: str) -> str:
    if not isinstance(content, str):
        raise ValueError("记忆内容必须是字符串。")
    sanitized = content
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub(_redact_secret, sanitized)
    normalized = SPACE_PATTERN.sub(" ", sanitized).strip()
    if not normalized:
        raise ValueError("记忆内容不能为空。")
    if len(normalized) > MAX_MEMORY_CHARS:
        raise ValueError(f"单条记忆不能超过 {MAX_MEMORY_CHARS} 个字符。")
    return normalized


def _redact_secret(match: re.Match[str]) -> str:
    prefix = match.group(1) if match.lastindex else ""
    return f"{prefix}[REDACTED]"


def _tokenize(text: str) -> set[str]:
    normalized = text.casefold()
    tokens = {match.group(0) for match in WORD_PATTERN.finditer(normalized)}
    for block in CHINESE_PATTERN.findall(normalized):
        tokens.add(block)
        if len(block) == 1:
            tokens.add(block)
        else:
            tokens.update(
                block[index : index + 2]
                for index in range(len(block) - 1)
            )
    return tokens


def _relevance(query: str, query_tokens: set[str], record: MemoryRecord) -> float:
    memory_tokens = set(record.keywords)
    if not query_tokens or not memory_tokens:
        similarity = 0.0
    else:
        intersection = len(query_tokens & memory_tokens)
        cosine = intersection / math.sqrt(len(query_tokens) * len(memory_tokens))
        jaccard = intersection / len(query_tokens | memory_tokens)
        similarity = 0.7 * cosine + 0.3 * jaccard

    normalized_query = SPACE_PATTERN.sub(" ", query.casefold()).strip()
    normalized_memory = record.content.casefold()
    if normalized_query and (
        normalized_query in normalized_memory or normalized_memory in normalized_query
    ):
        similarity += 0.35
    if record.category == "preference":
        similarity += 0.16
    elif record.category == "manual":
        similarity += 0.08
    return min(similarity, 1.0)


def _minimum_score(category: str) -> float:
    if category == "preference":
        return 0.16
    return 0.09


def _is_duplicate(left: str, right: str) -> bool:
    if left.casefold() == right.casefold():
        return True
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return overlap >= 0.9


def _record_from_payload(payload: Any) -> MemoryRecord:
    if not isinstance(payload, Mapping):
        raise MemoryError("长期记忆条目不是 JSON 对象。")
    memory_id = payload.get("id")
    category = payload.get("category")
    content = payload.get("content")
    source = payload.get("source")
    created_at = payload.get("created_at")
    updated_at = payload.get("updated_at")
    keywords = payload.get("keywords")
    if not isinstance(memory_id, str) or not MEMORY_ID_PATTERN.fullmatch(memory_id):
        raise MemoryError("长期记忆 ID 无效。")
    if category not in CATEGORY_LABELS:
        raise MemoryError("长期记忆类别无效。")
    if not isinstance(content, str) or not 0 < len(content) <= MAX_MEMORY_CHARS:
        raise MemoryError("长期记忆内容无效。")
    if source not in {"manual", "automatic"}:
        raise MemoryError("长期记忆来源无效。")
    if not isinstance(created_at, str) or not isinstance(updated_at, str):
        raise MemoryError("长期记忆时间字段无效。")
    if not isinstance(keywords, list) or not all(
        isinstance(keyword, str) for keyword in keywords
    ):
        raise MemoryError("长期记忆关键词无效。")
    return MemoryRecord(
        memory_id=memory_id,
        category=category,
        content=content,
        source=source,
        created_at=created_at,
        updated_at=updated_at,
        keywords=tuple(keywords),
    )


def _record_to_payload(record: MemoryRecord) -> Mapping[str, Any]:
    return {
        "id": record.memory_id,
        "category": record.category,
        "content": record.content,
        "source": record.source,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "keywords": list(record.keywords),
    }


def _default_root() -> Path:
    configured = os.environ.get("MINI_AGENT_MEMORY_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".mini-coding-agent" / "memories"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
