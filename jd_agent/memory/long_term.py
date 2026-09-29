"""L2 long-term memory backends."""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence

from ..core.text import lexical_overlap
from .interfaces import LongTermMemory
from .models import L2_LONG_TERM, MemoryHit, MemoryRecord


class JsonlLongTermMemory(LongTermMemory):
    """Durable local backend suitable for the MVP and offline development."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, MemoryRecord] = {}
        self._lock = RLock()
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self.path.is_file():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = MemoryRecord.from_dict(json.loads(line))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    self._records[record.memory_id] = record
            self._loaded = True

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._load()
        record.level = L2_LONG_TERM
        record.updated_at = time.time()
        with self._lock:
            existed = record.memory_id in self._records
            self._records[record.memory_id] = record
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
            if existed:
                self._rewrite()
        return record

    def retrieve(self, query: str, user_id: str = "default", limit: int = 5) -> List[MemoryHit]:
        self._load()
        now = time.time()
        with self._lock:
            values = [item for item in self._records.values() if item.user_id == user_id]
        hits: List[MemoryHit] = []
        for item in values:
            overlap = lexical_overlap(query, item.text)
            age_days = max(0.0, now - item.created_at) / 86400.0
            recency = 1.0 / (1.0 + age_days / 30.0)
            score = overlap * 0.7 + recency * 0.15 + max(0.0, min(1.0, item.importance)) * 0.15
            if overlap > 0 or not query.strip():
                item.touch()
                hits.append(MemoryHit(record=item, score=score, retrieval="jsonl"))
        return sorted(hits, key=lambda hit: (-hit.score, -hit.record.updated_at))[: max(1, limit)]

    def all(self, user_id: str = "default") -> Sequence[MemoryRecord]:
        self._load()
        with self._lock:
            return [item for item in self._records.values() if item.user_id == user_id]

    def delete(self, memory_ids: Sequence[str]) -> int:
        self._load()
        deleted = 0
        with self._lock:
            for memory_id in memory_ids:
                if self._records.pop(memory_id, None) is not None:
                    deleted += 1
            if deleted:
                self._rewrite()
        return deleted

    def count(self, user_id: str = "default") -> int:
        return len(self.all(user_id))

    def _rewrite(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self._records.values():
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


class MongoLongTermMemory(LongTermMemory):
    """Optional MongoDB backend; only imported when explicitly selected."""

    def __init__(self, uri: str, database: str = "jd_agent", collection: str = "memories") -> None:
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise ImportError("使用 MongoDB 长时记忆需要先安装：pip install pymongo") from exc
        self._collection = MongoClient(uri)[database][collection]
        self._collection.create_index("memory_id", unique=True)
        self._collection.create_index("user_id")

    def add(self, record: MemoryRecord) -> MemoryRecord:
        record.level = L2_LONG_TERM
        record.updated_at = time.time()
        self._collection.replace_one(
            {"memory_id": record.memory_id}, record.to_dict(), upsert=True
        )
        return record

    def retrieve(self, query: str, user_id: str = "default", limit: int = 5) -> List[MemoryHit]:
        candidates = list(self._collection.find({"user_id": user_id}))
        now = time.time()
        hits: List[MemoryHit] = []
        for payload in candidates:
            payload.pop("_id", None)
            record = MemoryRecord.from_dict(payload)
            overlap = lexical_overlap(query, record.text)
            age_days = max(0.0, now - record.created_at) / 86400.0
            score = overlap * 0.8 + (1.0 / (1.0 + age_days / 30.0)) * 0.2
            if overlap > 0 or not query.strip():
                hits.append(MemoryHit(record=record, score=score, retrieval="mongodb"))
        return sorted(hits, key=lambda hit: -hit.score)[: max(1, limit)]

    def all(self, user_id: str = "default") -> Sequence[MemoryRecord]:
        values = []
        for payload in self._collection.find({"user_id": user_id}):
            payload.pop("_id", None)
            values.append(MemoryRecord.from_dict(payload))
        return values

    def delete(self, memory_ids: Sequence[str]) -> int:
        result = self._collection.delete_many({"memory_id": {"$in": list(memory_ids)}})
        return int(result.deleted_count)

    def count(self, user_id: str = "default") -> int:
        return int(self._collection.count_documents({"user_id": user_id}))
