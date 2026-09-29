"""MVP L1 backend: in-process memory with TTL and capacity limits.

To switch to Redis later, replace this file with an implementation of the same
`ShortTermMemory` methods. No agent, CLI, or Streamlit code needs to change.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List

from ..core.text import lexical_overlap
from .interfaces import ShortTermMemory
from .models import L1_SHORT_TERM, MemoryHit, MemoryRecord


class InMemoryShortTermMemory(ShortTermMemory):
    def __init__(self, ttl_seconds: int = 1800, max_items_per_session: int = 32) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_items_per_session = max(1, int(max_items_per_session))
        self._items: Dict[str, MemoryRecord] = {}
        self._lock = threading.RLock()

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            record.level = L1_SHORT_TERM
            record.updated_at = time.time()
            if not record.expires_at:
                record.expires_at = record.updated_at + self.ttl_seconds
            self._items[record.memory_id] = record
            self._trim_session(record.session_id)
            return record

    def retrieve(self, query: str, session_id: str = "", limit: int = 5) -> List[MemoryHit]:
        self.prune()
        now = time.time()
        hits: List[MemoryHit] = []
        with self._lock:
            values = [
                item
                for item in self._items.values()
                if not session_id or item.session_id == session_id
            ]
        for item in values:
            overlap = lexical_overlap(query, item.text)
            age = max(0.0, now - item.created_at)
            recency = 1.0 / (1.0 + age / max(1.0, self.ttl_seconds))
            score = overlap * 0.7 + recency * 0.2 + max(0.0, min(1.0, item.importance)) * 0.1
            if overlap > 0 or not query.strip():
                item.touch()
                hits.append(MemoryHit(record=item, score=score, retrieval="in-memory"))
        return sorted(hits, key=lambda hit: (-hit.score, -hit.record.created_at))[: max(1, limit)]

    def list(self, session_id: str = "") -> List[MemoryRecord]:
        self.prune()
        with self._lock:
            values = [
                item
                for item in self._items.values()
                if not session_id or item.session_id == session_id
            ]
        return sorted(values, key=lambda item: item.created_at, reverse=True)

    def clear(self, session_id: str = "") -> int:
        with self._lock:
            if not session_id:
                count = len(self._items)
                self._items.clear()
                return count
            keys = [key for key, item in self._items.items() if item.session_id == session_id]
            for key in keys:
                self._items.pop(key, None)
            return len(keys)

    def prune(self) -> int:
        now = time.time()
        with self._lock:
            expired = [key for key, item in self._items.items() if item.expires_at <= now]
            for key in expired:
                self._items.pop(key, None)
        removed = len(expired)
        with self._lock:
            session_ids = {item.session_id for item in self._items.values() if item.session_id}
        for session_id in session_ids:
            removed += self._trim_session(session_id)
        return removed

    def count(self, session_id: str = "") -> int:
        self.prune()
        with self._lock:
            return len(
                [
                    item
                    for item in self._items.values()
                    if not session_id or item.session_id == session_id
                ]
            )

    def _trim_session(self, session_id: str) -> int:
        if not session_id:
            return 0
        with self._lock:
            values = sorted(
                (item for item in self._items.values() if item.session_id == session_id),
                key=lambda item: (item.importance, item.updated_at),
            )
            overflow = len(values) - self.max_items_per_session
            for item in values[: max(0, overflow)]:
                self._items.pop(item.memory_id, None)
        return max(0, overflow)
