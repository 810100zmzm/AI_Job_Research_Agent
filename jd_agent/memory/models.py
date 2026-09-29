"""Memory records and common constants."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

L0_SESSION = "L0-session"
L1_SHORT_TERM = "L1-short-term"
L2_LONG_TERM = "L2-long-term"

MEMORY_LEVELS = (L0_SESSION, L1_SHORT_TERM, L2_LONG_TERM)

KIND_EPISODIC = "episodic"
KIND_SEMANTIC = "semantic"
KIND_PROFILE = "profile"
KIND_DECISION = "decision"


@dataclass
class MemoryRecord:
    """One memory item with enough metadata for TTL and retrieval scoring."""

    text: str
    level: str = L1_SHORT_TERM
    kind: str = KIND_EPISODIC
    session_id: str = ""
    user_id: str = "default"
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    access_count: int = 0
    last_access_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return bool(self.expires_at and self.expires_at <= time.time())

    def touch(self) -> None:
        self.access_count += 1
        self.last_access_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "level": self.level,
            "kind": self.kind,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_access_at": self.last_access_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryRecord":
        values = dict(payload)
        values["metadata"] = dict(values.get("metadata") or {})
        return cls(**values)


@dataclass
class MemoryHit:
    record: MemoryRecord
    score: float
    retrieval: str = ""


@dataclass
class MemoryStats:
    session_count: int = 0
    short_term_count: int = 0
    long_term_count: int = 0
    backend: str = "in-memory"

    def lines(self) -> List[str]:
        return [
            f"L0 会话数：{self.session_count}",
            f"L1 短时记忆：{self.short_term_count}",
            f"L2 长时记忆：{self.long_term_count}",
            f"L1 后端：{self.backend}",
        ]
