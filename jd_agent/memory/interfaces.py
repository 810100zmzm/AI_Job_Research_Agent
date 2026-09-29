"""Stable interfaces for the three memory levels."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from .models import MemoryHit, MemoryRecord


class SessionMemory(ABC):
    """L0: exact state for one session.

    The in-memory implementation is suitable for CLI runs and local Streamlit.
    A LangGraph checkpointer is exposed alongside this store so graph state can
    be resumed without changing callers.
    """

    @abstractmethod
    def append(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def history(self, session_id: str) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def clear(self, session_id: str) -> None:
        raise NotImplementedError


class ShortTermMemory(ABC):
    """L1: bounded, expiring working memory.

    This is the swap point for Redis. Callers only depend on this interface;
    replacing the MVP implementation must not require changes in agents or CLI.
    """

    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, session_id: str = "", limit: int = 5) -> List[MemoryHit]:
        raise NotImplementedError

    @abstractmethod
    def list(self, session_id: str = "") -> List[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def clear(self, session_id: str = "") -> int:
        raise NotImplementedError

    @abstractmethod
    def prune(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def count(self, session_id: str = "") -> int:
        raise NotImplementedError


class LongTermMemory(ABC):
    """L2: durable memory across sessions and processes."""

    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, user_id: str = "default", limit: int = 5) -> List[MemoryHit]:
        raise NotImplementedError

    @abstractmethod
    def all(self, user_id: str = "default") -> Sequence[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_ids: Sequence[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    def count(self, user_id: str = "default") -> int:
        raise NotImplementedError
