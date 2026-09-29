"""L0 session state and optional LangGraph checkpointer helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .interfaces import SessionMemory


class InMemorySessionMemory(SessionMemory):
    """Process-local transcript for the current session."""

    def __init__(self) -> None:
        self._messages: Dict[str, List[dict]] = defaultdict(list)

    def append(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> None:
        if not session_id:
            return
        self._messages[session_id].append(
            {
                "role": role,
                "content": str(content),
                "metadata": dict(metadata or {}),
            }
        )

    def history(self, session_id: str) -> List[dict]:
        return list(self._messages.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        self._messages.pop(session_id, None)

    def count(self) -> int:
        return len(self._messages)


class LangGraphSessionCheckpoints:
    """Lazy wrapper around LangGraph's in-process checkpointer.

    Keeping the import lazy preserves the optional-langgraph behavior of the
    rule-only modes.
    """

    def __init__(self, saver: Optional[Any] = None) -> None:
        self._saver = saver

    @property
    def saver(self) -> Any:
        if self._saver is None:
            from langgraph.checkpoint.memory import InMemorySaver

            self._saver = InMemorySaver()
        return self._saver

    def config(self, session_id: str, checkpoint_ns: str = "job-research") -> Dict[str, Any]:
        return {
            "configurable": {
                "thread_id": session_id or "default",
                "checkpoint_ns": checkpoint_ns,
            }
        }
