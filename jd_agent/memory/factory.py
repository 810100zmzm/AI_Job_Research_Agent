"""Memory backend factory.

The rest of the application imports this factory, not a concrete L1 backend.
For Redis, replace `InMemoryShortTermMemory` in `short_term.py` and keep the
constructor signature, or add a `redis` branch here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .interfaces import LongTermMemory
from .long_term import JsonlLongTermMemory, MongoLongTermMemory
from .manager import MemoryManager
from .session import InMemorySessionMemory, LangGraphSessionCheckpoints
from .short_term import InMemoryShortTermMemory


def build_memory_manager(
    data_dir: Path,
    *,
    short_term_ttl_seconds: int = 1800,
    short_term_max_items: int = 32,
    long_term_backend: str = "jsonl",
    mongodb_uri: str = "",
    mongodb_database: str = "jd_agent",
) -> MemoryManager:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    backend = (long_term_backend or "jsonl").strip().lower()
    long_term: Optional[LongTermMemory]
    if backend == "mongodb":
        long_term = MongoLongTermMemory(mongodb_uri, database=mongodb_database)
    else:
        long_term = JsonlLongTermMemory(root / "memory" / "long_term.jsonl")
    return MemoryManager(
        session=InMemorySessionMemory(),
        short_term=InMemoryShortTermMemory(
            ttl_seconds=short_term_ttl_seconds,
            max_items_per_session=short_term_max_items,
        ),
        long_term=long_term,
        checkpoints=LangGraphSessionCheckpoints(),
    )
