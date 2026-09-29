"""Memory orchestration across L0, L1, and L2."""
from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence

from .interfaces import LongTermMemory, SessionMemory, ShortTermMemory
from .models import (
    KIND_DECISION,
    KIND_EPISODIC,
    L1_SHORT_TERM,
    L2_LONG_TERM,
    MemoryHit,
    MemoryRecord,
    MemoryStats,
)
from .session import LangGraphSessionCheckpoints


class MemoryManager:
    """Coordinate session state, short-term TTL memory, and durable memory."""

    def __init__(
        self,
        session: SessionMemory,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        checkpoints: Optional[LangGraphSessionCheckpoints] = None,
        auto_consolidate_at: float = 0.75,
    ) -> None:
        self.session = session
        self.short_term = short_term
        self.long_term = long_term
        self.checkpoints = checkpoints or LangGraphSessionCheckpoints()
        self.auto_consolidate_at = max(0.0, min(1.0, float(auto_consolidate_at)))

    def remember_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        kind: str = KIND_EPISODIC,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> MemoryRecord:
        self.session.append(session_id, role, content, metadata)
        record = MemoryRecord(
            text=str(content),
            level=L1_SHORT_TERM,
            kind=kind,
            session_id=session_id,
            importance=importance,
            metadata=dict(metadata or {}),
        )
        self.short_term.add(record)
        if importance >= self.auto_consolidate_at:
            self.consolidate_record(record, source="auto")
        return record

    def remember_run(
        self,
        session_id: str,
        *,
        jd_title: str,
        project_title: str,
        decision: str,
        verdict: str,
        coverage: float,
        gaps: Sequence[str] = (),
        source_files: Sequence[str] = (),
        importance: float = 0.6,
    ) -> MemoryRecord:
        gap_text = "、".join(gaps) if gaps else "无"
        text = (
            f"{project_title} 对照 {jd_title}：{verdict or '资料不足'}；"
            f"决策 {decision}，核心覆盖 {coverage:.0%}，缺口 {gap_text}"
        )
        metadata = {
            "jd_title": jd_title,
            "project_title": project_title,
            "decision": decision,
            "verdict": verdict,
            "coverage": round(float(coverage), 4),
            "gaps": list(gaps),
            "source_files": list(source_files),
        }
        return self.remember_message(
            session_id,
            "assistant",
            text,
            kind=KIND_DECISION,
            importance=importance,
            metadata=metadata,
        )

    def retrieve(
        self,
        query: str,
        session_id: str = "",
        *,
        user_id: str = "default",
        limit: int = 6,
        include_long_term: bool = True,
    ) -> List[MemoryHit]:
        hits = self.short_term.retrieve(query, session_id=session_id, limit=limit)
        if include_long_term:
            hits.extend(self.long_term.retrieve(query, user_id=user_id, limit=limit))
        deduped = {}
        for hit in hits:
            current = deduped.get(hit.record.memory_id)
            if current is None or hit.score > current.score:
                deduped[hit.record.memory_id] = hit
        return sorted(deduped.values(), key=lambda hit: -hit.score)[: max(1, limit)]

    def consolidate(
        self,
        session_id: str,
        *,
        min_importance: float = 0.7,
        force: bool = False,
    ) -> int:
        count = 0
        for record in self.short_term.list(session_id):
            if force or record.importance >= min_importance:
                self.consolidate_record(record, source="manual")
                count += 1
        return count

    def consolidate_record(self, record: MemoryRecord, source: str = "manual") -> MemoryRecord:
        payload = MemoryRecord(
            text=record.text,
            level=L2_LONG_TERM,
            kind=record.kind,
            session_id=record.session_id,
            user_id=record.user_id,
            importance=record.importance,
            created_at=record.created_at,
            metadata={**record.metadata, "consolidated_from": source},
        )
        return self.long_term.add(payload)

    def session_history(self, session_id: str) -> List[dict]:
        return self.session.history(session_id)

    def stats(self, session_id: str = "", user_id: str = "default") -> MemoryStats:
        return MemoryStats(
            session_count=self.session.count(),
            short_term_count=self.short_term.count(session_id),
            long_term_count=self.long_term.count(user_id),
            backend=self.short_term.__class__.__name__,
        )
