"""Knowledge-base models and tier constants."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

L1_STATIC = "L1-static"
L2_SEMI_STATIC = "L2-semi-static"
L3_DYNAMIC = "L3-dynamic"
KNOWLEDGE_TIERS = (L1_STATIC, L2_SEMI_STATIC, L3_DYNAMIC)
TIER_LABELS = {
    L1_STATIC: "L1 静态知识",
    L2_SEMI_STATIC: "L2 半静态知识",
    L3_DYNAMIC: "L3 动态知识",
}


@dataclass
class KnowledgeDocument:
    text: str
    tier: str
    title: str
    source: str = ""
    document_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


@dataclass
class KnowledgeChunk:
    text: str
    tier: str
    title: str
    source: str = ""
    document_id: str = ""
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    heading: str = ""
    ordinal: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "tier": self.tier,
            "title": self.title,
            "source": self.source,
            "heading": self.heading,
            "ordinal": self.ordinal,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "KnowledgeChunk":
        values = dict(payload)
        values["tags"] = list(values.get("tags") or [])
        values["metadata"] = dict(values.get("metadata") or {})
        return cls(**values)


@dataclass
class KnowledgeHit:
    chunk: KnowledgeChunk
    score: float
    retrieval: str = ""


@dataclass
class KnowledgeStats:
    chunk_count: int = 0
    tier_counts: Dict[str, int] = field(default_factory=dict)
    embedding_backend: str = ""
    index_backend: str = ""

    def lines(self) -> List[str]:
        rows = [f"知识库切片：{self.chunk_count}", f"嵌入后端：{self.embedding_backend}", f"索引后端：{self.index_backend}"]
        rows.extend(
            f"{TIER_LABELS.get(tier, tier)}：{self.tier_counts.get(tier, 0)}"
            for tier in KNOWLEDGE_TIERS
        )
        return rows
