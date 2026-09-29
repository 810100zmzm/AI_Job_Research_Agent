"""Durable local vector index used when Qdrant is not configured."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Dict, List, Sequence, Tuple

from ..core.text import cosine, lexical_overlap
from .interfaces import VectorIndex
from .models import KnowledgeChunk, KnowledgeHit


class JsonlVectorIndex(VectorIndex):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: Dict[str, Tuple[KnowledgeChunk, List[float]]] = {}
        self._lock = RLock()
        self._loaded = False

    @property
    def name(self) -> str:
        return "jsonl"

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
                        payload = json.loads(line)
                        vector = [float(value) for value in payload.get("vector") or []]
                        payload.pop("vector", None)
                        chunk = KnowledgeChunk.from_dict(payload)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if vector:
                        self._rows[chunk.chunk_id] = (chunk, vector)
            self._loaded = True

    def upsert(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> int:
        self._load()
        changed = 0
        with self._lock:
            for chunk, vector in zip(chunks, vectors):
                if not vector:
                    continue
                self._rows[chunk.chunk_id] = (chunk, [float(value) for value in vector])
                changed += 1
            if changed:
                self._rewrite()
        return changed

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        tiers: Sequence[str] = (),
        limit: int = 5,
    ) -> List[KnowledgeHit]:
        self._load()
        allowed = set(tiers or ())
        with self._lock:
            values = list(self._rows.values())
        hits: List[KnowledgeHit] = []
        for chunk, vector in values:
            if allowed and chunk.tier not in allowed:
                continue
            semantic = max(0.0, cosine(query_vector, vector))
            lexical = lexical_overlap(query, f"{chunk.title} {chunk.heading} {chunk.text}")
            score = semantic * 0.65 + lexical * 0.35
            if score > 0:
                hits.append(KnowledgeHit(chunk=chunk, score=score, retrieval="jsonl"))
        return sorted(hits, key=lambda hit: -hit.score)[: max(1, limit)]

    def list(self, tiers: Sequence[str] = ()) -> List[KnowledgeChunk]:
        self._load()
        allowed = set(tiers or ())
        with self._lock:
            return [
                chunk
                for chunk, _ in self._rows.values()
                if not allowed or chunk.tier in allowed
            ]

    def delete_document(self, document_id: str) -> int:
        self._load()
        with self._lock:
            keys = [key for key, (chunk, _) in self._rows.items() if chunk.document_id == document_id]
            for key in keys:
                self._rows.pop(key, None)
            if keys:
                self._rewrite()
        return len(keys)

    def count(self) -> int:
        self._load()
        with self._lock:
            return len(self._rows)

    def _rewrite(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for chunk, vector in self._rows.values():
                payload = chunk.to_dict()
                payload["vector"] = vector
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
