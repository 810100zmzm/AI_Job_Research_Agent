"""Knowledge-base ingestion and retrieval orchestration."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..domain.jd_html import markdown_from_html
from .chunking import chunk_document, stable_id
from .interfaces import EmbeddingProvider, VectorIndex
from .models import (
    KNOWLEDGE_TIERS,
    L1_STATIC,
    L2_SEMI_STATIC,
    L3_DYNAMIC,
    KnowledgeDocument,
    KnowledgeHit,
    KnowledgeStats,
)

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".json", ".yaml", ".yml"}


class KnowledgeBase:
    def __init__(self, embedder: EmbeddingProvider, index: VectorIndex) -> None:
        self.embedder = embedder
        self.index = index

    def add_text(
        self,
        text: str,
        *,
        tier: str,
        title: str,
        source: str = "",
        tags: Sequence[str] = (),
        metadata: Optional[dict] = None,
        max_chars: int = 900,
    ) -> int:
        if tier not in KNOWLEDGE_TIERS:
            raise ValueError(f"未知知识层级：{tier}")
        document_id = stable_id("document", tier, source, title)
        document = KnowledgeDocument(
            document_id=document_id,
            text=str(text or ""),
            tier=tier,
            title=title or source or document_id,
            source=source,
            tags=list(tags),
            metadata=dict(metadata or {}),
        )
        chunks = chunk_document(document, max_chars=max_chars)
        if not chunks:
            return self.index.delete_document(document_id)
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.index.delete_document(document_id)
        return self.index.upsert(chunks, vectors)

    def ingest_file(
        self,
        path: Path,
        *,
        tier: str,
        tags: Sequence[str] = (),
        force: bool = False,
    ) -> int:
        target = Path(path)
        if not target.is_file() or target.suffix.casefold() not in TEXT_SUFFIXES:
            return 0
        raw = target.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        source = str(target.resolve())
        if not force and self._is_current(source, fingerprint):
            return 0
        text = _decode_document(target, raw)
        return self.add_text(
            text,
            tier=tier,
            title=target.stem,
            source=source,
            tags=tags,
            metadata={"fingerprint": fingerprint, "mtime": target.stat().st_mtime},
        )

    def ingest_paths(
        self,
        paths: Iterable[Path],
        *,
        tier: str,
        tags: Sequence[str] = (),
        force: bool = False,
        recursive: bool = True,
    ) -> int:
        changed = 0
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.glob("*")
                for item in sorted(iterator):
                    changed += self.ingest_file(item, tier=tier, tags=tags, force=force)
            else:
                changed += self.ingest_file(path, tier=tier, tags=tags, force=force)
        return changed

    def ensure_default_index(self, project_root: Path, force: bool = False) -> int:
        root = Path(project_root)
        groups = (
            (L1_STATIC, [root / "knowledge"], ("static",)),
            (L2_SEMI_STATIC, [root / "input" / "jd", root / "input" / "project", root / "input" / "profile"], ("semi-static",)),
            (L3_DYNAMIC, [root / "output"], ("dynamic",)),
        )
        changed = 0
        for tier, paths, tags in groups:
            changed += self.ingest_paths(paths, tier=tier, tags=tags, force=force)
        return changed

    def search(
        self,
        query: str,
        *,
        tiers: Sequence[str] = (),
        limit: int = 6,
    ) -> List[KnowledgeHit]:
        if not str(query or "").strip() or not self.index.count():
            return []
        vectors = self.embedder.embed([query])
        if not vectors:
            return []
        return self.index.search(query, vectors[0], tiers=tiers, limit=limit)

    def stats(self) -> KnowledgeStats:
        chunks = self.index.list()
        counts = {tier: 0 for tier in KNOWLEDGE_TIERS}
        for chunk in chunks:
            counts[chunk.tier] = counts.get(chunk.tier, 0) + 1
        return KnowledgeStats(
            chunk_count=len(chunks),
            tier_counts=counts,
            embedding_backend=self.embedder.name,
            index_backend=self.index.name,
        )

    def list_documents(self, tiers: Sequence[str] = ()) -> List[dict]:
        rows = []
        seen = set()
        for chunk in self.index.list(tiers=tiers):
            if chunk.document_id in seen:
                continue
            seen.add(chunk.document_id)
            rows.append(
                {
                    "document_id": chunk.document_id,
                    "tier": chunk.tier,
                    "title": chunk.title,
                    "source": chunk.source,
                    "tags": list(chunk.tags),
                }
            )
        return rows

    def _is_current(self, source: str, fingerprint: str) -> bool:
        for chunk in self.index.list():
            if chunk.source == source and chunk.metadata.get("fingerprint") == fingerprint:
                return True
        return False


def _decode_document(path: Path, raw: bytes) -> str:
    if path.suffix.casefold() in {".html", ".htm"}:
        text = raw.decode("utf-8", errors="replace")
        markdown, _ = markdown_from_html(text, strict=False)
        return markdown
    return raw.decode("utf-8", errors="replace")
