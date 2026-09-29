"""Markdown-aware chunking for knowledge documents."""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Sequence, Tuple

from .models import KnowledgeChunk, KnowledgeDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def stable_id(prefix: str, *parts: str) -> str:
    raw = "\0".join([prefix, *[str(part or "") for part in parts]])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def chunk_document(document: KnowledgeDocument, max_chars: int = 900, overlap: int = 120) -> List[KnowledgeChunk]:
    """Split Markdown by headings, then by paragraphs with a small overlap."""
    max_chars = max(200, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 3))
    doc_id = document.document_id
    sections = _sections(document.text)
    chunks: List[KnowledgeChunk] = []
    ordinal = 0
    for heading, body in sections:
        for piece in _split_text(body, max_chars=max_chars, overlap=overlap):
            ordinal += 1
            chunk_id = stable_id("chunk", document.tier, document.source, doc_id, heading, str(ordinal), piece)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=piece,
                    tier=document.tier,
                    title=document.title,
                    source=document.source,
                    heading=heading,
                    ordinal=ordinal,
                    tags=list(document.tags),
                    metadata={**document.metadata, "updated_at": document.updated_at},
                )
            )
    return chunks


def _sections(text: str) -> Sequence[Tuple[str, str]]:
    lines = str(text or "").splitlines()
    current_heading = ""
    buffer: List[str] = []
    sections: List[Tuple[str, str]] = []
    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            body = "\n".join(buffer).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = match.group(2).strip()
            buffer = []
            continue
        buffer.append(line)
    body = "\n".join(buffer).strip()
    if body:
        sections.append((current_heading, body))
    return sections or [("", str(text or "").strip())]


def _split_text(text: str, max_chars: int, overlap: int) -> Iterable[str]:
    text = text.strip()
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            prefix = current[-overlap:] if overlap else ""
            current = f"{prefix}\n\n{paragraph}".strip()
        else:
            chunks.extend(_hard_split(paragraph, max_chars=max_chars, overlap=overlap))
            current = ""
    if current:
        chunks.append(current)
    return [item for item in chunks if item.strip()]


def _hard_split(text: str, max_chars: int, overlap: int) -> List[str]:
    step = max(1, max_chars - overlap)
    return [text[index : index + max_chars] for index in range(0, len(text), step) if text[index : index + max_chars].strip()]
