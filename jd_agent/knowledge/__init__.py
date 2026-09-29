"""Tiered knowledge base for static, semi-static, and dynamic project knowledge."""
from .chunking import chunk_document, stable_id
from .embedding import HashEmbeddingProvider, QwenEmbeddingProvider
from .factory import build_knowledge_base
from .interfaces import EmbeddingProvider, VectorIndex
from .local_index import JsonlVectorIndex
from .manager import KnowledgeBase
from .models import (
    KNOWLEDGE_TIERS,
    L1_STATIC,
    L2_SEMI_STATIC,
    L3_DYNAMIC,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeHit,
    KnowledgeStats,
    TIER_LABELS,
)

__all__ = [
    "KNOWLEDGE_TIERS",
    "L1_STATIC",
    "L2_SEMI_STATIC",
    "L3_DYNAMIC",
    "TIER_LABELS",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "JsonlVectorIndex",
    "KnowledgeBase",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "KnowledgeHit",
    "KnowledgeStats",
    "QwenEmbeddingProvider",
    "chunk_document",
    "stable_id",
    "VectorIndex",
    "build_knowledge_base",
]
