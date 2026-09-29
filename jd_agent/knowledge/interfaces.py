"""Interfaces for embeddings, indexing, and the knowledge base."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from .models import KnowledgeChunk, KnowledgeHit


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class VectorIndex(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        tiers: Sequence[str] = (),
        limit: int = 5,
    ) -> List[KnowledgeHit]:
        raise NotImplementedError

    @abstractmethod
    def list(self, tiers: Sequence[str] = ()) -> List[KnowledgeChunk]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
