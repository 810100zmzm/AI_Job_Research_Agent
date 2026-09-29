"""Optional Qdrant vector index."""
from __future__ import annotations

import uuid
from typing import List, Sequence

from .interfaces import VectorIndex
from .models import KnowledgeChunk, KnowledgeHit


class QdrantVectorIndex(VectorIndex):
    def __init__(
        self,
        url: str,
        *,
        collection: str = "job_research_knowledge",
        api_key: str = "",
        dimensions: int = 0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError("使用 Qdrant 需要先安装：pip install qdrant-client") from exc
        if not dimensions:
            raise ValueError("Qdrant 初始化需要 embedding dimensions")
        self._models = __import__("qdrant_client.models", fromlist=["Filter", "PointStruct"])
        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection = collection
        self.dimensions = int(dimensions)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )
        else:
            info = self.client.get_collection(collection)
            existing = getattr(getattr(info.config.params, "vectors", None), "size", None)
            if existing and int(existing) != self.dimensions:
                raise ValueError(
                    f"Qdrant collection {collection!r} 已按 {existing} 维建立，"
                    f"当前嵌入维度是 {self.dimensions}；请换 QDRANT_COLLECTION 或删除旧 collection"
                )

    @property
    def name(self) -> str:
        return f"qdrant:{self.collection}"

    def upsert(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> int:
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            points.append(
                self._models.PointStruct(
                    id=point_id,
                    vector=[float(value) for value in vector],
                    payload={**chunk.to_dict(), "chunk_id": chunk.chunk_id},
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        tiers: Sequence[str] = (),
        limit: int = 5,
    ) -> List[KnowledgeHit]:
        query_filter = None
        if tiers:
            query_filter = self._models.Filter(
                must=[self._models.FieldCondition(key="tier", match=self._models.MatchAny(any=list(tiers)))]
            )
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(query_vector),
                query_filter=query_filter,
                limit=max(1, limit),
                with_payload=True,
            )
            rows = response.points
        else:
            rows = self.client.search(
                collection_name=self.collection,
                query_vector=list(query_vector),
                query_filter=query_filter,
                limit=max(1, limit),
                with_payload=True,
            )
        hits: List[KnowledgeHit] = []
        for row in rows:
            payload = dict(row.payload or {})
            payload.pop("vector", None)
            hits.append(
                KnowledgeHit(
                    chunk=KnowledgeChunk.from_dict(payload),
                    score=float(row.score),
                    retrieval=self.name,
                )
            )
        return hits

    def list(self, tiers: Sequence[str] = ()) -> List[KnowledgeChunk]:
        query_filter = None
        if tiers:
            query_filter = self._models.Filter(
                must=[self._models.FieldCondition(key="tier", match=self._models.MatchAny(any=list(tiers)))]
            )
        rows, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        return [KnowledgeChunk.from_dict(dict(row.payload or {})) for row in rows]

    def delete_document(self, document_id: str) -> int:
        before = self.count()
        query_filter = self._models.Filter(
            must=[self._models.FieldCondition(key="document_id", match=self._models.MatchValue(value=document_id))]
        )
        self.client.delete(collection_name=self.collection, points_selector=query_filter)
        return max(0, before - self.count())

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection, exact=True)
        return int(result.count)
