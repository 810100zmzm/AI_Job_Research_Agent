"""Embedding providers: deterministic local fallback and Qwen/DashScope."""
from __future__ import annotations

from typing import List, Sequence

from ..core.llm import DEFAULT_DASHSCOPE_BASE_URL, OpenAICompatClient
from ..core.text import hash_embedding
from .interfaces import EmbeddingProvider

DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"


class HashEmbeddingProvider(EmbeddingProvider):
    """Offline fallback that requires no model or network access."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = max(64, int(dimensions))

    @property
    def name(self) -> str:
        return "local-hash"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [hash_embedding(text, self._dimensions) for text in texts]


class QwenEmbeddingProvider(EmbeddingProvider):
    """Alibaba Cloud DashScope OpenAI-compatible embeddings provider."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_DASHSCOPE_BASE_URL,
        model: str = DEFAULT_QWEN_EMBEDDING_MODEL,
        dimensions: int = 0,
        timeout: float = 60.0,
    ) -> None:
        if not str(api_key or "").strip():
            raise ValueError("缺少 DASHSCOPE_API_KEY / QWEN_API_KEY")
        self.client = OpenAICompatClient(base_url, api_key, model, timeout=timeout)
        self.model = model
        self._dimensions = max(0, int(dimensions or 0))

    @property
    def name(self) -> str:
        return f"qwen:{self.model}"

    @property
    def dimensions(self) -> int:
        if not self._dimensions:
            probe = self.embed(["dimension probe"])[0]
            self._dimensions = len(probe)
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        batch = [str(text or "") for text in texts]
        if not batch:
            return []
        return self.client.embed(
            batch,
            model=self.model,
            dimensions=self._dimensions or None,
        )
