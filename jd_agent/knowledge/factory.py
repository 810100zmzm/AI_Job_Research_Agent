"""Knowledge-base backend factory."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .embedding import DEFAULT_QWEN_EMBEDDING_MODEL, HashEmbeddingProvider, QwenEmbeddingProvider
from .interfaces import EmbeddingProvider
from .local_index import JsonlVectorIndex
from .manager import KnowledgeBase
from .qdrant_index import QdrantVectorIndex


def build_knowledge_base(
    data_dir: Path,
    *,
    embedding_backend: str = "auto",
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    embedding_model: str = DEFAULT_QWEN_EMBEDDING_MODEL,
    embedding_dimensions: int = 0,
    index_backend: str = "auto",
    qdrant_url: str = "",
    qdrant_api_key: str = "",
    qdrant_collection: str = "job_research_knowledge",
) -> KnowledgeBase:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    embedder: EmbeddingProvider = _build_embedder(
        embedding_backend,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        model=embedding_model,
        dimensions=embedding_dimensions,
    )
    selected = (index_backend or "auto").strip().lower()
    use_qdrant = selected == "qdrant" or (selected == "auto" and bool(qdrant_url))
    if use_qdrant:
        if not qdrant_url:
            raise ValueError("Qdrant 已启用但未配置 QDRANT_URL")
        index = QdrantVectorIndex(
            qdrant_url,
            collection=qdrant_collection,
            api_key=qdrant_api_key,
            dimensions=embedder.dimensions,
        )
    else:
        index = JsonlVectorIndex(root / "knowledge" / "index.jsonl")
    return KnowledgeBase(embedder, index)


def _build_embedder(
    backend: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    dimensions: int,
) -> EmbeddingProvider:
    selected = (backend or "auto").strip().lower()
    if selected == "local" or (selected == "auto" and not api_key):
        return HashEmbeddingProvider()
    if selected not in {"auto", "qwen", "dashscope"}:
        raise ValueError(f"未知嵌入后端：{backend}")
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    return QwenEmbeddingProvider(
        api_key,
        model=model,
        dimensions=dimensions,
        **kwargs,
    )
