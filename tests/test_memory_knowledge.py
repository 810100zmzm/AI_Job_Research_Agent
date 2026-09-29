"""L0-L2 memory and L1-L3 knowledge integration tests."""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jd_agent.agents.agent import run_agent  # noqa: E402
from jd_agent.core.settings import DEFAULT_DATA_DIR, resolve_storage_settings  # noqa: E402
from jd_agent.core.text import cosine  # noqa: E402
from jd_agent.knowledge import (  # noqa: E402
    L1_STATIC,
    L2_SEMI_STATIC,
    L3_DYNAMIC,
    HashEmbeddingProvider,
    JsonlVectorIndex,
    KnowledgeBase,
    KnowledgeDocument,
    chunk_document,
)
from jd_agent.memory import MemoryRecord, build_memory_manager  # noqa: E402
from jd_agent.memory.short_term import InMemoryShortTermMemory  # noqa: E402

SAMPLE_JD = """# AI 工程师

## 岗位一：AI 应用开发

**任职要求**

- 熟练使用 Python，了解 FastAPI 等 Web 框架。
- 了解 RAG 与向量数据库。
"""

RICH_PROJECT = """# 项目：AI 周报助手

## 我做了什么

- 独立完成采集脚本，用 Python 和 FastAPI 提供接口。
- 接入大模型完成摘要流程。

## 结果

- 摘要可用率从 62% 提升到 88%。
"""


class ProjectDataCase(unittest.TestCase):
    """Keep all test runtime data under the project data/ directory."""

    def setUp(self) -> None:
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="memory-knowledge-", dir=DEFAULT_DATA_DIR))
        self.jd_path = self.tmp / "jd.md"
        self.project_path = self.tmp / "project.md"
        self.jd_path.write_text(SAMPLE_JD, encoding="utf-8")
        self.project_path.write_text(RICH_PROJECT, encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestShortTermMemory(ProjectDataCase):
    def test_ttl_expiry(self) -> None:
        store = InMemoryShortTermMemory(ttl_seconds=60)
        record = MemoryRecord(
            text="短期记忆",
            session_id="s1",
            expires_at=time.time() - 1,
        )
        store.add(record)

        self.assertEqual(store.count("s1"), 0)

    def test_capacity_prunes_lowest_importance(self) -> None:
        store = InMemoryShortTermMemory(ttl_seconds=60, max_items_per_session=2)
        for text, importance in (("低", 0.1), ("高", 0.9), ("中", 0.4)):
            store.add(MemoryRecord(text=text, session_id="s1", importance=importance))

        self.assertEqual([item.text for item in store.list("s1")], ["中", "高"])


class TestMemoryPersistence(ProjectDataCase):
    def test_high_importance_consolidates_to_l2_and_persists(self) -> None:
        manager = build_memory_manager(self.tmp, short_term_ttl_seconds=60)
        manager.remember_message(
            "s1",
            "user",
            "用户偏好：优先展示可量化结果",
            importance=0.95,
        )

        self.assertEqual(manager.long_term.count(), 1)

        reloaded = build_memory_manager(self.tmp, short_term_ttl_seconds=60)
        self.assertEqual(reloaded.long_term.count(), 1)
        self.assertIn("可量化结果", reloaded.long_term.all()[0].text)


class TestKnowledgeBase(ProjectDataCase):
    def test_chunking_preserves_headings_and_tiers(self) -> None:
        document = KnowledgeDocument(
            text="# 后端\n\nFastAPI 接口。\n\n## 检索\n\n向量检索与召回评估。",
            tier=L1_STATIC,
            title="engineering",
        )

        chunks = chunk_document(document, max_chars=200, overlap=20)

        self.assertEqual([chunk.heading for chunk in chunks], ["后端", "检索"])
        self.assertTrue(all(chunk.tier == L1_STATIC for chunk in chunks))

    def test_local_embedding_fallback_and_tiered_search(self) -> None:
        embedder = HashEmbeddingProvider(dimensions=128)
        vector = embedder.embed(["Python FastAPI 接口"])[0]
        self.assertEqual(embedder.name, "local-hash")
        self.assertEqual(len(vector), 128)
        self.assertAlmostEqual(cosine(vector, vector), 1.0)

        knowledge = KnowledgeBase(embedder, JsonlVectorIndex(self.tmp / "index.jsonl"))
        knowledge.add_text(
            "FastAPI 提供 HTTP 接口，并用 Python 完成参数校验。",
            tier=L1_STATIC,
            title="静态工程知识",
            source="knowledge/backend.md",
        )
        knowledge.add_text(
            "MongoDB 保存任务状态。",
            tier=L3_DYNAMIC,
            title="动态运行数据",
            source="output/run.md",
        )

        hits = knowledge.search("FastAPI 接口", tiers=(L1_STATIC,), limit=3)

        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "knowledge/backend.md")

        reloaded = KnowledgeBase(
            HashEmbeddingProvider(dimensions=128),
            JsonlVectorIndex(self.tmp / "index.jsonl"),
        )
        self.assertEqual(reloaded.stats().chunk_count, 2)
        self.assertTrue(reloaded.search("FastAPI 接口", tiers=(L1_STATIC,), limit=3))

    def test_search_empty_index_returns_no_hits(self) -> None:
        knowledge = KnowledgeBase(
            HashEmbeddingProvider(dimensions=128),
            JsonlVectorIndex(self.tmp / "empty-index.jsonl"),
        )
        self.assertEqual(knowledge.search("任何内容"), [])

    def test_default_index_is_incremental_and_tiered(self) -> None:
        root = self.tmp / "project"
        files = {
            root / "knowledge" / "guide.md": "# 规则\n\n静态知识。",
            root / "input" / "jd" / "jd.md": "# 岗位\n\n半静态 JD。",
            root / "output" / "report.md": "# 报告\n\n动态结果。",
        }
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        knowledge = KnowledgeBase(
            HashEmbeddingProvider(dimensions=128),
            JsonlVectorIndex(self.tmp / "default-index.jsonl"),
        )
        first = knowledge.ensure_default_index(root)
        second = knowledge.ensure_default_index(root)

        self.assertGreaterEqual(first, 3)
        self.assertEqual(second, 0)
        self.assertEqual(
            {item["tier"] for item in knowledge.list_documents()},
            {L1_STATIC, L2_SEMI_STATIC, L3_DYNAMIC},
        )


class TestAgentContextIntegration(ProjectDataCase):
    def test_empty_context_adds_no_retrieve_trace(self) -> None:
        memory = build_memory_manager(self.tmp, short_term_ttl_seconds=60)

        state = run_agent(
            self.jd_path,
            self.project_path,
            memory=memory,
            session_id="empty",
        )

        self.assertTrue(state.context.empty)
        self.assertNotIn("RetrieveContext", [step.action for step in state.trace])

    def test_memory_hit_adds_supplemental_context(self) -> None:
        memory = build_memory_manager(self.tmp, short_term_ttl_seconds=60)
        memory.short_term.add(
            MemoryRecord(
                text="AI 周报助手用 Python FastAPI 提供接口，已完成上线。",
                session_id="s1",
                importance=0.8,
            )
        )

        state = run_agent(
            self.jd_path,
            self.project_path,
            memory=memory,
            session_id="s1",
        )

        self.assertFalse(state.context.empty)
        self.assertIn("RetrieveContext", [step.action for step in state.trace])
        self.assertTrue(all(item.source.startswith("memory:") for item in state.context.memory))


class TestStorageSettings(ProjectDataCase):
    def test_relative_data_dir_stays_under_project(self) -> None:
        settings = resolve_storage_settings({"DATA_DIR": "data"}, data_dir="")
        self.assertEqual(settings.data_dir, (ROOT / "data").resolve())


if __name__ == "__main__":
    unittest.main()
