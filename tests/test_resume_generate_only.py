"""Independent polish and generation path tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from jd_agent.agents.resume_polish import polish_resume
from jd_agent.api.app import create_app

RESUME = (
    "# 张三\n\n"
    "## 项目经历\n\n"
    "- 我用 Python 写了数据清洗脚本\n"
    "- 参与 FastAPI 接口开发\n"
)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **_kwargs):
        self.calls += 1
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        rows = [
            {
                "ref": fact["ref"],
                "original": fact["text"],
                "polished": fact["text"].replace("参与 FastAPI", "参与并推进 FastAPI"),
            }
            for fact in payload["verified_facts"]
        ]
        return json.dumps({"items": rows}, ensure_ascii=False)


class GenerateOnlyTest(unittest.TestCase):
    def test_generate_only_skips_dictionary_and_llm(self) -> None:
        class Bomb:
            def chat(self, *_args, **_kwargs):
                raise AssertionError("纯生成不应调用 LLM")

        result = polish_resume(
            RESUME,
            level="L2",
            text_client=Bomb(),
            enable_polish=False,
        )

        self.assertEqual(result.items, [])
        self.assertEqual(result.stats.total, 0)
        self.assertEqual(result.stats.dictionary, 0)
        self.assertEqual(result.stats.llm, 0)
        self.assertEqual(result.stats.calls, 0)
        self.assertIn("我用 Python 写了数据清洗脚本", result.markdown)
        self.assertNotIn("使用 Python 编写", result.markdown)
        self.assertIn("<!DOCTYPE html>", result.html)
        self.assertIn("未启用润色", result.notice)


class GenerateOnlyApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.clients: list[FakeClient] = []

        def client_factory():
            client = FakeClient()
            self.clients.append(client)
            return client

        app = create_app(
            data_dir=root / "data",
            output_dir=root / "output",
            frontend_dir=None,
            resume_text_client_factory=client_factory,
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.tmp.cleanup()

    def test_generate_and_polish_are_independent_actions(self) -> None:
        generated = self.client.post(
            "/api/resume/polish",
            json={
                "resume_input": RESUME,
                "level": "L2",
                "style": "classic",
                "enable_polish": False,
            },
        )
        self.assertEqual(generated.status_code, 200, generated.text)
        generated_payload = generated.json()
        self.assertEqual(generated_payload["items"], [])
        self.assertEqual(generated_payload["stats"]["calls"], 0)
        self.assertEqual(len(self.clients), 0)
        self.assertIn("我用 Python 写了数据清洗脚本", generated_payload["markdown"])
        self.assertNotIn("使用 Python 编写", generated_payload["markdown"])

        polished = self.client.post(
            "/api/resume/polish",
            json={
                "resume_input": RESUME,
                "level": "L2",
                "style": "classic",
                "enable_polish": True,
            },
        )
        self.assertEqual(polished.status_code, 200, polished.text)
        polished_payload = polished.json()
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(self.clients[0].calls, 0)
        self.assertTrue(polished_payload["items"])
        self.assertTrue(all(item["source"] == "llm-polish" for item in polished_payload["items"]))
        self.assertIn("使用 Python 编写数据清洗脚本", polished_payload["markdown"])


if __name__ == "__main__":
    unittest.main()
