"""Resume polishing / generation pipeline tests. No network calls."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from jd_agent.agents.resume_polish import (  # noqa: E402
    POLISH_LEVEL_PROMPTS,
    TERM_MAPPING_PROMPT,
    _apply_rules_with_hits,
    polish_resume,
)
from jd_agent.api.app import create_app  # noqa: E402
from jd_agent.core.llm import LLMError  # noqa: E402

RESUME = """# 李小明

## 项目经历

### 问答助手

- 参与 FastAPI 问答接口开发，响应时间从 800ms 降到 120ms
- 没做过 Docker 部署
"""

LLM_RESUME = """# 李小明

## 项目经历

### 问答助手

- 参与 FastAPI 问答接口维护
- 没做过 Docker 部署
"""

JD = """# 后端开发实习生

## 任职要求

- 熟悉 Python 和 FastAPI
"""


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def chat(self, messages, model=None, max_tokens=None, temperature=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        if self.reply is not None:
            return self.reply
        facts = json.loads(messages[1]["content"].split("\n", 1)[1])["verified_facts"]
        rows = []
        for fact in facts:
            rows.append(
                {
                    "ref": fact["ref"],
                    "original": fact["text"],
                    "polished": fact["text"].replace("参与 FastAPI", "参与并推进 FastAPI"),
                }
            )
        return json.dumps({"items": rows}, ensure_ascii=False)


class ResumePolishRulesTest(unittest.TestCase):
    def test_dictionary_beats_llm_and_keeps_source_label(self) -> None:
        class Bomb:
            def chat(self, *_args, **_kwargs):
                raise AssertionError("词典命中时不应调用 LLM")

        text = "# 张三\n\n## 项目经历\n\n- 我用 Python 写了数据清洗脚本\n"
        result = polish_resume(text, level="L2", text_client=Bomb())
        self.assertEqual(result.stats.llm, 0)
        self.assertGreaterEqual(result.stats.dictionary, 1)
        self.assertTrue(all(item.source == "llm-polish" for item in result.items))
        self.assertIn("使用 Python", result.markdown)

    def test_blocked_fact_never_reaches_the_model_or_generated_resume(self) -> None:
        client = FakeClient()
        result = polish_resume(LLM_RESUME, level="L2", text_client=client)
        payload = client.calls[0][1]["content"]
        self.assertNotIn("没做过 Docker 部署", payload)
        self.assertIn("没做过 Docker 部署", result.markdown)
        self.assertEqual(result.stats.blocked, 1)

    def test_new_metric_from_model_is_rejected(self) -> None:
        reply = json.dumps(
            {
                "items": [
                    {
                        "ref": "resume.md::L7",
                        "original": "参与 FastAPI 问答接口维护",
                        "polished": "参与 FastAPI 问答接口维护，准确率提升到 99%",
                    }
                ]
            },
            ensure_ascii=False,
        )
        result = polish_resume(LLM_RESUME, level="L3", text_client=FakeClient(reply=reply))
        item = next(item for item in result.items if "FastAPI" in item.original)
        self.assertEqual(item.engine, "original")
        self.assertIn("安全校验未通过", item.status)
        self.assertNotIn("99%", result.markdown)

    def test_llm_failure_degrades_to_original_and_still_generates_both_formats(self) -> None:
        result = polish_resume(LLM_RESUME, level="L3", text_client=FakeClient(error=LLMError("HTTP 500")))
        self.assertIn("润色未启用", result.notice)
        self.assertIn("参与 FastAPI 问答接口维护", result.markdown)
        self.assertIn("<!DOCTYPE html>", result.html)

    def test_three_styles_reuse_existing_templates(self) -> None:
        for style, marker in (
            ("classic", ".resume-doc h2"),
            ("structure", "section.awards"),
            ("accent", "--accent: #0f766e"),
        ):
            with self.subTest(style=style):
                result = polish_resume(RESUME, level="L2", style=style)
                self.assertIn(marker, result.html)


    def test_rule_levels_produce_distinct_dictionary_rewrites(self) -> None:
        class Bomb:
            def chat(self, *_args, **_kwargs):
                raise AssertionError("词典命中时不应调用 LLM")

        text = "# 张三\n\n## 项目经历\n\n- 我用 Python 写了数据清洗脚本\n"
        outputs = {}
        for level in ("L1", "L2", "L3"):
            result = polish_resume(text, level=level, text_client=Bomb())
            self.assertEqual(result.stats.llm, 0)
            self.assertGreaterEqual(result.stats.dictionary, 1)
            outputs[level] = result.items[0].polished

        self.assertEqual(outputs["L1"], "用 Python 写了数据清洗脚本")
        self.assertEqual(outputs["L2"], "使用 Python 编写数据清洗脚本")
        self.assertEqual(outputs["L3"], "基于 Python 实现数据清洗脚本")
        self.assertEqual(len(set(outputs.values())), 3)

    def test_l2_l3_prompts_embed_the_term_mapping_table(self) -> None:
        self.assertNotIn("术语映射表", POLISH_LEVEL_PROMPTS["L1"])
        self.assertIn(TERM_MAPPING_PROMPT, POLISH_LEVEL_PROMPTS["L2"])
        self.assertIn(TERM_MAPPING_PROMPT, POLISH_LEVEL_PROMPTS["L3"])
        self.assertIn("参与 -> L2: 参与；L3: 参与（绝对禁止改成主导）", TERM_MAPPING_PROMPT)
        self.assertIn("了解 -> L2: 了解；L3: 了解（绝对禁止改成精通）", TERM_MAPPING_PROMPT)

    def test_l3_frontloads_the_metric_and_derives_its_percentage(self) -> None:
        class Bomb:
            def chat(self, *_args, **_kwargs):
                raise AssertionError("命中规则时不应调用 LLM")

        text = "# 张三\n\n## 项目经历\n\n- 我参与 FastAPI 接口开发，响应时间从 800ms 降到 120ms\n"
        outputs = {
            level: polish_resume(text, level=level, text_client=Bomb()).items[0].polished
            for level in ("L1", "L2", "L3")
        }

        self.assertEqual(outputs["L1"], "参与 FastAPI 接口开发，响应时间从 800ms 降到 120ms")
        self.assertIn("研发", outputs["L2"])
        self.assertTrue(outputs["L3"].startswith("响应时间"))
        self.assertIn("降低 85%", outputs["L3"])
        self.assertIn("工程化研发", outputs["L3"])

    def test_l1_removes_grammar_noise_without_upgrading_the_verb(self) -> None:
        class Bomb:
            def chat(self, *_args, **_kwargs):
                raise AssertionError("规则命中时不应调用 LLM")

        text = "# 张三\n\n## 项目经历\n\n- 我进行了数据清洗\n"
        result = polish_resume(text, level="L1", text_client=Bomb())
        self.assertEqual(result.items[0].polished, "进行了数据清洗")

    def test_model_cannot_upgrade_ownership_or_competence(self) -> None:
        cases = (
            ("参与 FastAPI 问答接口维护", "主导 FastAPI 问答接口维护"),
            ("了解 Python", "精通 Python"),
        )
        for original, polished in cases:
            with self.subTest(original=original):
                reply = json.dumps(
                    {
                        "items": [
                            {"ref": "resume.md::L5", "original": original, "polished": polished}
                        ]
                    },
                    ensure_ascii=False,
                )
                text = f"# 张三\n\n## 项目经历\n\n- {original}\n"
                result = polish_resume(text, level="L3", text_client=FakeClient(reply=reply))
                self.assertEqual(result.items[0].engine, "original")
                self.assertIn("安全校验未通过", result.items[0].status)

    def test_l3_accepts_only_derived_percentages_from_original_numbers(self) -> None:
        original = "当前耗时 800ms，调整后耗时 120ms"
        reply = json.dumps(
            {
                "items": [
                    {
                        "ref": "resume.md::L5",
                        "original": original,
                        "polished": "耗时降低 85%，调整后耗时 120ms，当前耗时 800ms",
                    }
                ]
            },
            ensure_ascii=False,
        )
        text = f"# 张三\n\n## 项目经历\n\n- {original}\n"
        result = polish_resume(text, level="L3", text_client=FakeClient(reply=reply))
        self.assertEqual(result.items[0].engine, "llm")
        self.assertIn("85%", result.items[0].polished)

    def test_dictionary_does_not_rewrite_nominal_compounds(self) -> None:
        cases = (
            "负责推进接口维护",
            "测试工程师维护测试用例",
            "FastAPI 开发者维护接口",
        )
        for text in cases:
            with self.subTest(text=text):
                polished, hits = _apply_rules_with_hits(text, "L3")
                self.assertEqual(polished, text)
                self.assertEqual(hits, [])

        polished, hits = _apply_rules_with_hits("开发 FastAPI 接口", "L2")
        self.assertEqual(polished, "研发 FastAPI 接口")
        self.assertEqual(hits, ["开发->研发"])

        polished, hits = _apply_rules_with_hits("跑了脚本", "L2")
        self.assertEqual(polished, "执行了脚本")
        self.assertEqual(hits, ["跑->执行"])

    def test_user_is_not_rewritten_as_use_user(self) -> None:
        class Bomb:
            def chat(self, *_args, **_kwargs):
                raise AssertionError("词典命中时不应调用 LLM")

        text = "# 张三\n\n## 项目经历\n\n- 处理用户反馈\n"
        result = polish_resume(text, level="L2", text_client=Bomb())
        self.assertIn("处理用户反馈", result.markdown)
        self.assertNotIn("处理使用户反馈", result.markdown)


class ResumePolishApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        app = create_app(
            data_dir=root / "data",
            output_dir=root / "output",
            frontend_dir=None,
            resume_text_client_factory=lambda: FakeClient(),
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.tmp.cleanup()

    def test_options_polish_and_download(self) -> None:
        options = self.client.get("/api/resume/options")
        self.assertEqual(options.status_code, 200, options.text)
        self.assertEqual([item["key"] for item in options.json()["levels"]], ["L1", "L2", "L3"])
        self.assertEqual(
            [item["key"] for item in options.json()["styles"]],
            ["classic", "structure", "accent"],
        )

        jd = self.client.post(
            "/api/upload",
            files={"file": ("jd.md", JD.encode("utf-8"), "text/markdown")},
        )
        resume = self.client.post(
            "/api/upload",
            files={"file": ("resume.md", RESUME.encode("utf-8"), "text/markdown")},
        )
        analyze = self.client.post(
            "/api/analyze",
            json={
                "jd_input": jd.json()["file_id"],
                "resume_input": resume.json()["file_id"],
                "session_id": "resume-test",
                "user_id": "tester",
            },
        )
        task_id = analyze.json()["task_id"]

        generated = self.client.post(
            "/api/resume/polish",
            json={"task_id": task_id, "level": "L3", "style": "accent"},
        )
        self.assertEqual(generated.status_code, 200, generated.text)
        payload = generated.json()
        self.assertEqual(payload["level"], "L3")
        self.assertEqual(payload["style"], "accent")
        self.assertTrue(payload["resume_id"])
        self.assertTrue(payload["items"])
        self.assertTrue(all(item["source"] == "llm-polish" for item in payload["items"]))
        self.assertIn("<!DOCTYPE html>", payload["html"])

        markdown = self.client.get(
            f"/api/resume/{payload['resume_id']}",
            params={"format": "md"},
        )
        self.assertEqual(markdown.status_code, 200, markdown.text)
        self.assertIn("# 李小明", markdown.text)

        html = self.client.get(
            f"/api/resume/{payload['resume_id']}",
            params={"format": "html"},
        )
        self.assertEqual(html.status_code, 200, html.text)
        self.assertIn("resume-doc", html.text)


if __name__ == "__main__":
    unittest.main()
