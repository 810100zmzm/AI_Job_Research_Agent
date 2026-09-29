"""核心流程测试。

运行方式（项目根目录）：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jd_agent import cli  # noqa: E402
from jd_agent.agents.agent import run_agent  # noqa: E402
from jd_agent.domain.evidence import match_requirements  # noqa: E402
from jd_agent.domain.jd import load_jd, parse_jd_text  # noqa: E402
from jd_agent.domain.project import load_project, parse_project_text  # noqa: E402
from jd_agent.core.schema import (  # noqa: E402
    DECISION_ASK,
    DECISION_STOP,
    DECISIONS,
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_NONE,
    EVIDENCE_RESULT,
)

SAMPLE_JD = """# 某公司 AI 岗

## 岗位一：大模型应用实习生

**岗位职责**

- 参与大模型应用开发与调试，协助完成知识库 RAG 功能。

**任职要求**

- 熟练使用 Python，了解 FastAPI 等 Web 框架。
- 熟悉大模型基础原理，了解 RAG 与向量数据库。
- 本科及以上在读。

**加分项**

- 熟悉 Docker 部署流程
- 了解 PyTorch / TensorFlow 任一框架

## 岗位二：AI 产品实习生

**任职要求**

- 逻辑清晰，擅长文档撰写与信息整理。
"""

RICH_PROJECT = """# 项目：AI 周报助手

## 背景

- 每周整理资讯要 3 小时。

## 我做了什么

- 独立完成整个项目，代码已开源到 GitHub。
- 用 Python 写了采集脚本，接入大模型 API 做摘要。
- 调试阶段定位过长文截断问题并修复。

## 结果

- 摘要可用率从 62% 提升到 88%。

## 复盘

- 没做过模型微调。
"""

THIN_PROJECT = """# 项目：校园问答小助手

## 项目简介

- 一个给学院同学用的小工具。

## 技术方案

- 前端用 Streamlit，后端调用大模型 API。
- 用向量数据库做知识库检索。
"""

STACK_ONLY_PROJECT = """# 项目：某工具

## 技术栈

- Python、FastAPI、Chroma
"""


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.jd_path = _write(self.tmp, "jd.md", SAMPLE_JD)
        self.rich_path = _write(self.tmp, "rich.md", RICH_PROJECT)
        self.thin_path = _write(self.tmp, "thin.md", THIN_PROJECT)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestJDParsing(TempCase):
    def test_split_positions_and_default_select(self) -> None:
        posting = load_jd(self.jd_path)
        self.assertEqual(posting.position_count, 2)
        self.assertEqual(posting.title, "岗位一：大模型应用实习生")
        self.assertEqual(len(posting.position_titles), 2)

    def test_select_position_by_title(self) -> None:
        posting = load_jd(self.jd_path, "岗位二")
        self.assertIn("岗位二", posting.title)
        keys = {item.capability_key for item in posting.requirements}
        self.assertIn("docs", keys)

    def test_select_unknown_title_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_jd(self.jd_path, "岗位九")

    def test_requirement_level_line_and_dedupe(self) -> None:
        posting = load_jd(self.jd_path)
        by_key = {item.capability_key: item for item in posting.requirements}
        self.assertEqual(by_key["python"].level, "must")
        self.assertEqual(by_key["python"].line_no, 11)
        # 「大模型」在职责和任职要求里都出现，只保留层级更高的那条
        self.assertEqual(by_key["llm_basics"].level, "must")
        self.assertEqual(by_key["llm_basics"].line_no, 12)
        self.assertEqual(by_key["docker"].level, "plus")

    def test_threshold_requirements_are_not_provable(self) -> None:
        posting = load_jd(self.jd_path)
        education = [item for item in posting.requirements if item.capability_key == "education"][0]
        self.assertFalse(education.provable)


class TestProjectParsing(TempCase):
    def test_evidence_levels(self) -> None:
        project = load_project(self.rich_path)
        by_line = {fact.line_no: fact for fact in project.facts}
        self.assertEqual(by_line[9].level, EVIDENCE_RESULT)    # 独立完成 + 已开源
        self.assertEqual(by_line[10].level, EVIDENCE_ACTION)   # 有动作、没结果
        self.assertEqual(by_line[15].level, EVIDENCE_RESULT)   # 量化结果
        self.assertTrue(by_line[15].has_metric)

    def test_background_facts_are_downgraded(self) -> None:
        project = load_project(self.rich_path)
        background = [fact for fact in project.facts if fact.section == "背景"][0]
        self.assertEqual(background.level, EVIDENCE_MENTION)
        self.assertTrue(background.metrics)   # 数字仍被记下来，但不作为交付证据

    def test_negation_is_recorded(self) -> None:
        project = load_project(self.rich_path)
        negated = {key for fact in project.facts for key in fact.negated}
        self.assertIn("finetune", negated)
        self.assertEqual(project.facts_for("finetune"), [])

    def test_tech_stack_line_is_mention(self) -> None:
        project = parse_project_text("# 某工具\n\n## 技术栈\n\n- Python、FastAPI、Chroma\n", "x.md")
        self.assertEqual(project.facts[0].level, EVIDENCE_MENTION)
        self.assertFalse(project.actionable_facts)


class TestEvidenceMatching(TempCase):
    def test_strict_vs_relaxed(self) -> None:
        posting = load_jd(self.jd_path)
        python = [item for item in posting.requirements if item.capability_key == "python"][0]
        project = parse_project_text(STACK_ONLY_PROJECT, "x.md")
        strict = match_requirements([python], project, allow_mention=False)[0]
        relaxed = match_requirements([python], project, allow_mention=True)[0]
        self.assertEqual(strict.level, EVIDENCE_NONE)
        self.assertEqual(relaxed.level, EVIDENCE_MENTION)
        self.assertTrue(relaxed.relaxed)

    def test_sub_item_coverage(self) -> None:
        posting = load_jd(self.jd_path)
        framework = [item for item in posting.requirements if item.capability_key == "dl_framework"][0]
        project = parse_project_text("# p\n\n- 用 PyTorch 训练过 CNN 分类模型\n", "x.md")
        match = match_requirements([framework], project, allow_mention=True)[0]
        self.assertIn("PyTorch", match.matched_subs)
        self.assertIn("TensorFlow", match.missing_subs)


class TestAgentLoop(TempCase):
    def test_every_step_has_four_parts(self) -> None:
        state = run_agent(self.jd_path, self.rich_path)
        self.assertTrue(state.trace)
        self.assertEqual([step.index for step in state.trace], list(range(1, len(state.trace) + 1)))
        for step in state.trace:
            self.assertIn(step.decision, DECISIONS)
            self.assertTrue(step.action, step)
            self.assertTrue(step.observation, step)
            self.assertTrue(step.state_update, step)
            self.assertTrue(step.decision_reason, step)
        self.assertIn(state.decision, (DECISION_ASK, DECISION_STOP))

    def test_adjust_then_stop(self) -> None:
        state = run_agent(self.jd_path, self.rich_path)
        decisions = [step.decision for step in state.trace]
        self.assertIn("Adjust", decisions)
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertEqual(decisions[-1], DECISION_STOP)

    def test_verdict_shape(self) -> None:
        state = run_agent(self.jd_path, self.rich_path)
        verdict = state.verdict
        self.assertIsNotNone(verdict)
        self.assertGreaterEqual(len(verdict.reasons), 2)
        self.assertLessEqual(len(verdict.reasons), 3)
        self.assertGreaterEqual(len(verdict.risks), 1)
        self.assertLessEqual(len(verdict.risks), 2)
        self.assertTrue(verdict.stop_reason)
        self.assertEqual(state.stop_reason, verdict.stop_reason)
        for reason in verdict.reasons:
            self.assertTrue(reason.jd_ref)
            self.assertTrue(reason.text)
        for risk in verdict.risks:
            self.assertTrue(risk.kind)
            self.assertTrue(risk.text)

    def test_missing_facts_trigger_ask(self) -> None:
        state = run_agent(self.jd_path, self.thin_path)
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertIsNone(state.verdict)
        self.assertTrue(state.question)
        self.assertTrue(state.question.endswith("？"))
        # 一轮只问一个问题：Trace 里只允许出现一次 Ask
        self.assertEqual([step.decision for step in state.trace].count(DECISION_ASK), 1)
        self.assertEqual(state.trace[-1].action, "CheckSufficiency")

    def test_answer_resumes_loop_and_stops(self) -> None:
        answer = "知识库检索是我独立做的：用 LangChain 把 200 多条 FAQ 切进 Chroma，正确率从 55% 提升到 78%"
        state = run_agent(self.jd_path, self.thin_path, answer=answer)
        actions = [step.action for step in state.trace]
        self.assertIn("ApplyUserAnswer", actions)
        self.assertEqual(state.rounds, 2)
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertIsNotNone(state.verdict)
        self.assertTrue(state.stop_reason)


class TestCli(TempCase):
    def test_writes_three_formats_and_exits_zero(self) -> None:
        out = self.tmp / "out"
        code = cli.main(
            ["--jd", str(self.jd_path), "--project", str(self.rich_path), "--out", str(out), "--quiet"]
        )
        self.assertEqual(code, cli.EXIT_OK)
        for suffix in ("md", "json", "html"):
            self.assertTrue((out / f"resume_decision.{suffix}").is_file())
        payload = _payload(out)
        self.assertEqual(payload["decision"], "Stop")
        self.assertTrue(payload["verdict"]["stop_reason"])
        self.assertEqual(len(payload["trace"]), len(payload["trace"]))
        for step in payload["trace"]:
            self.assertTrue(step["observation"])
            self.assertTrue(step["state_update"])

    def test_ask_exits_three(self) -> None:
        out = self.tmp / "out3"
        code = cli.main(
            ["--jd", str(self.jd_path), "--project", str(self.thin_path), "--out", str(out), "--quiet"]
        )
        self.assertEqual(code, cli.EXIT_ASK)
        payload = _payload(out)
        self.assertEqual(payload["decision"], "Ask")
        self.assertIsNone(payload["verdict"])
        self.assertTrue(payload["question"])

    def test_bad_jd_title_exits_two(self) -> None:
        code = cli.main(
            ["--jd", str(self.jd_path), "--project", str(self.rich_path), "--jd-title", "岗位九", "--quiet"]
        )
        self.assertEqual(code, cli.EXIT_ERROR)

    def test_missing_project_exits_two(self) -> None:
        code = cli.main(["--jd", str(self.jd_path), "--project", str(self.tmp / "nope.md"), "--quiet"])
        self.assertEqual(code, cli.EXIT_ERROR)


def _payload(out: Path):
    return json.loads((out / "resume_decision.json").read_text(encoding="utf-8"))
