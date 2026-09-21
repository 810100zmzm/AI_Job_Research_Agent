"""可选大模型层（DeepSeek 生成层 / Qwen-VL 多模态）的测试。

纪律：全部注入假 client，一个网络请求都不发；真实 key 在测试期间被移出环境变量。
重点覆盖 v1.1 的三条红线：规则优先（默认完全一样）、来源可区分（source 标记）、失败可降级。

运行方式（项目根目录）：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jd_agent import cli  # noqa: E402
from jd_agent.agent import run_agent  # noqa: E402
from jd_agent.generate import (  # noqa: E402
    CallBudget,
    build_report_payload,
    parse_draft,
    parse_interview,
    parse_polish,
    verdict_word,
)
from jd_agent.render import render_html, render_markdown  # noqa: E402
from jd_agent.settings import LLMSettings, load_env, resolve_settings  # noqa: E402
from jd_agent.llm import (  # noqa: E402
    DEFAULT_CALL_LIMIT,
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_RETRIES,
    LLMError,
    OpenAICompatClient,
    check_llm,
    parse_json_reply,
)
from jd_agent.project import parse_project_text  # noqa: E402
from jd_agent.schema import (  # noqa: E402
    DECISIONS,
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_NONE,
    EVIDENCE_RESULT,
    SOURCE_LLM_INTERVIEW,
    SOURCE_LLM_POLISH,
    SOURCE_LLM_SUGGEST,
    SOURCE_RULE,
    STATUS_DISCARDED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_IDLE,
    STATUS_OK,
    STATUS_OFF,
    STATUS_SKIPPED,
    LlmBlock,
)
from jd_agent.vision import (  # noqa: E402
    build_vision_messages,
    collect_images,
    enrich_project,
    parse_image_facts,
)

SAMPLE_JD = """# 某公司 AI 岗

## 岗位一：大模型应用实习生

**任职要求**

- 熟练使用 Python，了解 FastAPI 等 Web 框架。

**加分项**

- 熟悉 Docker 基础部署流程
"""

SAMPLE_PROJECT = """# 项目：AI 周报助手

## 我做了什么

- 独立完成采集与摘要流程，代码已开源到 GitHub。
- 用 Python 写了采集脚本，接入大模型 API 做摘要。

## 结果

- 摘要可用率从 62% 提升到 88%。

![效果截图](assets/chart.png)
"""

PLAIN_PROJECT = """# 项目：AI 周报助手

## 我做了什么

- 独立完成采集与摘要流程，代码已开源到 GitHub。
- 用 Python 写了采集脚本，接入大模型 API 做摘要。

## 结果

- 摘要可用率从 62% 提升到 88%。
"""

THIN_PROJECT = """# 项目：校园问答小助手

## 项目简介

- 一个给学院同学用的小工具。

## 技术方案

- 前端用 Streamlit，后端调用大模型 API。
"""

# 模型嘴上说这两条是 result / mention，真实等级必须由 classify() 说了算
VISION_REPLY = json.dumps(
    {
        "facts": [
            {"text": "部署后可用率从 71% 提升到 89%", "level": "mention"},
            {"text": "界面里用 Docker 容器接入服务", "level": "mention"},
            {"text": "图例里还有一条没有数字的说明", "level": "result"},
        ]
    },
    ensure_ascii=False,
)

# v1.1 三块任务各自的正常回复
DRAFT_REPLY = json.dumps(
    {"draft": "用 Python 写了采集脚本，接入大模型 API 做摘要；摘要可用率从 62% 提升到 88%。"},
    ensure_ascii=False,
)

INTERVIEW_REPLY = json.dumps(
    {
        "questions": [
            {
                "question": "Docker 那套部署环境是怎么搭起来的？",
                "target": "Docker 与部署",
                "prepare": "准备一句：镜像里放了什么、启动命令是什么。",
            },
            {
                "question": "FastAPI 接口的并发与超时是怎么处理的？",
                "target": "Python / FastAPI",
                "prepare": "准备一句：并发上限、超时阈值与失败重试。",
            },
            {
                "question": "摘要可用率 88% 是按什么口径评测出来的？",
                "target": "结果事实：摘要可用率",
                "prepare": "准备一句：样本量与人工抽检口径。",
            },
        ]
    },
    ensure_ascii=False,
)

POLISH_REPLY = json.dumps(
    {
        "text": "候选人独立完成采集与摘要流程，代码已开源（项目 L13）；"
        "摘要可用率从 62% 提升到 88%（项目 L27）。"
    },
    ensure_ascii=False,
)

LLM_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DASHSCOPE_BASE_URL",
    "QWEN_VL_MODEL",
    "LLM_TIMEOUT",
)


class FakeClient:
    """假客户端：返回固定文本，或按需抛错；同时记录被问过什么。"""

    def __init__(self, reply: str = "", error: Optional[Exception] = None):
        self.reply = reply
        self.error = error
        self.calls: List[Any] = []

    def chat(self, messages, model=None, max_tokens=None, temperature=None) -> str:
        self.calls.append({"messages": messages, "model": model})
        if self.error is not None:
            raise self.error
        return self.reply

    @property
    def sent(self) -> str:
        """把发出去的全部内容拼成一段文本，便于断言「模型看到了什么」。"""
        return json.dumps(self.calls, ensure_ascii=False)


class TaskClient:
    """按系统提示词判断当前是哪块任务，分别返回草稿 / 追问 / 润色，模拟一个正常工作的 DeepSeek。"""

    def __init__(
        self,
        draft: str = DRAFT_REPLY,
        interview: str = INTERVIEW_REPLY,
        polish: str = POLISH_REPLY,
        error: Optional[Exception] = None,
    ):
        self.draft = draft
        self.interview = interview
        self.polish = polish
        self.error = error
        self.calls: List[Any] = []

    def chat(self, messages, model=None, max_tokens=None, temperature=None) -> str:
        self.calls.append({"messages": messages, "model": model})
        if self.error is not None:
            raise self.error
        system = messages[0]["content"]
        if "简历写作教练" in system:
            return self.draft
        if "面试官教练" in system:
            return self.interview
        return self.polish

    @property
    def sent(self) -> str:
        return json.dumps(self.calls, ensure_ascii=False)

    @property
    def tasks(self) -> List[str]:
        """这一轮被问到的任务顺序（按系统提示词归类）。"""
        names = []
        for call in self.calls:
            system = call["messages"][0]["content"]
            if "简历写作教练" in system:
                names.append(SOURCE_LLM_SUGGEST)
            elif "面试官教练" in system:
                names.append(SOURCE_LLM_INTERVIEW)
            else:
                names.append(SOURCE_LLM_POLISH)
        return names


class BombClient:
    """一被实例化就报错：用来证明默认关闭时不会创建任何客户端。"""

    def __init__(self, *args, **kwargs):
        raise AssertionError("默认关闭时不应该创建大模型客户端")


class NoRealKeys:
    """测试期间把机器上可能存在的真实 key 挪走，退出时原样恢复。"""

    def __enter__(self):
        self._patcher = mock.patch.dict(os.environ, {}, clear=False)
        self._patcher.start()
        for key in LLM_ENV_KEYS:
            os.environ.pop(key, None)
        return self

    def __exit__(self, *exc):
        self._patcher.stop()
        return False


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.assets = self.tmp / "assets"
        self.assets.mkdir()
        (self.assets / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        self.jd_path = self._write("jd.md", SAMPLE_JD)
        self.project_path = self._write("project.md", SAMPLE_PROJECT)
        self.env_file = self._write("empty.env", "")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def _settings(self) -> LLMSettings:
        return LLMSettings(text_api_key="test-key", vision_api_key="test-key")

    def baseline(self):
        return run_agent(self.jd_path, self.project_path)

    def run_with_llm(self, client=None, **kwargs):
        return run_agent(
            self.jd_path,
            self.project_path,
            llm=True,
            settings=self._settings(),
            text_client=client if client is not None else TaskClient(),
            **kwargs,
        )

    def run_with_vision(self, client=None, project_path=None, **kwargs):
        return run_agent(
            self.jd_path,
            project_path or self.project_path,
            vision=True,
            settings=self._settings(),
            vision_client=client if client is not None else FakeClient(VISION_REPLY),
            **kwargs,
        )


# ---- 1. 默认关闭：行为与纯规则版本一致 --------------------------------------

class TestDefaultOff(TempCase):
    def test_default_run_never_builds_a_client(self) -> None:
        with mock.patch("jd_agent.agent.OpenAICompatClient", BombClient):
            state = run_agent(self.jd_path, self.project_path)
        self.assertEqual(
            [step.action for step in state.trace],
            ["ReadJD", "ReadProject", "ExtractRequirements", "RetrieveEvidence", "AdjustEvidence", "Judge"],
        )
        self.assertIsNone(state.verdict.llm_report)
        self.assertIsNone(state.llm_report)
        self.assertFalse(state.vision.enabled)
        self.assertEqual(state.vision.facts_added, 0)

    def test_default_off_matches_rule_only_run(self) -> None:
        base = self.baseline()
        with mock.patch("jd_agent.agent.OpenAICompatClient", BombClient):
            state = run_agent(self.jd_path, self.project_path)
        self.assertEqual([step.action for step in state.trace], [step.action for step in base.trace])
        self.assertEqual([step.observation for step in state.trace], [step.observation for step in base.trace])
        self.assertEqual(state.trace[-1].decision, "Stop")
        self.assertEqual(state.verdict.call, base.verdict.call)
        self.assertEqual(state.verdict.headline, base.verdict.headline)
        self.assertEqual([reason.text for reason in state.verdict.reasons], [r.text for r in base.verdict.reasons])

    def test_image_reference_is_mentioned_but_not_parsed(self) -> None:
        state = self.baseline()
        self.assertIn("1 张本地图片", state.trace[1].observation)
        self.assertIn("--vision", state.trace[1].observation)
        self.assertNotIn("EnrichWithVision", [step.action for step in state.trace])


# ---- 2. 视觉层：图片 → 文字事实 → 规则定级 -----------------------------------

class TestVisionModule(TempCase):
    def test_collect_images_finds_markdown_html_and_bare_references(self) -> None:
        (self.assets / "flow.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        text = (
            "![效果](assets/chart.png)\n"
            '<img src="assets/flow.jpg">\n'
            "见 assets/chart.png 这张图\n"
            "![远程](https://example.com/a.png)\n"
            "![不存在](assets/missing.png)\n"
        )
        refs, notes = collect_images(text, self.project_path)
        self.assertEqual([ref.line_no for ref in refs], [1, 2])
        self.assertTrue(all(ref.ref.startswith("图片:") for ref in refs))
        self.assertEqual(notes, [])

    def test_explicit_image_argument_is_included(self) -> None:
        extra = self.tmp / "extra.png"
        extra.write_bytes(b"\x89PNG\r\n\x1a\n")
        refs, notes = collect_images("这里没有任何图片引用", self.project_path, [str(extra)])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].line_no, 0)
        self.assertEqual(notes, [])

    def test_unusable_explicit_image_is_only_a_note(self) -> None:
        refs, notes = collect_images("这里没有任何图片引用", self.project_path, ["assets/nope.png"])
        self.assertEqual(refs, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("nope.png", notes[0])

    def test_facts_are_built_by_make_fact_and_graded_by_rules(self) -> None:
        project = parse_project_text(SAMPLE_PROJECT, "x.md")
        refs, _ = collect_images(SAMPLE_PROJECT, self.project_path)
        before = len(project.facts)
        report = enrich_project(project, refs, FakeClient(VISION_REPLY), "qwen-vl-max")

        self.assertEqual(report.facts_added, 3)
        self.assertEqual(report.parsed, 1)
        self.assertEqual(report.error, "")
        facts = project.facts[before:]
        # 模型自述的 level 全部丢弃，等级由 classify() 判定
        self.assertEqual([fact.level for fact in facts], [EVIDENCE_RESULT, EVIDENCE_ACTION, EVIDENCE_MENTION])
        self.assertTrue(all(fact.source_file.startswith("图片:") for fact in facts))
        self.assertTrue(all(fact.source_file.endswith("chart.png") for fact in facts))
        self.assertTrue(all(fact.section == "图片解析" for fact in facts))
        self.assertTrue(all(fact.line_no == 0 for fact in facts))
        self.assertTrue(facts[0].has_metric)

    def test_model_failure_records_one_note_and_changes_nothing(self) -> None:
        project = parse_project_text(SAMPLE_PROJECT, "x.md")
        refs, _ = collect_images(SAMPLE_PROJECT, self.project_path)
        before = len(project.facts)
        report = enrich_project(
            project, refs, FakeClient(error=LLMError("HTTP 429：rate limit")), "qwen-vl-max"
        )
        self.assertEqual(len(project.facts), before)
        self.assertEqual(report.facts_added, 0)
        self.assertEqual(len(report.notes), 1)
        self.assertIn("HTTP 429", report.error)

    def test_parse_image_facts_falls_back_to_lines(self) -> None:
        self.assertEqual(parse_image_facts('{"facts": ["a", "b"]}'), ["a", "b"])
        self.assertEqual(parse_image_facts('```json\n{"facts": ["a"]}\n```'), ["a"])
        self.assertEqual(parse_image_facts("- 图中有一条折线\n- 准确率 88%"), ["图中有一条折线", "准确率 88%"])
        self.assertEqual(parse_image_facts(""), [])

    def test_vision_message_carries_the_image_as_data_url(self) -> None:
        refs, _ = collect_images(SAMPLE_PROJECT, self.project_path)
        messages = build_vision_messages(refs[0], "AI 周报助手")
        content = messages[1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("图片:", content[0]["text"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


class TestVisionStep(TempCase):
    def test_step_sits_between_read_project_and_extract_requirements(self) -> None:
        state = self.run_with_vision()
        actions = [step.action for step in state.trace]
        self.assertEqual(actions[:4], ["ReadJD", "ReadProject", "EnrichWithVision", "ExtractRequirements"])
        self.assertEqual(state.trace[-1].action, "Judge")
        self.assertEqual(state.trace[-1].decision, "Stop")
        step = state.trace[2]
        self.assertEqual(step.index, 3)
        self.assertEqual(step.decision, "Continue")
        self.assertTrue(step.observation and step.state_update and step.decision_reason)

    def test_vision_evidence_enters_rule_matching(self) -> None:
        def docker_level(state):
            for match in state.matches:
                if match.requirement_name == "Docker 与部署":
                    return match.level
            return None

        base = self.baseline()
        state = self.run_with_vision()
        self.assertEqual(docker_level(base), EVIDENCE_NONE)      # 文字描述里没有 Docker
        self.assertEqual(docker_level(state), EVIDENCE_RESULT)   # 图片事实按规则定级后进入检索
        self.assertGreater(state.result_match_count, base.result_match_count)

    def test_vision_failure_keeps_the_rule_conclusion(self) -> None:
        base = self.baseline()
        state = self.run_with_vision(FakeClient(error=LLMError("HTTP 401：invalid api key")))
        self.assertEqual(state.verdict.call, base.verdict.call)
        self.assertEqual(state.verdict.headline, base.verdict.headline)
        self.assertEqual(state.trace[-1].action, "Judge")
        self.assertEqual(state.trace[-1].decision, "Stop")
        self.assertEqual(state.vision.facts_added, 0)
        self.assertEqual(len(state.vision.notes), 1)
        self.assertEqual(len(state.project.facts), len(base.project.facts))

    def test_vision_without_key_stays_offline(self) -> None:
        with mock.patch("jd_agent.agent.OpenAICompatClient", BombClient):
            state = run_agent(self.jd_path, self.project_path, vision=True, settings=LLMSettings())
        step = [item for item in state.trace if item.action == "EnrichWithVision"][0]
        self.assertEqual(step.decision, "Continue")
        self.assertIn("DASHSCOPE_API_KEY", step.observation)
        self.assertIn("DASHSCOPE_API_KEY", state.vision.error)
        self.assertEqual(state.vision.facts_added, 0)
        self.assertEqual(state.verdict.call, self.baseline().verdict.call)

    def test_vision_without_images_still_records_the_step(self) -> None:
        plain = self._write("plain.md", PLAIN_PROJECT)
        client = FakeClient(VISION_REPLY)
        state = self.run_with_vision(client, project_path=plain)
        step = [item for item in state.trace if item.action == "EnrichWithVision"][0]
        self.assertIn("没有可解析的内容", step.observation)
        self.assertEqual(client.calls, [])
        self.assertEqual(state.vision.facts_added, 0)


# ---- 3. LLM 装饰层：报告生成之后才介入，三块内容各带 source -------------------

class TestLlmBlocks(TempCase):
    def test_rule_report_is_not_touched_by_the_model(self) -> None:
        base = self.baseline()
        state = self.run_with_llm()
        self.assertEqual(state.verdict.call, base.verdict.call)
        self.assertEqual(state.verdict.headline, base.verdict.headline)
        self.assertEqual(state.verdict.rewrite, base.verdict.rewrite)
        self.assertEqual(state.verdict.stop_reason, base.verdict.stop_reason)
        self.assertEqual([i.text for i in state.verdict.reasons], [i.text for i in base.verdict.reasons])
        self.assertEqual([i.text for i in state.verdict.risks], [i.text for i in base.verdict.risks])
        self.assertEqual(state.sufficiency, base.sufficiency)
        self.assertEqual(state.coverage, base.coverage)

    def test_three_blocks_with_distinct_sources(self) -> None:
        state = self.run_with_llm()
        report = state.llm_report
        self.assertEqual(report.sources, [SOURCE_LLM_SUGGEST, SOURCE_LLM_INTERVIEW, SOURCE_LLM_POLISH])
        self.assertEqual([block.label for block in report.blocks], ["[LLM]"] * 3)
        # 面试追问预演按用户要求闲置：块还在、状态是「已闲置」，但没有发出调用
        self.assertEqual([block.status for block in report.blocks], [STATUS_OK, STATUS_IDLE, STATUS_OK])
        self.assertEqual(report.ok_count, 2)
        self.assertEqual([block.source for block in report.degraded], [SOURCE_LLM_INTERVIEW])
        self.assertTrue(report.block(SOURCE_LLM_SUGGEST).text)
        interview = report.block(SOURCE_LLM_INTERVIEW)
        self.assertTrue(interview.idle)
        self.assertEqual(interview.calls, 0)
        self.assertEqual(interview.items, [])
        self.assertTrue(report.block(SOURCE_LLM_POLISH).text)
        self.assertEqual(state.llm_report, state.verdict.llm_report)

    def test_llm_steps_sit_after_the_rule_report_and_stop_last(self) -> None:
        state = self.run_with_llm()
        actions = [step.action for step in state.trace]
        self.assertEqual(
            actions,
            [
                "ReadJD",
                "ReadProject",
                "ExtractRequirements",
                "RetrieveEvidence",
                "AdjustEvidence",
                "Judge",
                "GenerateDraft",
                "RehearseInterview",
                "PolishReport",
                "Finish",
            ],
        )
        judge = state.trace[actions.index("Judge")]
        self.assertEqual(judge.decision, "Continue")
        self.assertIn("已经定稿", judge.observation)
        self.assertIn("state.verdict", judge.state_update)
        self.assertEqual(state.trace[-1].decision, "Stop")
        self.assertEqual(state.trace[-1].decision_reason, state.stop_reason)
        self.assertEqual(state.decision, "Stop")
        self.assertTrue(state.finished)
        for step in state.trace:
            self.assertIn(step.decision, DECISIONS)
            self.assertTrue(step.observation and step.state_update and step.decision_reason)

    def test_every_block_records_calls_and_discipline(self) -> None:
        client = TaskClient()
        state = self.run_with_llm(client)
        report = state.llm_report
        # 追问预演闲置后，一次 --llm 运行只发 2 次请求
        self.assertEqual(client.tasks, [SOURCE_LLM_SUGGEST, SOURCE_LLM_POLISH])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(report.calls, 2)
        self.assertEqual([block.calls for block in report.blocks], [1, 0, 1])
        self.assertEqual(report.call_limit, DEFAULT_CALL_LIMIT)
        self.assertEqual(report.timeout, DEFAULT_CALL_TIMEOUT)
        self.assertEqual(report.retries, DEFAULT_RETRIES)
        self.assertEqual(report.model, "deepseek-chat")
        self.assertEqual(report.blocks[0].model, "deepseek-chat")

    def test_model_sees_the_rule_report_json_only(self) -> None:
        client = TaskClient()
        state = self.run_with_llm(client)
        payload = build_report_payload(state)
        sent = client.sent
        self.assertIn(payload["headline"], sent)
        self.assertIn(payload["reasons"][0]["text"][:20], sent)
        self.assertIn("verified_facts", sent)
        self.assertIn("stop_reason", sent)
        self.assertNotIn('"trace"', sent)
        self.assertNotIn('"facts"', sent)
        self.assertNotIn(state.project.text, sent)
        self.assertEqual(payload["stage"], "rule-report-ready")

    def test_report_labels_every_section(self) -> None:
        state = self.run_with_llm()
        md = render_markdown(state, "2026-01-01 00:00")
        self.assertIn("一、结论 [规则]", md)
        self.assertIn("二、LLM 生成内容", md)
        for source in (SOURCE_LLM_SUGGEST, SOURCE_LLM_INTERVIEW, SOURCE_LLM_POLISH):
            self.assertIn(f"source: {source}", md)
        self.assertIn("三、岗位要求 vs 项目证据 [规则]", md)
        self.assertIn("四、运行 Trace", md)
        self.assertIn("五、判定口径", md)
        html = render_html(state, "2026-01-01 00:00")
        self.assertIn("[规则]", html)
        self.assertIn("source: llm-polish", html)

    def test_rule_only_report_has_no_llm_section(self) -> None:
        state = self.baseline()
        md = render_markdown(state, "2026-01-01 00:00")
        self.assertNotIn("LLM 生成内容", md)
        self.assertIn("二、岗位要求 vs 项目证据 [规则]", md)


# ---- 3.1 「LLM 不做结论」：出现档位词就整块丢弃 -------------------------------

class TestNoConclusion(TempCase):
    def test_banned_word_detector(self) -> None:
        self.assertEqual(verdict_word("这段经历值得写进简历"), "值得写")
        self.assertEqual(verdict_word("结论是暂不建议写"), "暂不建议写")
        self.assertEqual(verdict_word("用 Python 写了采集脚本"), "")

    def test_draft_block_is_discarded(self) -> None:
        reply = json.dumps({"draft": "这个项目不值得写。"}, ensure_ascii=False)
        state = self.run_with_llm(TaskClient(draft=reply))
        block = state.llm_report.block(SOURCE_LLM_SUGGEST)
        self.assertEqual(block.status, STATUS_DISCARDED)
        self.assertTrue(block.discarded)
        self.assertEqual(block.text, "")
        self.assertIn("LLM 不做结论", block.error)
        self.assertEqual(state.verdict.call, self.baseline().verdict.call)
        self.assertEqual(state.decision, "Stop")

    def test_interview_block_is_idle_and_the_model_is_never_asked(self) -> None:
        """追问预演已闲置：模型不会收到这块任务，但 Trace 与报告里都保留它的位置。"""
        client = TaskClient()
        state = self.run_with_llm(client)
        block = state.llm_report.block(SOURCE_LLM_INTERVIEW)
        self.assertTrue(block.idle)
        self.assertEqual(block.status, STATUS_IDLE)
        self.assertEqual(block.calls, 0)
        self.assertIn("闲置", block.notes[0])
        self.assertEqual(client.tasks, [SOURCE_LLM_SUGGEST, SOURCE_LLM_POLISH])
        step = [item for item in state.trace if item.action == "RehearseInterview"][0]
        self.assertIn("已闲置", step.observation)
        self.assertEqual(step.decision, "Continue")

    def test_interview_parser_still_rejects_conclusion_words(self) -> None:
        """解析与校验都还在：把 INTERVIEW_ENABLED 改回 True 就能直接恢复调用。"""
        block = LlmBlock(source=SOURCE_LLM_INTERVIEW, title="面试追问预演")
        parse_interview(
            block,
            json.dumps(
                {"questions": [{"question": "这个项目值得写吗？", "target": "x", "prepare": "y"}]},
                ensure_ascii=False,
            ),
        )
        self.assertEqual(block.status, STATUS_DISCARDED)
        self.assertEqual(block.items, [])

    def test_polish_block_is_discarded(self) -> None:
        reply = json.dumps({"text": "结论：暂不建议写进简历。"}, ensure_ascii=False)
        state = self.run_with_llm(TaskClient(polish=reply))
        block = state.llm_report.block(SOURCE_LLM_POLISH)
        self.assertEqual(block.status, STATUS_DISCARDED)
        self.assertEqual(block.text, "")
        self.assertIn("LLM 不做结论", block.error)


# ---- 3.2 失败可降级：规则结论、Trace 与退出码都不受影响 ----------------------

class TestLlmDegradation(TempCase):
    def test_failure_retries_once_and_keeps_the_report(self) -> None:
        base = self.baseline()
        client = TaskClient(error=LLMError("HTTP 402：Insufficient Balance"))
        state = self.run_with_llm(client)
        report = state.llm_report
        # 2 块任务 × (1 次调用 + 1 次重试)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(report.calls, 4)
        self.assertEqual([block.status for block in report.blocks], [STATUS_FAILED, STATUS_IDLE, STATUS_FAILED])
        for block in (report.blocks[0], report.blocks[2]):
            self.assertIn("HTTP 402", block.error)
            self.assertIn("重试 1 次", block.error)
        self.assertEqual(state.verdict.call, base.verdict.call)
        self.assertEqual(state.verdict.headline, base.verdict.headline)
        self.assertEqual(state.decision, "Stop")
        self.assertIn("LLM 0/3 块", state.trace[-1].observation)

    def test_unexpected_exception_is_also_degraded(self) -> None:
        state = self.run_with_llm(FakeClient(error=RuntimeError("boom")))
        block = state.llm_report.block(SOURCE_LLM_SUGGEST)
        self.assertEqual(block.status, STATUS_FAILED)
        self.assertIn("RuntimeError", block.error)
        self.assertEqual(state.decision, "Stop")
        self.assertTrue(state.verdict.stop_reason)

    def test_llm_without_key_notes_and_stays_offline(self) -> None:
        with mock.patch("jd_agent.agent.OpenAICompatClient", BombClient):
            state = run_agent(self.jd_path, self.project_path, llm=True, settings=LLMSettings())
        report = state.llm_report
        self.assertEqual([block.status for block in report.blocks], [STATUS_OFF] * 3)
        self.assertIn("DEEPSEEK_API_KEY", report.note)
        draft = [step for step in state.trace if step.action == "GenerateDraft"][0]
        self.assertIn("没有发出调用", draft.observation)
        self.assertIn("DEEPSEEK_API_KEY", draft.observation)
        self.assertEqual(state.verdict.call, self.baseline().verdict.call)
        self.assertEqual(state.decision, "Stop")

    def test_call_limit_is_enforced(self) -> None:
        client = TaskClient()
        state = self.run_with_llm(client, llm_max_calls=1)
        report = state.llm_report
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(report.calls, 1)
        self.assertEqual(report.call_limit, 1)
        self.assertEqual([block.status for block in report.blocks], [STATUS_OK, STATUS_IDLE, STATUS_SKIPPED])
        self.assertIn("调用上限", report.blocks[2].error)
        self.assertEqual(state.verdict.call, self.baseline().verdict.call)
        self.assertEqual(state.decision, "Stop")

    def test_ask_path_never_calls_the_model(self) -> None:
        thin = self._write("thin.md", THIN_PROJECT)
        client = TaskClient()
        state = run_agent(self.jd_path, thin, llm=True, settings=self._settings(), text_client=client)
        self.assertEqual(state.decision, "Ask")
        self.assertIsNone(state.verdict)
        self.assertIsNone(state.llm_report)
        self.assertEqual(client.calls, [])
        self.assertEqual(state.trace[-1].action, "CheckSufficiency")

    def test_generation_layer_crash_is_contained(self) -> None:
        with mock.patch("jd_agent.agent.generate_llm_report", side_effect=RuntimeError("boom")):
            state = self.run_with_llm()
        report = state.llm_report
        self.assertEqual([block.status for block in report.blocks], [STATUS_FAILED] * 3)
        self.assertIn("生成层异常", report.note)
        self.assertEqual(state.decision, "Stop")
        self.assertTrue(state.verdict.stop_reason)
        self.assertEqual(state.verdict.call, self.baseline().verdict.call)


# ---- 3.3 调用纪律：30 秒超时、重试 1 次、一次运行最多 10 次 --------------------

class TestCallDiscipline(TempCase):
    def test_generation_client_gets_the_30s_timeout(self) -> None:
        created: List[Any] = []

        class Recorder(OpenAICompatClient):
            def __init__(self, *args, **kwargs):
                created.append(args)
                super().__init__(*args, **kwargs)

        with mock.patch("jd_agent.agent.OpenAICompatClient", Recorder):
            run_agent(self.jd_path, self.project_path, llm=True, settings=self._settings())
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0][3], DEFAULT_CALL_TIMEOUT)
        self.assertEqual(DEFAULT_CALL_TIMEOUT, 30.0)
        self.assertEqual(DEFAULT_CALL_LIMIT, 10)
        self.assertEqual(DEFAULT_RETRIES, 1)

    def test_budget_stops_at_the_limit(self) -> None:
        budget = CallBudget(FakeClient("ok"), "deepseek-chat", limit=1, retries=1)
        self.assertEqual(budget.remaining, 1)
        self.assertEqual(budget.ask([{"role": "user", "content": "hi"}]), "ok")
        self.assertEqual(budget.remaining, 0)
        with self.assertRaises(LLMError):
            budget.ask([{"role": "user", "content": "hi"}])
        self.assertEqual(budget.calls, 1)

    def test_parsers_accept_json_and_plain_text(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_SUGGEST, title="建议写法草稿")
        parse_draft(block, "建议写法：用 Python 写了采集脚本，可用率 88%。")
        self.assertTrue(block.text)
        self.assertEqual(block.status, STATUS_OK)

        block = LlmBlock(source=SOURCE_LLM_INTERVIEW, title="面试追问预演")
        parse_interview(block, "1. 评测口径是什么？\n2. Docker 怎么搭的？")
        self.assertEqual([item["question"] for item in block.items], ["评测口径是什么？", "Docker 怎么搭的？"])
        self.assertEqual(block.status, STATUS_OK)

        block = LlmBlock(source=SOURCE_LLM_POLISH, title="报告润色")
        parse_polish(block, "{\"text\": \"候选人独立完成采集与摘要。\"}")
        self.assertEqual(block.text, "候选人独立完成采集与摘要。")
        self.assertEqual(block.status, STATUS_OK)

    def test_empty_reply_is_reported_but_not_a_failure(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_POLISH, title="报告润色")
        parse_polish(block, "   ")
        self.assertTrue(block.error)
        self.assertFalse(block.failed)
        self.assertEqual(block.status, STATUS_EMPTY)

# ---- 4. 客户端、自检与 .env --------------------------------------------------

class TestClientAndChecks(unittest.TestCase):
    def test_repr_and_errors_never_leak_the_key(self) -> None:
        client = OpenAICompatClient("https://api.deepseek.com/v1", "sk-super-secret", "deepseek-chat")
        self.assertNotIn("sk-super-secret", repr(client))
        self.assertIn("deepseek-chat", repr(client))
        self.assertEqual(client.url, "https://api.deepseek.com/v1/chat/completions")

    def test_chat_without_key_raises_llm_error(self) -> None:
        client = OpenAICompatClient("https://api.deepseek.com/v1", "", "deepseek-chat")
        with self.assertRaises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_check_llm_reports_missing_key_for_each_enabled_layer(self) -> None:
        with NoRealKeys():
            results = check_llm(LLMSettings(), check_vision=True)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not item.ok for item in results))
        self.assertIn("DEEPSEEK_API_KEY", results[0].detail)
        self.assertIn("DASHSCOPE_API_KEY", results[1].detail)

    def test_check_llm_passes_with_injected_client(self) -> None:
        results = check_llm(LLMSettings(), text_client=FakeClient("pong"))
        self.assertTrue(results[0].ok)
        self.assertIn("pong", results[0].detail)
        self.assertEqual(len(results), 1)          # 没开 --vision 就不检查多模态层

    def test_check_llm_surfaces_the_failure_detail(self) -> None:
        results = check_llm(LLMSettings(), text_client=FakeClient(error=LLMError("HTTP 401：bad key")))
        self.assertFalse(results[0].ok)
        self.assertIn("HTTP 401", results[0].detail)

    def test_parse_json_reply_tolerates_fences_and_noise(self) -> None:
        self.assertEqual(parse_json_reply('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_reply('好的：{"a": 2} 以上'), {"a": 2})
        self.assertIsNone(parse_json_reply("完全没有 JSON"))


class TestEnvFile(unittest.TestCase):
    """`.env` 交给 python-dotenv 读：只补缺失的键，已有环境变量一律不动。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.env_file = self.tmp / ".env"
        self.env_file.write_text(
            "# 注释行\n"
            "DEEPSEEK_API_KEY=sk-from-file\n"
            'export DASHSCOPE_API_KEY="sk-qwen"  # 行尾注释\n'
            "DEEPSEEK_MODEL=deepseek-reasoner\n",
            encoding="utf-8",
        )
        self._saved = {key: os.environ.get(key) for key in LLM_ENV_KEYS}
        for key in LLM_ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in LLM_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value
        self._tmp.cleanup()

    def test_load_env_reads_file_into_environment(self) -> None:
        result = load_env(self.env_file)
        self.assertTrue(result.found)
        self.assertEqual(result.error, "")
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-from-file")
        self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "sk-qwen")   # export 前缀 + 引号 + 行尾注释
        self.assertEqual(os.environ["DEEPSEEK_MODEL"], "deepseek-reasoner")
        self.assertEqual(
            sorted(result.applied), ["DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"]
        )

    def test_env_file_does_not_override_existing_environment(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "from-os-env"
        result = load_env(self.env_file)
        self.assertTrue(result.found)
        self.assertIn("DEEPSEEK_API_KEY", result.skipped)
        self.assertNotIn("DEEPSEEK_API_KEY", result.applied)
        self.assertIn("DASHSCOPE_API_KEY", result.applied)
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from-os-env")   # 没有被 .env 覆盖
        self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "sk-qwen")

    def test_missing_env_file_is_not_an_error(self) -> None:
        result = load_env(self.tmp / "nope.env")
        self.assertFalse(result.found)
        self.assertEqual(result.applied, [])
        self.assertEqual(result.error, "")
        self.assertIn("没有找到", result.summary())

    def test_summary_prints_key_names_only(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "sk-super-secret"
        text = load_env(self.env_file).summary()
        self.assertIn("DEEPSEEK_API_KEY", text)
        self.assertNotIn("sk-super-secret", text)      # 被跳过的键不回显值
        self.assertNotIn("sk-qwen", text)              # 被加载的键同样不回显值

    def test_resolve_settings_reads_env_and_falls_back(self) -> None:
        env = {
            "DEEPSEEK_API_KEY": "sk-a",
            "DEEPSEEK_MODEL": "deepseek-reasoner",
            "QWEN_API_KEY": "sk-b",
            "LLM_TIMEOUT": "12.5",
        }
        settings = resolve_settings(env)
        self.assertEqual(settings.text_api_key, "sk-a")
        self.assertEqual(settings.text_model, "deepseek-reasoner")
        self.assertEqual(settings.vision_api_key, "sk-b")       # DASHSCOPE_API_KEY 缺失时兼容 QWEN_API_KEY
        self.assertEqual(settings.timeout, 12.5)
        self.assertTrue(settings.text_ready and settings.vision_ready)

    def test_cli_override_beats_env_and_defaults_hold(self) -> None:
        settings = resolve_settings({"DEEPSEEK_MODEL": "deepseek-reasoner"}, text_model="deepseek-chat")
        self.assertEqual(settings.text_model, "deepseek-chat")
        self.assertEqual(settings.text_base_url, "https://api.deepseek.com/v1")
        self.assertEqual(settings.vision_model, "qwen-vl-max")
        self.assertEqual(resolve_settings({"LLM_TIMEOUT": "abc"}).timeout, 60.0)
        self.assertFalse(resolve_settings({}).text_ready)

    def test_describe_never_prints_values(self) -> None:
        text = "\n".join(LLMSettings(text_api_key="sk-secret", vision_api_key="sk-secret-2").describe())
        self.assertNotIn("sk-secret", text)
        self.assertIn("DEEPSEEK_API_KEY", text)
        self.assertIn("DASHSCOPE_API_KEY", text)
        self.assertIn("deepseek-chat", text)


# ---- 5. CLI：开关、自检与退出码 ----------------------------------------------

class TestCliOptional(TempCase):
    def _payload(self, name: str):
        return json.loads((self.tmp / name / "resume_decision.json").read_text(encoding="utf-8"))

    def _base_args(self, out: str) -> List[str]:
        return [
            "--env",
            str(self.env_file),
            "--jd",
            str(self.jd_path),
            "--project",
            str(self.project_path),
            "--out",
            str(self.tmp / out),
            "--quiet",
        ]

    def test_default_cli_run_is_rule_only(self) -> None:
        code = cli.main(
            ["--jd", str(self.jd_path), "--project", str(self.project_path), "--out", str(self.tmp / "rule"), "--quiet"]
        )
        self.assertEqual(code, cli.EXIT_OK)
        payload = self._payload("rule")
        self.assertIsNone(payload["llm"])
        self.assertEqual(payload["verdict"]["source"], "rule")
        self.assertFalse(payload["vision"]["enabled"])
        self.assertFalse(payload["params"]["llm"])
        self.assertEqual(payload["params"]["llm_blocks"], {})
        self.assertEqual(
            [step["action"] for step in payload["trace"]],
            ["ReadJD", "ReadProject", "ExtractRequirements", "RetrieveEvidence", "AdjustEvidence", "Judge"],
        )

    def test_llm_flag_writes_three_blocks_into_json(self) -> None:
        code = cli.main(["--llm", *self._base_args("llm")], text_client=TaskClient())
        self.assertEqual(code, cli.EXIT_OK)
        payload = self._payload("llm")
        report = payload["llm"]
        self.assertEqual(
            [item["source"] for item in report["blocks"]],
            [SOURCE_LLM_SUGGEST, SOURCE_LLM_INTERVIEW, SOURCE_LLM_POLISH],
        )
        self.assertEqual([item["status"] for item in report["blocks"]], [STATUS_OK, STATUS_IDLE, STATUS_OK])
        self.assertEqual([item["label"] for item in report["blocks"]], ["[LLM]"] * 3)
        self.assertEqual(report["calls"], 2)
        self.assertEqual(report["call_limit"], 10)
        self.assertEqual(report["timeout"], 30.0)
        self.assertEqual(report["retries"], 1)
        self.assertEqual(report["blocks"][1]["items"], [])
        self.assertTrue(report["blocks"][0]["text"])
        self.assertTrue(payload["verdict"]["headline"])
        self.assertEqual(payload["verdict"]["source"], "rule")
        self.assertTrue(payload["params"]["llm"])
        self.assertEqual(payload["params"]["llm_blocks"][SOURCE_LLM_POLISH], STATUS_OK)
        self.assertEqual(payload["decision"], "Stop")
        self.assertEqual([step["action"] for step in payload["trace"]][-1], "Finish")

    def test_vision_flag_writes_image_facts_and_the_step(self) -> None:
        code = cli.main(["--vision", *self._base_args("vision")], vision_client=FakeClient(VISION_REPLY))
        self.assertEqual(code, cli.EXIT_OK)
        payload = self._payload("vision")
        self.assertTrue(payload["vision"]["enabled"])
        self.assertEqual(payload["vision"]["facts_added"], 3)
        self.assertEqual(len(payload["vision"]["images"]), 1)
        self.assertEqual(payload["vision"]["images"][0]["ref"].startswith("图片:"), True)
        self.assertTrue(payload["params"]["vision"])
        actions = [step["action"] for step in payload["trace"]]
        self.assertIn("EnrichWithVision", actions)
        self.assertEqual(actions.index("EnrichWithVision"), 2)
        image_facts = [fact for fact in payload["project"]["facts"] if fact["source_file"].startswith("图片:")]
        self.assertEqual(len(image_facts), 3)

    def test_explicit_image_argument_is_used(self) -> None:
        extra = self.tmp / "extra.png"
        extra.write_bytes(b"\x89PNG\r\n\x1a\n")
        code = cli.main(
            ["--vision", "--image", str(extra), *self._base_args("image_extra")],
            vision_client=FakeClient(VISION_REPLY),
        )
        self.assertEqual(code, cli.EXIT_OK)
        payload = self._payload("image_extra")
        self.assertEqual(len(payload["vision"]["images"]), 2)

    def test_model_failure_keeps_exit_code_zero(self) -> None:
        code = cli.main(["--llm", *self._base_args("failed")], text_client=FakeClient(error=LLMError("HTTP 500")))
        self.assertEqual(code, cli.EXIT_OK)
        payload = self._payload("failed")
        self.assertEqual(
            [item["status"] for item in payload["llm"]["blocks"]], [STATUS_FAILED, STATUS_IDLE, STATUS_FAILED]
        )
        self.assertIn("HTTP 500", payload["llm"]["blocks"][0]["error"])
        self.assertTrue(payload["verdict"]["stop_reason"])
        self.assertEqual(payload["decision"], "Stop")

    def test_check_llm_without_key_exits_two(self) -> None:
        with NoRealKeys():
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(["--check-llm", "--env", str(self.env_file)])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("DEEPSEEK_API_KEY", buffer.getvalue())

    def test_check_llm_with_injected_client_exits_zero(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--check-llm", "--env", str(self.env_file), "--quiet"], text_client=FakeClient("pong"))
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("通过", buffer.getvalue())

    def test_check_llm_reports_vision_layer_failure(self) -> None:
        with NoRealKeys():
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(
                    ["--check-llm", "--vision", "--env", str(self.env_file), "--quiet"],
                    text_client=FakeClient("pong"),
                )
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("Qwen-VL", buffer.getvalue())

    def test_missing_env_file_exits_two(self) -> None:
        code = cli.main(
            [
                "--llm",
                "--env",
                str(self.tmp / "nope.env"),
                "--jd",
                str(self.jd_path),
                "--project",
                str(self.project_path),
                "--quiet",
            ]
        )
        self.assertEqual(code, cli.EXIT_ERROR)

    def test_terminal_output_prints_key_names_only(self) -> None:
        secret_env = self._write("secret.env", "DEEPSEEK_API_KEY=sk-super-secret\n")
        with NoRealKeys():
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                cli.main(
                    ["--llm", "--env", str(secret_env), *self._base_args("secret")[2:-1]],
                    text_client=TaskClient(),
                )
        output = buffer.getvalue()
        self.assertNotIn("sk-super-secret", output)
        self.assertIn("DEEPSEEK_API_KEY", output)


if __name__ == "__main__":
    unittest.main()
