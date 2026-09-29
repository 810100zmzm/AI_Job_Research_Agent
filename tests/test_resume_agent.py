"""Resume Agent（LangGraph）测试：事实抽取 + 证据等级 + 否定/背景识别 + Agent Loop 四要素。

全程不联网：网址抓取注入假 opener，图片走注入的假客户端；文本 / 文件两条分支本来就不联网。

运行方式（项目根目录）：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jd_agent import cli, tools  # noqa: E402
from jd_agent.services.jd_source import (  # noqa: E402
    SOURCE_FILE,
    SOURCE_IMAGE,
    SOURCE_TEXT,
    SOURCE_URL,
    SourceError,
    detect_sources,
    unusable_notes,
)
from jd_agent.core.llm import LLMError  # noqa: E402
from jd_agent.domain.resume_facts import (  # noqa: E402
    FACT_BLOCK_BACKGROUND,
    FACT_BLOCK_NEGATED,
    ROLE_CLAIM,
    ROLE_EDU,
    ROLE_EXP,
    ROLE_META,
    ROLE_SKILL,
    background_hit,
    extract_resume,
    is_background_section,
    is_course,
    judge_level,
    negated_capability_keys,
    negation_hit,
    plan_hit,
    section_role,
    summarize,
)
from jd_agent.agents.resume_graph import (  # noqa: E402
    MIN_FACTS,
    MIN_SOLID_FACTS,
    NODE_NAMES,
    RESUME_IMAGE_SYSTEM_PROMPT,
    ResumeRequest,
    compiled_graph,
    render_resume_console,
    render_resume_markdown,
    render_resume_trace,
    run_resume_agent,
)
from jd_agent.core.schema import (  # noqa: E402
    DECISION_ADJUST,
    DECISION_ASK,
    DECISION_CONTINUE,
    DECISION_STOP,
    DECISIONS,
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_RESULT,
)
from jd_agent.tools.resume_evidence import (  # noqa: E402
    RESUME_EVIDENCE_TOOL,
    render_evidence_markdown,
    summarize_evidence,
)
from jd_agent.tools.resume_negation import (  # noqa: E402
    RESUME_NEGATION_TOOL,
    RULE_LINES,
    explain_line,
    render_negation_markdown,
    screen_facts,
)
from jd_agent.tools.resume_struct import (  # noqa: E402
    RESUME_STRUCT_TOOL,
    render_struct_markdown,
    structure_resume_text,
)

SAMPLE_RESUME = """# 李小明 · 个人简历

> 求职意向：AI 应用开发实习生

## 基本信息

- 姓名：李小明
- 求职意向：AI 大模型应用开发实习生
- 届别与可实习时间：2027 届本科在读，每周 4 天

## 教育背景

- 某某大学 · 软件工程 · 本科在读（2027 届），GPA 3.7/4.0
- 主修课程：数据结构、机器学习、数据库原理

## 项目经历

### 智能问答助手 ｜ 后端开发

- 课程项目：用 FastAPI 搭建问答接口，独立完成接口封装与联调，接口响应时间从 800ms 降到 120ms
- 了解向量数据库，没做过 Docker 部署

## 技能清单

- 熟练使用 Python，写过数据清洗脚本
- 熟悉 Git 基本操作
- 未接触过 Kubernetes

## 自我评价

- 学习能力强，能在 3 天内上手新工具
"""

# 全是「了解 / 熟悉」级描述：严格口径下一条硬证据都没有，必须走一次 Adjust
WEAK_RESUME = """# 王五

## 技能清单

- 熟悉 Python
- 了解 RAG 与向量数据库
- 会用 Excel 做透视表
- 了解大模型基本原理
"""

# 每条都写了「没做过」：可当证据的事实是 0，应该停下来问人
NEGATED_RESUME = """# 赵六

## 技能清单

- 未接触过向量数据库
- 没做过 Docker 部署
- 尚未有互联网公司实习经历
"""

# 整节都是背景 / 目标：条目一条都不能算「我做过什么」
BACKGROUND_RESUME = """# 钱七

## 项目简介

- 一个给学院同学用的 AI 问答小工具，开学初做的。
- 想解决新生问重复问题太多的问题。

## 技术方案

- 前端用 Streamlit，后端调用大模型 API。
- 用向量数据库做知识库检索。

## 我的角色

- 参与了项目的开发和调试。

## 结果

- 目前还在试用阶段。
"""

CLEAN_PAGE = """<html><body><h2>教育背景</h2><ul>
<li>某某大学 · 计算机科学与技术 · 本科在读（2027 届）</li>
</ul><h2>项目经历</h2><ul>
<li>个人项目：文档问答助手，独立完成接口封装，命中率从 0.71 提到 0.89，已开源到 GitHub</li>
<li>熟练使用 Python 与 FastAPI，写过数据清洗脚本</li>
</ul></body></html>"""

SHELL_PAGE = """<html><body><div id="app"></div><script>render()</script></body></html>"""

IMAGE_REPLY = """## 教育背景
- 某某大学 · 软件工程 · 本科在读

## 项目经历
- 用 PyTorch 复现 ResNet50，独立完成训练与调参，准确率提升到 91%

## 技能清单
- 熟悉 Python 与 Git，写过自动化脚本
"""


class FakeOpener:
    """假抓取器：按 url 返回预置页面，并记录被请求过哪些地址。"""

    def __init__(self, pages=None, error: Exception = None):
        self.pages = pages or {}
        self.error = error
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        html = self.pages.get(url, SHELL_PAGE)
        return html.encode("utf-8"), "utf-8", url


class FakeVision:
    """假多模态客户端：返回预置 Markdown，并记录收到过什么。"""

    def __init__(self, reply: str = IMAGE_REPLY, error: Exception = None):
        self.reply = reply
        self.error = error
        self.calls = []

    def chat(self, messages, model=None, max_tokens=None, temperature=None) -> str:
        self.calls.append({"messages": messages, "model": model})
        if self.error is not None:
            raise self.error
        return self.reply


def make_doc(text: str = SAMPLE_RESUME, source: str = "简历.md"):
    """把一段简历文本解析成文档：测试里反反复复要用的一句话简写。"""
    return extract_resume(text, source_file=source)


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.tmp / name
        path.write_text(text, encoding=encoding)
        return path

    def assert_trace_shape(self, state) -> None:
        """每条 Trace 都必须四要素齐全，Decision 只能是那四种。"""
        self.assertTrue(state.trace, "Trace 不能是空的")
        for index, step in enumerate(state.trace, start=1):
            self.assertEqual(step.index, index, "Trace 序号必须连续")
            for field in (step.action, step.observation, step.state_update, step.decision_reason):
                self.assertTrue(str(field).strip(), f"Trace 四要素缺一：{step}")
            self.assertIn(step.decision, DECISIONS)

    def doc(self, text: str = SAMPLE_RESUME, source: str = "简历.md"):
        return make_doc(text, source)


class TestNegationAndBackground(unittest.TestCase):
    """否定 / 背景识别：这一层存在的理由就是不让「写了但没做过」混进证据。"""

    def test_negation_before_the_keyword_is_caught(self) -> None:
        keys, marker = negated_capability_keys("未接触过向量数据库")
        self.assertTrue(keys)
        self.assertEqual(marker, "未接触")

    def test_negation_after_the_keyword_is_caught(self) -> None:
        keys, _ = negated_capability_keys("了解 TensorFlow 但没做过完整项目")
        self.assertIn("dl_framework", keys)                      # 否定写在后面也算数

    def test_punctuation_stops_the_negation_scope(self) -> None:
        keys, _ = negated_capability_keys("熟悉 Python，没写过 C++")
        self.assertEqual(keys, [])                               # 逗号后面的否定不连坐前面

    def test_absence_of_experience_is_caught(self) -> None:
        self.assertTrue(negation_hit("尚未有互联网公司实习经历"))
        self.assertTrue(negation_hit("无相关实习经验"))
        self.assertTrue(negation_hit("实习经历：无"))

    def test_plain_positives_are_not_flagged(self) -> None:
        for line in ("独立完成接口封装，已开源", "熟练使用 Python 写过脚本", "负责数据处理与建模"):
            self.assertEqual(negation_hit(line), "", line)
            self.assertEqual(negated_capability_keys(line)[0], [], line)

    def test_background_and_plan_are_caught(self) -> None:
        self.assertEqual(background_hit("项目背景：学院需要有人统计每周数据"), "项目背景：")
        self.assertEqual(background_hit("本项目旨在解决长文本纠错问题"), "旨在")
        self.assertEqual(plan_hit("计划学习 Go 语言做重构"), "计划学习")

    def test_planned_action_is_still_an_action(self) -> None:
        for line in ("按计划完成了实验", "原计划做了三版方案"):
            self.assertEqual(plan_hit(line), "", line)

    def test_course_line_is_capped_at_mention(self) -> None:
        self.assertTrue(is_course("主修课程：数据结构、机器学习"))
        level, basis, metrics, _ = judge_level("培训期间完成了数据标注")
        self.assertEqual(level, EVIDENCE_ACTION)                 # 判等级本身先看到「完成」
        self.assertEqual(metrics, [])
        fact = make_doc("## 技能清单\n\n- 培训期间完成了数据标注\n").facts[0]
        self.assertEqual(fact.level, EVIDENCE_MENTION)           # 课程口径把它封顶
        self.assertIn("课程", fact.basis)

    def test_course_line_with_metrics_keeps_its_result(self) -> None:
        fact = make_doc("## 项目经历\n\n- 课程项目：独立完成调参，准确率从 72% 提升到 85%\n").facts[0]
        self.assertEqual(fact.level, EVIDENCE_RESULT)
        self.assertFalse(fact.blocked)

    def test_award_and_metric_and_action_map_to_the_three_levels(self) -> None:
        cases = {
            "校级一等奖学金": EVIDENCE_RESULT,
            "准确率从 72% 提升到 85%": EVIDENCE_RESULT,
            "独立完成接口封装与联调": EVIDENCE_ACTION,
            "熟悉 Git 基本操作": EVIDENCE_MENTION,
        }
        for text, expected in cases.items():
            self.assertEqual(judge_level(text)[0], expected, text)

    def test_section_roles(self) -> None:
        cases = {
            "基本信息": ROLE_META,
            "教育背景": ROLE_EDU,
            "项目经历": ROLE_EXP,
            "实习经历": ROLE_EXP,
            "专业技能": ROLE_SKILL,
            "自我评价": ROLE_CLAIM,
            "一、个人信息": ROLE_META,
            "七、自我评价": ROLE_CLAIM,
            "八、别的什么": "other",
        }
        for title, expected in cases.items():
            self.assertEqual(section_role(title), expected, title)

    def test_education_section_is_not_treated_as_background(self) -> None:
        """「教育背景」是学历信息，不能因为名字带「背景」就整节作废。"""
        self.assertFalse(is_background_section("教育背景"))
        self.assertFalse(is_background_section("个人简介"))
        self.assertTrue(is_background_section("项目简介"))
        self.assertTrue(is_background_section("课题背景"))
        self.assertTrue(is_background_section("研究背景"))


class TestResumeExtraction(TempCase):
    """把简历拆成事实条目：抬头信息、章节、行号、挡掉的条目。"""

    def setUp(self) -> None:
        super().setUp()
        self.doc = extract_resume(SAMPLE_RESUME, source_file="简历.md")

    def test_meta_sections_become_header_info(self) -> None:
        keys = [key for key, _ in self.doc.meta]
        self.assertIn("姓名", keys)
        self.assertIn("求职意向", keys)
        self.assertEqual(self.doc.meta_value("姓名"), "李小明")
        self.assertEqual(self.doc.meta_value("求职意向"), "AI 大模型应用开发实习生")

    def test_document_title_and_sections(self) -> None:
        self.assertEqual(self.doc.title, "李小明")
        self.assertEqual(
            [section.title for section in self.doc.sections],
            ["教育背景", "项目经历", "技能清单", "自我评价"],
        )

    def test_facts_keep_the_original_line_numbers(self) -> None:
        line = self.doc.facts[0].line_no
        self.assertEqual(SAMPLE_RESUME.splitlines()[line - 1].strip().lstrip("- "), self.doc.facts[0].text)
        self.assertTrue(self.doc.facts[0].ref.endswith(f"::L{line}"))

    def test_negated_lines_never_become_evidence(self) -> None:
        blocked = {fact.text: fact for fact in self.doc.facts if fact.blocked}
        self.assertIn("了解向量数据库，没做过 Docker 部署", blocked)
        self.assertIn("未接触过 Kubernetes", blocked)
        for fact in blocked.values():
            self.assertEqual(fact.blocked, FACT_BLOCK_NEGATED)
            self.assertFalse(fact.usable)
            self.assertIn("反向证据", fact.block_reason)

    def test_one_sided_tf_line_is_blocked_but_the_other_line_survives(self) -> None:
        doc = extract_resume(
            "## 技能清单\n\n- 了解 TensorFlow 但没做过完整项目\n- 熟练使用 Python，写过清洗脚本\n",
            source_file="简历.md",
        )
        self.assertEqual(doc.facts[0].blocked, FACT_BLOCK_NEGATED)
        self.assertTrue(doc.facts[1].usable)

    def test_background_section_blocks_its_bullets(self) -> None:
        doc = extract_resume(BACKGROUND_RESUME, source_file="简历.md")
        intro = next(section for section in doc.sections if section.title == "项目简介")
        self.assertTrue(intro.background)
        self.assertTrue(all(fact.blocked == FACT_BLOCK_BACKGROUND for fact in intro.facts))
        self.assertTrue(any("整节" in fact.block_reason for fact in intro.facts))

    def test_sentence_level_background_is_blocked_too(self) -> None:
        doc = extract_resume(
            "## 项目经历\n\n- 项目背景：学院需要一个统计工具\n- 独立完成统计脚本，覆盖 12 个班级\n",
            source_file="简历.md",
        )
        self.assertEqual(doc.facts[0].blocked, FACT_BLOCK_BACKGROUND)
        self.assertTrue(doc.facts[1].usable)

    def test_subsection_titles_are_kept(self) -> None:
        subsections = {fact.subsection for fact in self.doc.facts}
        self.assertIn("智能问答助手 ｜ 后端开发", subsections)

    def test_query_lines_are_meta_not_facts(self) -> None:
        self.assertNotIn("求职意向：AI 应用开发实习生", [fact.text for fact in self.doc.facts])

    def test_summary_counts_strict_and_relaxed(self) -> None:
        strict = summarize(self.doc.facts)
        relaxed = summarize(self.doc.facts, relaxed=True)
        self.assertEqual(len(strict.blocked), 2)
        self.assertEqual(len(strict.usable), len(relaxed.usable))
        self.assertLess(strict.evidence_count(), relaxed.evidence_count())
        for fact in relaxed.evidence:
            self.assertTrue(fact.usable)                          # 放宽不会把挡掉的放进来
        self.assertIn("本次按严格口径计入", strict.conclusion)

    def test_tables_and_numbered_bullets_are_read(self) -> None:
        doc = extract_resume(
            "## 项目经历\n\n| 项目 | 内容 |\n| --- | --- |\n| 周报助手 | 独立完成接口封装 |\n\n1. 用 Python 写了清洗脚本\n",
            source_file="简历.md",
        )
        texts = [fact.text for fact in doc.facts]
        self.assertIn("周报助手：独立完成接口封装", texts)
        self.assertIn("用 Python 写了清洗脚本", texts)


class TestResumeStructTool(TempCase):
    def test_runs_on_text_and_reports_a_summary(self) -> None:
        result = RESUME_STRUCT_TOOL.run(text=SAMPLE_RESUME, source_file="简历.md")
        self.assertTrue(result.ok, result.error)
        labels = [label for label, _ in result.summary]
        self.assertIn("章节", labels)
        self.assertIn("事实条目", labels)

    def test_runs_on_a_file_and_writes_markdown(self) -> None:
        path = self.write("cv.md", SAMPLE_RESUME)
        result = RESUME_STRUCT_TOOL.run(path=path, out_dir=self.tmp / "docs")
        self.assertTrue(result.ok, result.error)
        written = result.written[0]
        self.assertTrue(written.is_file())
        self.assertIn("简历结构化", written.read_text(encoding="utf-8"))

    def test_needs_exactly_one_source(self) -> None:
        self.assertFalse(RESUME_STRUCT_TOOL.run().ok)
        self.assertFalse(RESUME_STRUCT_TOOL.run(text=SAMPLE_RESUME, path=self.tmp / "cv.md").ok)
        self.assertFalse(RESUME_STRUCT_TOOL.run(text="   ").ok)

    def test_empty_resume_is_reported_not_raised(self) -> None:
        result = RESUME_STRUCT_TOOL.run(text="# 张三\n")
        self.assertFalse(result.ok)
        self.assertIn("没有读到任何事实条目", result.error)

    def test_render_keeps_sections_and_line_numbers(self) -> None:
        markdown = render_struct_markdown(structure_resume_text(SAMPLE_RESUME, source_file="简历.md"))
        self.assertIn("## 项目经历", markdown)
        self.assertIn("### 智能问答助手 ｜ 后端开发", markdown)
        self.assertIn("（L13）", markdown)          # 行号指的是原文里的真实行
        self.assertIn("整节是背景", render_struct_markdown(extract_resume(BACKGROUND_RESUME)))

    def test_struct_does_not_judge_levels(self) -> None:
        """结构化 Tool 只管结构：等级是证据 Tool 的事，这里不该出现等级字样。"""
        markdown = render_struct_markdown(structure_resume_text(SAMPLE_RESUME))
        for label in ("有结果", "有动作", "仅提及"):
            self.assertNotIn(label, markdown)


class TestResumeEvidenceTool(TempCase):
    def test_strict_and_relaxed_counts(self) -> None:
        doc = extract_resume(SAMPLE_RESUME, source_file="简历.md")
        strict = summarize_evidence(doc.facts)
        relaxed = summarize_evidence(doc.facts, relaxed=True)
        self.assertEqual(strict.counts[EVIDENCE_RESULT], 1)
        self.assertEqual(strict.counts[EVIDENCE_ACTION], 1)
        self.assertEqual(len(strict.weak), 4)
        self.assertEqual(relaxed.evidence_count(True), len(relaxed.usable))

    def test_render_contains_both_tables(self) -> None:
        doc = extract_resume(SAMPLE_RESUME, source_file="简历.md")
        markdown = render_evidence_markdown(doc, summarize_evidence(doc.facts))
        self.assertIn("证据等级清单", markdown)
        self.assertIn("| # | 事实 | 等级 | 命中能力 | 依据 | 出处 |", markdown)
        self.assertIn("## 不能当证据的内容", markdown)
        self.assertIn("没做过 Docker 部署", markdown)

    def test_runs_on_a_file_and_writes_markdown(self) -> None:
        path = self.write("cv.md", SAMPLE_RESUME)
        result = RESUME_EVIDENCE_TOOL.run(path=path, out_dir=self.tmp / "docs")
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.written[0].is_file())
        labels = dict(result.summary)
        self.assertIn("可当证据", labels)
        self.assertIn("本次计入", labels)

    def test_relaxed_flag_is_reported(self) -> None:
        result = RESUME_EVIDENCE_TOOL.run(text=WEAK_RESUME, relaxed=True)
        self.assertTrue(result.ok, result.error)
        self.assertIn("放宽", dict(result.summary)["本次计入"])

    def test_needs_exactly_one_source(self) -> None:
        self.assertFalse(RESUME_EVIDENCE_TOOL.run().ok)
        self.assertFalse(RESUME_EVIDENCE_TOOL.run(path=self.tmp / "nope.md").ok)


class TestResumeNegationTool(TempCase):
    def test_screen_groups_blocked_facts(self) -> None:
        doc = extract_resume(SAMPLE_RESUME, source_file="简历.md")
        report = screen_facts(doc.facts)
        self.assertEqual(len(report.negated), 2)
        self.assertEqual(report.background, [])
        self.assertEqual(len(report.kept), len(doc.facts) - 2)
        self.assertEqual(len(report.facts_of(FACT_BLOCK_NEGATED)), 2)
        self.assertIn("被挡住", report.headline)

    def test_background_section_shows_up_as_background(self) -> None:
        doc = extract_resume(BACKGROUND_RESUME, source_file="简历.md")
        report = screen_facts(doc.facts)
        self.assertTrue(report.background)
        self.assertEqual(report.negated, [])

    def test_explain_line_answers_for_a_single_sentence(self) -> None:
        kind, reason = explain_line("没做过 Docker 部署")
        self.assertEqual(kind, FACT_BLOCK_NEGATED)
        self.assertIn("反向证据", reason)
        self.assertEqual(explain_line("独立完成接口封装，已开源"), ("", ""))
        self.assertEqual(explain_line("计划学习 Go")[0], FACT_BLOCK_BACKGROUND)

    def test_render_lists_rules_and_blocked_rows(self) -> None:
        doc = extract_resume(SAMPLE_RESUME, source_file="简历.md")
        markdown = render_negation_markdown(doc, screen_facts(doc.facts))
        self.assertIn("否定与背景识别", markdown)
        self.assertIn("判定规则（原文照抄，可直接核对）", markdown)
        for rule in RULE_LINES:
            self.assertIn(rule, markdown)
        self.assertIn("| # | 原文 | 判定 | 为什么不算证据 | 出处 |", markdown)

    def test_runs_on_a_file_and_writes_markdown(self) -> None:
        path = self.write("cv.md", NEGATED_RESUME)
        result = RESUME_NEGATION_TOOL.run(path=path, out_dir=self.tmp / "docs")
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.written[0].is_file())
        self.assertEqual(dict(result.summary)["挡住"], "3 条（否定 3 / 背景 0）")

    def test_needs_exactly_one_source(self) -> None:
        self.assertFalse(RESUME_NEGATION_TOOL.run().ok)
        self.assertFalse(RESUME_NEGATION_TOOL.run(path=self.tmp / "nope.md").ok)


class TestResumeAgentLoop(TempCase):
    def test_text_source_runs_the_whole_loop(self) -> None:
        state = run_resume_agent(ResumeRequest(text=SAMPLE_RESUME), out_dir=self.tmp, stem="facts")
        self.assertEqual(
            [step.action for step in state.trace],
            ["DetectSource", "ReadText", "StructureResume", "ScreenFacts", "JudgeEvidence", "WriteFacts"],
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertEqual(len(state.facts), 8)
        self.assertEqual(state.blocked_count, 2)
        self.assertTrue((self.tmp / "facts.md").is_file())
        self.assert_trace_shape(state)

    def test_screen_runs_before_evidence(self) -> None:
        """顺序不能反：先挡「没做过」，再判等级。"""
        actions = [step.action for step in run_resume_agent(ResumeRequest(text=SAMPLE_RESUME)).trace]
        self.assertLess(actions.index("ScreenFacts"), actions.index("JudgeEvidence"))

    def test_text_source_never_touches_the_network(self) -> None:
        opener = FakeOpener(error=AssertionError("文本分支不该发请求"))
        vision = FakeVision(error=AssertionError("文本分支不该调模型"))
        state = run_resume_agent(ResumeRequest(text=SAMPLE_RESUME), fetcher=opener, vision_client=vision)
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertEqual(opener.calls, [])
        self.assertEqual(vision.calls, [])

    def test_file_source_points_refs_at_the_real_file(self) -> None:
        path = self.write("cv.md", SAMPLE_RESUME)
        state = run_resume_agent(ResumeRequest(files=(str(path),)), out_dir=self.tmp, stem="facts")
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertIn("cv.md::L", state.facts[0].ref)
        self.assertIn("cv.md::L", (self.tmp / "facts.md").read_text(encoding="utf-8"))

    def test_strict_scope_relaxes_once_and_rejudges(self) -> None:
        state = run_resume_agent(ResumeRequest(text=WEAK_RESUME))
        actions = [step.action for step in state.trace]
        self.assertEqual(actions.count("RelaxScope"), 1)
        self.assertEqual(actions.count("JudgeEvidence"), 2)
        # RelaxScope 写在 Adjust 那一步，紧随其后的 JudgeEvidence 再重判一次，最后才是 WriteFacts
        self.assertEqual([step.decision for step in state.trace][-3], DECISION_ADJUST)
        self.assertTrue(state.relaxed)
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertIn("放宽口径", state.stop_reason)

    def test_no_relax_when_something_solid_exists(self) -> None:
        state = run_resume_agent(ResumeRequest(text=SAMPLE_RESUME))
        self.assertGreaterEqual(state.solid_count, MIN_SOLID_FACTS)
        self.assertNotIn("RelaxScope", [step.action for step in state.trace])

    def test_relax_never_unblocks_negation(self) -> None:
        """放宽只放宽证据强弱：被否定 / 背景挡掉的条目，放宽之后照样不算证据。"""
        state = run_resume_agent(ResumeRequest(text=NEGATED_RESUME))
        self.assertTrue(state.relaxed)
        self.assertEqual(state.evidence_count, 0)
        self.assertEqual(state.blocked_count, 3)
        for fact in state.summary.evidence:
            self.assertTrue(fact.usable)

    def test_all_blocked_asks_a_question(self) -> None:
        state = run_resume_agent(ResumeRequest(text=NEGATED_RESUME))
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertEqual(state.trace[-1].action, "AskSource")
        self.assertIn("全被否定或背景挡住", state.question)
        self.assertEqual(state.markdown, "")
        self.assert_trace_shape(state)

    def test_thin_resume_asks_a_question(self) -> None:
        state = run_resume_agent(ResumeRequest(text="# 钱七\n\n## 技能清单\n\n- 熟悉 Python\n"))
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertLess(state.evidence_count, MIN_FACTS)
        self.assertIn(str(MIN_FACTS), state.question)

    def test_no_source_at_all_asks_to_paste_something(self) -> None:
        state = run_resume_agent(ResumeRequest(text=""))
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertIn("没有取到任何简历内容", state.question)

    def test_background_only_resume_still_writes_what_it_found(self) -> None:
        state = run_resume_agent(ResumeRequest(text=BACKGROUND_RESUME))
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertEqual(state.blocked_count, 2)
        self.assertIn("整节都是", state.markdown)

    def test_clean_url_page_needs_no_adjust(self) -> None:
        opener = FakeOpener(pages={"https://x.com/cv": CLEAN_PAGE})
        state = run_resume_agent(ResumeRequest(urls=("https://x.com/cv",)), fetcher=opener)
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertEqual(opener.calls, ["https://x.com/cv"])
        self.assertEqual([step.action for step in state.trace][1], "FetchUrl")
        self.assertTrue(any("命中率" in fact.text for fact in state.facts))

    def test_shell_page_asks_instead_of_inventing_content(self) -> None:
        opener = FakeOpener(pages={"https://x.com/cv": SHELL_PAGE})
        state = run_resume_agent(ResumeRequest(urls=("https://x.com/cv",)), fetcher=opener)
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertIn("没有取到任何简历内容", state.question)

    def test_failed_fetch_degrades_and_still_uses_the_rest(self) -> None:
        opener = FakeOpener(error=SourceError("网络不可达：连接超时"))
        state = run_resume_agent(
            ResumeRequest(text=SAMPLE_RESUME, urls=("https://x.com/cv",)), fetcher=opener
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue(state.failed_chunks)
        self.assertIn("没取到", state.stop_reason)
        self.assertIn("熟练使用 Python", state.markdown)

    def test_image_source_uses_the_vision_client_and_the_resume_prompt(self) -> None:
        shot = self.write("shot.png", "fake-png-bytes")
        vision = FakeVision()
        state = run_resume_agent(ResumeRequest(images=(str(shot),)), vision_client=vision)
        self.assertEqual(
            [step.action for step in state.trace][:2], ["DetectSource", "OcrImage"]
        )
        self.assertEqual(len(vision.calls), 1)
        self.assertEqual(vision.calls[0]["messages"][0]["content"], RESUME_IMAGE_SYSTEM_PROMPT)
        self.assertIn("简历", vision.calls[0]["messages"][0]["content"])
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue(any("91%" in fact.text or "91" in fact.text for fact in state.facts))

    def test_failed_image_call_is_degraded_then_asked(self) -> None:
        shot = self.write("shot.png", "fake-png-bytes")
        vision = FakeVision(error=LLMError("HTTP 500"))
        state = run_resume_agent(ResumeRequest(images=(str(shot),)), vision_client=vision)
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertTrue(state.notes)
        self.assert_trace_shape(state)

    def test_merged_sources_note_their_line_numbers(self) -> None:
        path = self.write("cv.md", SAMPLE_RESUME)
        state = run_resume_agent(ResumeRequest(text=WEAK_RESUME, files=(str(path),)), out_dir=self.tmp)
        self.assertTrue(any("行号按合并后的文本计" in note for note in state.notes))

    def test_graph_keeps_the_documented_nodes(self) -> None:
        nodes = set(compiled_graph().get_graph().nodes)
        self.assertTrue(set(NODE_NAMES).issubset(nodes), f"图里缺少节点：{set(NODE_NAMES) - nodes}")

    def test_renders_console_and_trace_without_network(self) -> None:
        state = run_resume_agent(ResumeRequest(text=SAMPLE_RESUME))
        console = "\n".join(render_resume_console(state))
        self.assertIn("Action      : DetectSource", console)
        self.assertIn("Decision    : Stop", console)
        trace = render_resume_trace(state)
        self.assertIn("| # | Action |", trace)
        self.assertIn("**StopReason**", trace)
        self.assertIn("## 被挡住的内容", trace)

    def test_markdown_is_rerenderable_and_judge_readable(self) -> None:
        state = run_resume_agent(ResumeRequest(text=SAMPLE_RESUME))
        first = render_resume_markdown(state.doc, state.summary, source_line="简历.md")
        second = render_resume_markdown(state.doc, state.summary, source_line="简历.md")
        self.assertEqual(first, second)                          # 同样的输入 -> 同样的输出
        for markdown in (first, state.markdown):                 # 默认写入的版本也一样可读
            self.assertIn("**姓名**：李小明", markdown)
            self.assertIn("## 不能当证据的内容（挡住「写了但没做过」）", markdown)

    def test_source_detection_helpers_still_apply(self) -> None:
        sources = detect_sources(text=SAMPLE_RESUME, files=("nope.md",), vision_ready=False)
        kinds = [source.kind for source in sources]
        self.assertEqual(kinds, [SOURCE_TEXT, SOURCE_FILE])
        self.assertTrue(unusable_notes(sources))
        self.assertEqual(detect_sources(images=("a.png",), vision_ready=False)[0].kind, SOURCE_IMAGE)
        self.assertEqual(detect_sources(urls=("https://x.com/cv",))[0].kind, SOURCE_URL)


class TestResumeAgentCLI(TempCase):
    def run_cli(self, argv) -> int:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(argv)
        self.last_output = buffer.getvalue()
        return code

    def test_agent_mode_writes_markdown_and_trace(self) -> None:
        path = self.write("cv.md", SAMPLE_RESUME)
        out = self.tmp / "facts.md"
        code = self.run_cli(["--resume-agent", "--cv-file", str(path), "--cv-out", str(out), "--cv-trace"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue(out.is_file())
        self.assertTrue((self.tmp / "facts.trace.md").is_file())
        self.assertIn("简历事实清单", out.read_text(encoding="utf-8"))
        self.assertIn("Action      : ScreenFacts", self.last_output)

    def test_text_input_stays_offline(self) -> None:
        code = self.run_cli(["--resume-agent", "--cv-text", SAMPLE_RESUME, "--cv-out", str(self.tmp / "f.md")])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("[输入]", self.last_output)
        self.assertNotIn("[配置]", self.last_output)             # 文本分支不读 .env

    def test_agent_mode_without_source_is_an_input_error(self) -> None:
        self.assertEqual(self.run_cli(["--resume-agent", "--quiet"]), cli.EXIT_ERROR)

    def test_unusable_source_is_an_input_error_with_the_reason(self) -> None:
        code = self.run_cli(["--resume-agent", "--cv-image", str(self.tmp / "nope.png")])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("没有一个能读的简历来源", self.last_output)

    def test_quiet_run_prints_nothing(self) -> None:
        path = self.write("cv.md", SAMPLE_RESUME)
        out = self.tmp / "facts.md"
        code = self.run_cli(["--resume-agent", "--cv-file", str(path), "--cv-out", str(out), "--quiet"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(self.last_output.strip(), "")
        self.assertTrue(out.is_file())

    def test_thin_input_leaves_with_exit_code_three(self) -> None:
        code = self.run_cli(["--resume-agent", "--cv-text", "# 钱七\n\n## 技能清单\n\n- 熟悉 Python\n"])
        self.assertEqual(code, cli.EXIT_ASK)
        self.assertIn("需要你回答 1 个问题", self.last_output)

    def test_cv_title_overrides_the_document_title(self) -> None:
        code = self.run_cli(
            ["--resume-agent", "--cv-text", SAMPLE_RESUME, "--cv-title", "我的简历",
             "--cv-out", str(self.tmp / "f.md")]
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("我的简历", (self.tmp / "f.md").read_text(encoding="utf-8"))

    def test_new_tools_are_registered(self) -> None:
        self.assertIs(tools.TOOL_BY_SLUG["resume-struct"], tools.RESUME_STRUCT_TOOL)
        self.assertIs(tools.TOOL_BY_SLUG["resume-evidence"], tools.RESUME_EVIDENCE_TOOL)
        self.assertIs(tools.TOOL_BY_SLUG["resume-negation"], tools.RESUME_NEGATION_TOOL)
        slugs = [tool.slug for tool in tools.TOOLS]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertEqual(len(tools.tool_help_lines()), len(slugs))

    def test_cv_file_flag_does_not_collide_with_resume_file(self) -> None:
        """--cv-file 是 Resume Agent 的输入，--resume-file 是排版工具的输入，两个各管各的。"""
        parser = cli.build_arg_parser()
        args = parser.parse_args(["--resume-agent", "--cv-file", "a.md"])
        self.assertEqual(args.cv_files, ["a.md"])
        self.assertIsNone(args.resume_file)
        args = parser.parse_args(["--build-resume", "--resume-file", "b.md"])
        self.assertEqual(args.resume_file, "b.md")
        self.assertIsNone(args.cv_files)


if __name__ == "__main__":
    unittest.main()
