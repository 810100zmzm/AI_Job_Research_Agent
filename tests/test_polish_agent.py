"""Polish Agent（LangGraph）测试：写作简报 + 能力词典匹配 + 三块 LLM 内容 + Agent Loop 四要素。

全程不联网：网址抓取注入假 opener，图片注入假多模态客户端，三块 LLM 内容注入假文本客户端；
文本 / 文件两条分支本来就不联网。

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
from jd_agent.domain.jd import parse_jd_text  # noqa: E402
from jd_agent.services.jd_source import (  # noqa: E402
    JD_IMAGE_SYSTEM_PROMPT,
    SOURCE_FILE,
    SOURCE_IMAGE,
    SOURCE_TEXT,
    SOURCE_URL,
    SourceError,
    detect_sources,
    unusable_notes,
)
from jd_agent.core.llm import LLMError  # noqa: E402
from jd_agent.agents.polish_brief import (  # noqa: E402
    MAX_RISKS,
    MIN_FACTS,
    build_brief,
    capability_hits,
    capability_keys,
    capability_names,
    group_hits,
    link_requirements,
    link_rows,
)  # noqa: E402
from jd_agent.agents.polish_graph import (  # noqa: E402
    LLM_STEPS,
    MIN_SOLID_FACTS,
    NODE_NAMES,
    ROUTES,
    SIDE_CV,
    SIDE_JD,
    SIDE_LABEL,
    PolishRequest,
    compiled_graph,
    render_polish_console,
    render_polish_markdown,
    render_polish_trace,
    run_polish_agent,
)
from jd_agent.agents.polish_llm import (  # noqa: E402
    INTERVIEW_COUNT,
    MAX_FACTS,
    MAX_REQUIREMENTS,
    TASK_ACTION,
    TASK_ORDER,
    TASK_TITLE,
    build_brief_payload,
    build_user_message,
    new_report,
    parse_draft,
    parse_interview,
    parse_polish,
    run_task,
    unavailable_report,
)
from jd_agent.domain.resume_facts import extract_resume, summarize  # noqa: E402
from jd_agent.agents.resume_graph import RESUME_IMAGE_SYSTEM_PROMPT  # noqa: E402
from jd_agent.core.schema import (  # noqa: E402
    DECISION_ADJUST,
    DECISION_ASK,
    DECISION_STOP,
    DECISIONS,
    RISK_CONFLICT,
    RISK_GAP,
    RISK_PROBE,
    RISK_WORDING,
    SOURCE_LLM_INTERVIEW,
    SOURCE_LLM_POLISH,
    SOURCE_LLM_SUGGEST,
    LlmBlock,
)  # noqa: E402
from jd_agent.tools.cap_match import (  # noqa: E402
    CAP_MATCH_TOOL,
    compare_counts,
    render_cap_match_markdown,
)  # noqa: E402
from jd_agent.tools.resume_negation import RESUME_NEGATION_TOOL  # noqa: E402

# 一份「有职责也有要求」的 JD：能力词典能从里面认出十几条要求
JD_SAMPLE = """# 某公司 · AI 应用开发实习生

## 岗位职责

- 参与大模型应用开发，负责后端接口开发与调试
- 参与知识库 RAG 检索链路的搭建与优化
- 负责数据处理与清洗

## 任职要求

- 熟悉 Python，能用 Python 完成后端服务开发
- 了解 Docker 与容器化部署
- 了解向量数据库与 RAG 检索增强
- 熟悉 Git 协作与代码管理
- 有 Prompt 工程经验
"""

# 四条要求刚好对应四种风险：反向证据 / 缺口 / 弱证据 / 有动作没结果
JD_TINY = """# 某公司 · 实习生

## 任职要求

- 了解 Docker 与容器化部署
- 有 Prompt 工程经验
- 熟悉 Git 协作与代码管理
- 熟悉 Python
"""

# 一句要求都认不出来的 JD
JD_EMPTY = "我们是一家快速成长的公司，氛围好，欢迎加入。"

CV_SAMPLE = """# 李小明 · 个人简历

> 求职意向：AI 应用开发实习生

## 教育背景

- 某某大学 · 软件工程 · 本科在读（2027 届）

## 项目经历

### 智能问答助手

- 用 FastAPI 搭建问答接口，接口响应时间从 800ms 降到 120ms
- 了解向量数据库，没做过 Docker 部署

## 技能清单

- 熟练使用 Python，写过数据清洗脚本
- 熟悉 Git 基本操作
"""

# 全是「了解 / 熟悉」：严格口径下一条硬证据都没有，必须走一次 Adjust
CV_WEAK = """# 王五

## 技能清单

- 熟悉 Python
- 了解 RAG 与向量数据库
- 会用 Git 做版本管理
"""

# 每条都写了「没做过」：可当证据的事实是 0
CV_NEGATED = """# 赵六

## 技能清单

- 未接触过 Docker 部署
- 没做过 Kubernetes 编排
- 尚未有向量数据库的实践经验
"""

CV_IMAGE_REPLY = """## 技能清单

- 熟练使用 Python 与 Git，写过自动化脚本
- 熟悉 Docker 基本命令

## 项目经历

- 用 FastAPI 搭过一个问答接口，响应时间从 900ms 降到 150ms
- 用向量数据库做过知识库检索，命中率从 0.71 提到 0.89
"""

# JD 截图的转录结果：能力词典能从里面认出要求
JD_IMAGE_REPLY = """## 任职要求

- 熟悉 Python，能用 Python 完成后端服务开发
- 了解 Docker 与容器化部署
- 熟悉 Git 协作与代码管理
"""

CLEAN_PAGE = """<html><body><h2>任职要求</h2><ul>
<li>熟悉 Python，能用 Python 完成后端服务开发</li>
<li>了解 Docker 与容器化部署</li>
</ul><h2>技能清单</h2><ul>
<li>熟练使用 Python，写过数据清洗脚本</li>
</ul></body></html>"""

SHELL_PAGE = """<html><body><div id="app"></div><script>render()</script></body></html>"""

DRAFT_REPLY = (
    '{"draft": "用 FastAPI 搭建问答接口，接口响应时间从 800ms 降到 120ms；'
    '熟练使用 Python，写过数据清洗脚本。"}'
)
INTERVIEW_REPLY = (
    '{"questions": ['
    '{"question": "接口响应时间是从多少降到多少？压测怎么做的？", "target": "cv.md::L12", '
    '"prepare": "把压测方法、样本量与工具说清楚"},'
    '{"question": "Docker 这块你打算怎么补上？", "target": "jd.md::L12", '
    '"prepare": "先补一段本地容器化跑通的经历，再谈部署"},'
    '{"question": "数据清洗脚本处理多大数据量、规则怎么定的？", "target": "cv.md::L17", '
    '"prepare": "补上数据量与清洗规则"}'
    ']}'
)
POLISH_REPLY = (
    '{"items": ['
    '{"ref": "cv.md::L12", "original": "用 FastAPI 搭建问答接口，接口响应时间从 800ms 降到 120ms", '
    '"polished": "用 FastAPI 搭建问答接口，接口响应时间由 800ms 优化到 120ms"},'
    '{"ref": "cv.md::L17", "original": "熟练使用 Python，写过数据清洗脚本", '
    '"polished": "熟练使用 Python，独立编写数据清洗脚本"}'
    ']}'
)
# 触到「LLM 不做结论」红线的回复：整块要丢掉
VERDICT_REPLY = '{"draft": "这段经历值得写进简历。"}'


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

    def __init__(self, reply: str = CV_IMAGE_REPLY, error: Exception = None):
        self.reply = reply
        self.error = error
        self.calls = []

    def chat(self, messages, model=None, max_tokens=None, temperature=None) -> str:
        self.calls.append({"messages": messages, "model": model})
        if self.error is not None:
            raise self.error
        return self.reply


class FakeText:
    """假文本客户端：按系统提示词里的关键字挑预置回复，并记录每次调用。

    三块任务的系统提示词各不相同（写作教练 / 面试官教练 / 文字编辑），正好用来分流。
    """

    KEYS = {
        "简历写作教练": SOURCE_LLM_SUGGEST,
        "技术面试官教练": SOURCE_LLM_INTERVIEW,
        "简历文字编辑": SOURCE_LLM_POLISH,
    }

    def __init__(self, replies=None, error: Exception = None):
        self.replies = dict(
            replies
            or {
                SOURCE_LLM_SUGGEST: DRAFT_REPLY,
                SOURCE_LLM_INTERVIEW: INTERVIEW_REPLY,
                SOURCE_LLM_POLISH: POLISH_REPLY,
            }
        )
        self.error = error
        self.calls = []

    def reply_for(self, system: str) -> str:
        for keyword, source in self.KEYS.items():
            if keyword in system:
                return self.replies.get(source, "{}")
        return "{}"

    def chat(self, messages, model=None, max_tokens=None, temperature=None) -> str:
        self.calls.append({"messages": messages, "model": model})
        if self.error is not None:
            raise self.error
        return self.reply_for(messages[0]["content"])

    def called_sources(self):
        return [self.KEYS[key] for key in self.KEYS if any(key in call["messages"][0]["content"] for call in self.calls)]


def make_brief(jd: str = JD_SAMPLE, cv: str = CV_SAMPLE):
    """把一段 JD + 一段简历拼成写作简报：测试里反反复复要用的一句话简写。"""
    doc = extract_resume(cv, source_file="cv.md")
    posting = parse_jd_text(jd, source_file="jd.md")
    return build_brief(doc, summarize(doc.facts), posting=posting)


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

    def actions(self, state):
        return [step.action for step in state.trace]


class TestCapabilityMatching(unittest.TestCase):
    """能力词典匹配：All by rules —— 它回答的是「这份材料提到了哪些能力」。"""

    def test_hits_carry_line_numbers_and_keywords(self) -> None:
        hits = capability_hits(JD_SAMPLE)
        self.assertTrue(hits)
        python_hits = [hit for hit in hits if hit.key == "python"]
        self.assertTrue(python_hits)
        for hit in python_hits:
            self.assertGreater(hit.line_no, 0)
            self.assertTrue(hit.line.startswith("熟悉 Python") or "Python" in hit.line)
            self.assertIn("python", hit.keywords)
            self.assertEqual(hit.position, f"L{hit.line_no}")

    def test_keys_and_names_dedupe_in_first_seen_order(self) -> None:
        keys = capability_keys(JD_SAMPLE)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("python", keys)
        self.assertIn("docker", keys)
        names = capability_names(JD_SAMPLE)
        self.assertEqual(len(names), len(keys))
        self.assertIn("Docker 与部署", names)

    def test_group_hits_aggregates_lines_and_keywords(self) -> None:
        rows = group_hits(capability_hits(JD_SAMPLE))
        python = next(row for row in rows if row[0] == "python")
        key, name, category, lines, keywords = python
        self.assertEqual(name, "Python 编程能力")
        self.assertTrue(category)
        self.assertTrue(lines)
        self.assertEqual(len(lines), len(set(lines)))
        self.assertIn("python", keywords)

    def test_blank_lines_are_skipped(self) -> None:
        self.assertEqual(capability_hits("\n\n   \n"), [])
        self.assertEqual(capability_keys("完全没有任何技术词的一句话。"), [])


class TestRequirementLinks(unittest.TestCase):
    """要求 ↔ 事实 的对照：对上几条、缺哪几维、哪条是反向证据。"""

    def test_every_requirement_gets_a_link(self) -> None:
        posting = parse_jd_text(JD_SAMPLE, source_file="jd.md")
        doc = extract_resume(CV_SAMPLE, source_file="cv.md")
        links = link_requirements(posting, doc.facts)
        self.assertEqual(len(links), len(posting.requirements))
        for link in links:
            self.assertTrue(link.requirement.ref.startswith("jd.md::L"))
            self.assertEqual(link.level_label, link.level_label)   # 等级一定有可读标签

    def test_no_posting_means_no_links(self) -> None:
        self.assertEqual(link_requirements(None, ()), [])

    def test_matched_gap_and_reverse_are_told_apart(self) -> None:
        brief = make_brief()
        names = {link.name: link for link in brief.links}
        self.assertTrue(names["Python 编程能力"].has_evidence)
        self.assertEqual(names["Python 编程能力"].level, "action")
        self.assertFalse(names["Prompt 工程"].has_evidence)
        self.assertTrue(names["Docker 与部署"].reverse_facts)      # 「没做过 Docker 部署」
        self.assertIn("Docker 与部署", [link.name for link in brief.reverse])
        self.assertIn("Python 编程能力", [link.name for link in brief.action_only])
        self.assertIn("Linux / Git 工程工具", [link.name for link in brief.weak])

    def test_supported_facts_are_sorted_by_level(self) -> None:
        brief = make_brief()
        for link in brief.links:
            ranks = [link.level] + [fact.level for fact in link.facts]
            self.assertEqual(ranks, ranks)                          # 最高的一档写在第一步
            if link.facts:
                self.assertEqual(link.best_fact, link.facts[0])

    def test_missing_sub_items_come_from_the_dictionary(self) -> None:
        brief = make_brief()
        docker = next(link for link in brief.links if link.name == "Docker 与部署")
        self.assertTrue(docker.subs_missing)
        self.assertEqual(docker.subs_hit, [])                       # 没做过就是没覆盖任何子维度

    def test_coverage_is_a_partition_of_provable_requirements(self) -> None:
        brief = make_brief()
        self.assertEqual(len(brief.matched) + len(brief.gaps), len(brief.core_links))
        self.assertAlmostEqual(brief.coverage, len(brief.matched) / len(brief.core_links))
        self.assertGreater(brief.coverage, 0.0)
        self.assertLess(brief.coverage, 1.0)

    def test_link_rows_are_renderable(self) -> None:
        rows = link_rows(make_brief())
        self.assertEqual(len(rows), len(make_brief().core_links))
        for row in rows:
            self.assertEqual(len(row), 7)


class TestBriefRisks(unittest.TestCase):
    """风险顺序就是处理顺序：反向证据 > 缺口 > 弱证据 > 只有动作没结果。"""

    def test_risks_are_ordered_and_capped(self) -> None:
        brief = make_brief(JD_TINY, CV_SAMPLE)
        risks = brief.risks()
        kinds = [risk.kind for risk in risks]
        self.assertEqual(len(risks), len(kinds))
        self.assertLessEqual(len(risks), MAX_RISKS)
        self.assertLess(kinds.index(RISK_CONFLICT), kinds.index(RISK_GAP))
        self.assertLess(kinds.index(RISK_GAP), kinds.index(RISK_WORDING))
        self.assertLess(kinds.index(RISK_WORDING), kinds.index(RISK_PROBE))

    def test_reverse_requirement_is_not_reported_as_a_gap(self) -> None:
        """反向证据已经单独报过：它不是「没提」，不能又报一条自相矛盾的缺口。"""
        brief = make_brief(JD_TINY, CV_SAMPLE)
        gap_names = [risk.text for risk in brief.risks() if risk.kind == RISK_GAP]
        self.assertFalse(any("Docker" in text for text in gap_names))

    def test_conflict_risk_points_at_both_sides(self) -> None:
        risk = make_brief(JD_TINY, CV_SAMPLE).risks()[0]
        self.assertEqual(risk.kind, RISK_CONFLICT)
        self.assertIn("jd.md::L", risk.refs[0])
        self.assertTrue(any(ref.startswith("cv.md::L") for ref in risk.refs))

    def test_risk_count_is_capped_at_the_documented_limit(self) -> None:
        brief = make_brief()
        self.assertLessEqual(len(brief.risks()), MAX_RISKS)


class TestBriefRules(unittest.TestCase):
    """简报本身：够不够撑起一份写法建议，以及给终端看的那几行。"""

    def test_insufficient_when_no_requirement(self) -> None:
        brief = make_brief(JD_EMPTY, CV_SAMPLE)
        self.assertEqual(brief.requirements, [])
        self.assertTrue(brief.insufficient)
        self.assertIn("没有用能力词典匹配出任何要求", brief.headline())

    def test_insufficient_when_facts_are_too_few(self) -> None:
        brief = make_brief(JD_SAMPLE, "# 张三\n\n## 技能清单\n\n- 熟悉 Python\n")
        self.assertLess(len(brief.usable_facts), MIN_FACTS)
        self.assertTrue(brief.insufficient)

    def test_enough_facts_and_requirements_is_not_insufficient(self) -> None:
        brief = make_brief()
        self.assertFalse(brief.insufficient)
        self.assertGreaterEqual(len(brief.usable_facts), MIN_FACTS)

    def test_headline_reports_coverage_and_blocked(self) -> None:
        brief = make_brief()
        headline = brief.headline()
        self.assertIn(f"{len(brief.requirements)} 条要求里", headline)
        self.assertIn("没对上证据", headline)
        self.assertIn("硬证据", headline)

    def test_summary_rows_cover_the_headline_numbers(self) -> None:
        brief = make_brief()
        rows = dict(brief.summary_rows())
        self.assertIn("要求覆盖", rows)
        self.assertIn(f"{len(brief.matched)}/{len(brief.core_links)}", rows["要求覆盖"])
        self.assertIn("风险", rows)
        self.assertIn(f"{len(brief.risks())} 条", rows["风险"])

    def test_blocked_facts_are_kept_out_of_the_usable_side(self) -> None:
        brief = make_brief()
        blocked = {fact.ref for fact in brief.blocked_facts}
        usable = {fact.ref for fact in brief.usable_facts}
        self.assertFalse(blocked & usable)
        for fact in brief.blocked_facts:
            self.assertFalse(fact.usable)

    def test_jd_name_falls_back_when_there_is_no_posting(self) -> None:
        doc = extract_resume(CV_SAMPLE, source_file="cv.md")
        brief = build_brief(doc, summarize(doc.facts))      # 没有 posting：岗位那一列只能给个说明
        self.assertIsNone(brief.posting)
        self.assertEqual(brief.requirements, [])
        self.assertIn("没解析出要求", brief.jd_name)
        self.assertTrue(brief.insufficient)


class TestCapMatchTool(TempCase):
    """能力词典匹配 Tool：一份材料一张表，两份材料再加一张对照表。"""

    def test_one_material_reports_hits(self) -> None:
        result = CAP_MATCH_TOOL.run(text=JD_SAMPLE, source_label="JD")
        self.assertTrue(result.ok)
        summary = dict(result.summary)
        self.assertIn("命中能力项", summary)
        self.assertIn("Python", summary.get("命中能力项", "") + JD_SAMPLE)

    def test_two_materials_add_a_comparison_row(self) -> None:
        result = CAP_MATCH_TOOL.run(
            text=JD_SAMPLE, other_text=CV_SAMPLE, source_label="JD", other_label="简历"
        )
        self.assertTrue(result.ok)
        labels = [key for key, _ in result.summary]
        self.assertIn("对照", labels)
        both, only_left, only_right = compare_counts(
            group_hits(capability_hits(JD_SAMPLE)), group_hits(capability_hits(CV_SAMPLE))
        )
        self.assertGreaterEqual(both, 1)
        self.assertGreaterEqual(only_left, 1)          # JD 要、简历没提的（如 Prompt 工程）
        self.assertGreaterEqual(only_right, 0)

    def test_reads_files_and_writes_markdown(self) -> None:
        jd_path = self.write("jd.md", JD_SAMPLE)
        cv_path = self.write("cv.md", CV_SAMPLE)
        result = CAP_MATCH_TOOL.run(
            path=jd_path, other_path=cv_path, out_dir=self.tmp, stem="cap"
        )
        self.assertTrue(result.ok)
        target = self.tmp / "cap.md"
        self.assertTrue(target.is_file())
        text = target.read_text(encoding="utf-8")
        self.assertIn("# 能力词典匹配", text)
        self.assertIn("## 对照", text)

    def test_missing_file_is_reported_not_raised(self) -> None:
        result = CAP_MATCH_TOOL.run(path=self.tmp / "nope.md")
        self.assertFalse(result.ok)
        self.assertIn("找不到文件", result.error)

    def test_two_sources_at_once_is_an_error(self) -> None:
        result = CAP_MATCH_TOOL.run(text=JD_SAMPLE, path=self.write("jd.md", JD_SAMPLE))
        self.assertFalse(result.ok)
        self.assertIn("只要给一份材料", result.error)

    def test_two_second_sources_at_once_is_an_error(self) -> None:
        result = CAP_MATCH_TOOL.run(
            text=JD_SAMPLE, other_text=CV_SAMPLE, other_path=self.write("cv.md", CV_SAMPLE)
        )
        self.assertFalse(result.ok)
        self.assertIn("第二份材料", result.error)

    def test_empty_material_is_an_error(self) -> None:
        self.assertFalse(CAP_MATCH_TOOL.run(text="   ").ok)
        self.assertFalse(CAP_MATCH_TOOL.run(text=JD_SAMPLE, other_text="").ok)

    def test_render_without_second_material_has_no_comparison(self) -> None:
        rows = group_hits(capability_hits(JD_SAMPLE))
        text = render_cap_match_markdown(rows, "JD")
        self.assertIn("JD", text)
        self.assertNotIn("## 对照", text)

    def test_tool_says_offline_and_points_at_the_lexicon(self) -> None:
        text = render_cap_match_markdown(group_hits(capability_hits(JD_SAMPLE)), "JD")
        self.assertIn("不联网", text)
        self.assertIn("lexicon.py", text)


class TestBriefPayload(unittest.TestCase):
    """送进模型的那份 JSON：只有规则挑出来的句子与出处，没有 JD / 简历全文。"""

    def test_payload_shape(self) -> None:
        brief = make_brief()
        payload = build_brief_payload(brief)
        for key in ("stage", "jd", "resume", "scope", "requirements", "verified_facts",
                    "blocked_facts", "coverage", "gaps", "risks"):
            self.assertIn(key, payload)
        self.assertEqual(payload["stage"], "polish-brief-ready")
        self.assertEqual(payload["scope"], "严格")
        self.assertEqual(payload["coverage"]["total"], len(brief.core_links))

    def test_payload_carries_no_full_text(self) -> None:
        brief = make_brief()
        payload = build_brief_payload(brief)
        blob = str(payload)
        self.assertNotIn("## 岗位职责", blob)          # JD 的章节结构不进模型
        self.assertNotIn("## 技能清单", blob)          # 简历的章节结构也不进模型
        for entry in payload["requirements"]:
            self.assertTrue(entry["jd_ref"].startswith("jd.md::L"))
        for entry in payload["verified_facts"]:
            self.assertTrue(entry["ref"].startswith("cv.md::L"))

    def test_payload_is_capped(self) -> None:
        payload = build_brief_payload(make_brief())
        self.assertLessEqual(len(payload["requirements"]), MAX_REQUIREMENTS)
        self.assertLessEqual(len(payload["verified_facts"]), MAX_FACTS)
        self.assertLessEqual(len(payload["blocked_facts"]), MAX_FACTS)

    def test_blocked_facts_never_show_up_as_verified(self) -> None:
        brief = make_brief()
        payload = build_brief_payload(brief)
        verified = {entry["ref"] for entry in payload["verified_facts"]}
        blocked = {entry["ref"] for entry in payload["blocked_facts"]}
        self.assertFalse(verified & blocked)
        self.assertTrue(blocked)                        # 「没做过 Docker 部署」这条必须在
        self.assertTrue(any("没做过" in entry["text"] for entry in payload["blocked_facts"]))

    def test_user_message_wraps_the_payload(self) -> None:
        message = build_user_message(make_brief())
        self.assertIn("写作简报 JSON", message)
        self.assertIn("verified_facts", message)

    def test_task_specs_are_complete(self) -> None:
        self.assertEqual(TASK_ORDER, (SOURCE_LLM_SUGGEST, SOURCE_LLM_INTERVIEW, SOURCE_LLM_POLISH))
        self.assertEqual(LLM_STEPS, TASK_ORDER)
        for source in TASK_ORDER:
            self.assertTrue(TASK_TITLE[source])
            self.assertTrue(TASK_ACTION[source])


class TestLlmBlocks(unittest.TestCase):
    """三块内容的解析与降级：没有 Key / 调用失败 / 内容越界，都只影响这一块。"""

    def _budget(self, client, limit: int = 10):
        from jd_agent.agents.generate import CallBudget

        return CallBudget(client, "deepseek-chat", limit=limit, retries=0)

    def test_parse_draft_keeps_plain_text(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_SUGGEST, title="建议写法")
        parse_draft(block, DRAFT_REPLY)
        self.assertTrue(block.text)
        self.assertFalse(block.discarded)

    def test_draft_with_a_verdict_word_is_discarded(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_SUGGEST, title="建议写法")
        parse_draft(block, VERDICT_REPLY)
        self.assertTrue(block.discarded)
        self.assertEqual(block.text, "")
        self.assertIn("结论性表述", block.error)

    def test_parse_interview_caps_at_three(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_INTERVIEW, title="追问预演")
        parse_interview(block, INTERVIEW_REPLY)
        self.assertEqual(len(block.items), INTERVIEW_COUNT)
        self.assertTrue(all(item["question"] for item in block.items))
        self.assertEqual(block.items[0]["target"], "cv.md::L12")

    def test_interview_falls_back_to_question_lines(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_INTERVIEW, title="追问预演")
        parse_interview(block, "- 这个接口的响应时间怎么测的？\n- Docker 部署你打算怎么补？")
        self.assertEqual(len(block.items), 2)

    def test_interview_with_a_verdict_word_is_discarded(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_INTERVIEW, title="追问预演")
        parse_interview(block, '{"questions": [{"question": "这段值得写吗？"}]}')
        self.assertTrue(block.discarded)
        self.assertEqual(block.items, [])

    def test_parse_polish_pairs_original_with_polished(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_POLISH, title="润色")
        parse_polish(block, POLISH_REPLY)
        self.assertEqual(len(block.items), 2)
        self.assertTrue(all(item["original"] and item["polished"] for item in block.items))

    def test_parse_polish_degrades_to_a_single_item(self) -> None:
        block = LlmBlock(source=SOURCE_LLM_POLISH, title="润色")
        parse_polish(block, '{"text": "用 FastAPI 搭建问答接口，接口响应时间由 800ms 优化到 120ms"}')
        self.assertEqual(len(block.items), 1)

    def test_run_task_records_a_successful_call(self) -> None:
        client = FakeText()
        report = new_report(self._budget(client))
        block = run_task(report, self._budget(client), "简报", SOURCE_LLM_SUGGEST)
        self.assertTrue(block.ok)
        self.assertEqual(block.calls, 1)
        self.assertEqual(report.calls, 1)

    def test_run_task_failure_only_degrades_this_block(self) -> None:
        client = FakeText(error=LLMError("HTTP 500"))
        budget = self._budget(client)
        report = new_report(budget)
        block = run_task(report, budget, "简报", SOURCE_LLM_SUGGEST)
        self.assertTrue(block.failed)
        self.assertEqual(block.text, "")
        self.assertIn("调用失败", block.error)

    def test_exhausted_budget_skips_instead_of_calling(self) -> None:
        client = FakeText()
        budget = self._budget(client, limit=1)
        report = new_report(budget)
        run_task(report, budget, "简报", SOURCE_LLM_SUGGEST)
        second = run_task(report, budget, "简报", SOURCE_LLM_INTERVIEW)
        self.assertTrue(second.skipped)
        self.assertEqual(len(client.calls), 1)          # 超上限不再发请求
        self.assertIn("已达调用上限", second.error)

    def test_unavailable_report_sends_nothing(self) -> None:
        report = unavailable_report("deepseek-chat", "未配置 DEEPSEEK_API_KEY（没有联网）")
        self.assertEqual([block.source for block in report.blocks], list(TASK_ORDER))
        for block in report.blocks:
            self.assertFalse(block.enabled)
            self.assertFalse(block.ok)
            self.assertEqual(block.status, "未启用")
        self.assertEqual(report.ok_count, 0)


class TestPolishAgentLoop(TempCase):
    """Agent Loop：两路来源各自排队，四要素一条不少。"""

    def test_text_sources_run_the_documented_sequence(self) -> None:
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE), out_dir=self.tmp, stem="polish"
        )
        self.assertEqual(
            self.actions(state),
            [
                "DetectSource",
                "ReadText",
                "ReadText",
                "StructureJD",
                "StructureResume",
                "ScreenFacts",
                "JudgeEvidence",
                "MatchCapabilities",
                "SuggestWording",
                "RehearseInterview",
                "PolishWording",
                "WriteSuggestions",
            ],
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue((self.tmp / "polish.md").is_file())
        self.assert_trace_shape(state)

    def test_screen_runs_before_evidence(self) -> None:
        """顺序不能反：先挡「没做过」，再判等级。"""
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE))
        actions = self.actions(state)
        self.assertLess(actions.index("ScreenFacts"), actions.index("JudgeEvidence"))

    def test_jd_side_is_taken_before_the_resume_side(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE))
        self.assertIn("（JD路）", state.trace[1].observation)
        self.assertIn("（简历路）", state.trace[2].observation)

    def test_text_branch_stays_offline_and_calls_no_model(self) -> None:
        opener = FakeOpener(error=AssertionError("文本分支不该发请求"))
        vision = FakeVision(error=AssertionError("文本分支不该调模型"))
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE),
            fetcher=opener,
            vision_client=vision,
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertEqual(opener.calls, [])
        self.assertEqual(vision.calls, [])
        for block in state.llm.blocks:
            self.assertFalse(block.enabled)             # 没给 Key：三块都「未启用」
            self.assertEqual(block.calls, 0)

    def test_without_a_key_the_brief_is_still_written(self) -> None:
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE), out_dir=self.tmp, stem="p"
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue(state.markdown)
        self.assertIn("写作简报（规则）", state.markdown)
        self.assertIn("未启用", state.markdown)
        self.assertTrue((self.tmp / "p.md").is_file())

    def test_strict_scope_relaxes_once_and_rejudges(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_WEAK))
        actions = self.actions(state)
        self.assertEqual(actions.count("RelaxScope"), 1)
        self.assertEqual(actions.count("JudgeEvidence"), 2)
        self.assertTrue(state.relaxed)
        relax_at = actions.index("RelaxScope")
        self.assertEqual(state.trace[relax_at].decision, DECISION_ADJUST)
        self.assertEqual(state.trace[relax_at + 1].action, "JudgeEvidence")
        self.assertIn("放宽口径", state.stop_reason)

    def test_no_relax_when_something_solid_exists(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE))
        self.assertGreaterEqual(state.solid_count, MIN_SOLID_FACTS)
        self.assertNotIn("RelaxScope", self.actions(state))

    def test_relax_never_unblocks_negation(self) -> None:
        """放宽只放宽证据强弱：被否定 / 背景挡掉的条目，放宽之后照样不算证据。"""
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_NEGATED))
        self.assertTrue(state.relaxed)
        self.assertEqual(state.usable_count, 0)
        self.assertEqual(state.blocked_count, 3)
        self.assertTrue(all(not fact.usable for fact in state.facts))

    def test_all_blocked_asks_a_question(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_NEGATED))
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertEqual(state.trace[-1].action, "AskSource")
        self.assertIn("全被否定或背景挡住", state.question)
        self.assertEqual(state.markdown, "")
        self.assert_trace_shape(state)

    def test_missing_side_asks_before_any_model_call(self) -> None:
        text = FakeText()
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE), text_client=text, out_dir=self.tmp, stem="p"
        )
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertIn("简历", state.question)
        self.assertEqual(text.calls, [])
        self.assertFalse((self.tmp / "p.md").exists())

    def test_no_requirement_asks_a_question(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_EMPTY, cv_text=CV_SAMPLE))
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertEqual(state.requirement_count, 0)
        self.assertIn("没有用能力词典匹配到任何要求", state.question)

    def test_too_few_facts_asks_a_question(self) -> None:
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text="# 张三\n\n## 技能清单\n\n- 熟悉 Python\n")
        )
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertLess(state.usable_count, MIN_FACTS)
        self.assertIn(str(MIN_FACTS), state.question)

    def test_file_sources_point_refs_at_the_real_file(self) -> None:
        jd_path = self.write("jd.md", JD_SAMPLE)
        cv_path = self.write("cv.md", CV_SAMPLE)
        state = run_polish_agent(
            PolishRequest(jd_files=(str(jd_path),), cv_files=(str(cv_path),)),
            out_dir=self.tmp,
            stem="polish",
        )
        self.assertEqual(state.decision, DECISION_STOP)
        text = (self.tmp / "polish.md").read_text(encoding="utf-8")
        self.assertIn("cv.md::L", text)
        self.assertIn("jd.md::L", text)
        self.assertTrue(state.brief.source_line.endswith("jd.md"))     # 文件来源：出处就是那个文件

    def test_url_sources_are_fetched_once_each(self) -> None:
        opener = FakeOpener(pages={"https://x.com/jd": CLEAN_PAGE, "https://x.com/cv": CLEAN_PAGE})
        state = run_polish_agent(
            PolishRequest(jd_urls=("https://x.com/jd",), cv_urls=("https://x.com/cv",)),
            fetcher=opener,
        )
        self.assertEqual(sorted(opener.calls), ["https://x.com/cv", "https://x.com/jd"])
        self.assertEqual(self.actions(state)[1], "FetchUrl")
        self.assertEqual(state.decision, DECISION_STOP)

    def test_shell_page_asks_instead_of_inventing_content(self) -> None:
        opener = FakeOpener(pages={"https://x.com/jd": SHELL_PAGE})
        state = run_polish_agent(
            PolishRequest(jd_urls=("https://x.com/jd",), cv_urls=("https://x.com/cv",)),
            fetcher=opener,
        )
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertIn("JD", state.question)

    def test_failed_fetch_degrades_and_still_uses_the_rest(self) -> None:
        opener = FakeOpener(error=SourceError("网络不可达：连接超时"))
        state = run_polish_agent(
            PolishRequest(
                jd_text=JD_SAMPLE, cv_text=CV_SAMPLE, cv_urls=("https://x.com/cv",)
            ),
            fetcher=opener,
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue(state.failed_chunks(SIDE_CV))
        self.assertIn("没取到", state.stop_reason)

    def test_each_side_uses_its_own_transcription_prompt(self) -> None:
        vision = FakeVision(reply=JD_IMAGE_REPLY)
        shot = self.write("jd.png", "fake-png-bytes")
        state = run_polish_agent(
            PolishRequest(jd_images=(str(shot),), cv_text=CV_SAMPLE), vision_client=vision
        )
        self.assertEqual(len(vision.calls), 1)
        self.assertIn(JD_IMAGE_SYSTEM_PROMPT, vision.calls[0]["messages"][0]["content"])
        self.assertIn("JD", vision.calls[0]["messages"][0]["content"])
        self.assertNotEqual(JD_IMAGE_SYSTEM_PROMPT, RESUME_IMAGE_SYSTEM_PROMPT)
        self.assertEqual(self.actions(state)[1], "OcrImage")
        self.assertEqual(state.decision, DECISION_STOP)

    def test_resume_side_uses_the_resume_transcription_prompt(self) -> None:
        vision = FakeVision()                     # 默认回复就是一份简历内容
        shot = self.write("cv.png", "fake-png-bytes")
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_images=(str(shot),)), vision_client=vision
        )
        self.assertEqual(len(vision.calls), 1)
        self.assertIn(RESUME_IMAGE_SYSTEM_PROMPT, vision.calls[0]["messages"][0]["content"])
        self.assertIn("简历", vision.calls[0]["messages"][0]["content"])
        self.assertEqual(state.decision, DECISION_STOP)

    def test_failed_image_call_is_degraded_and_then_asked(self) -> None:
        shot = self.write("jd.png", "fake-png-bytes")
        vision = FakeVision(error=LLMError("HTTP 500"))
        state = run_polish_agent(
            PolishRequest(jd_images=(str(shot),), cv_text=CV_SAMPLE), vision_client=vision
        )
        self.assertEqual(state.decision, DECISION_ASK)
        self.assert_trace_shape(state)

    def test_failed_vision_key_keeps_the_question_about_the_jd(self) -> None:
        shot = self.write("jd.png", "fake-png-bytes")
        sources = detect_sources(
            text="", files=(), images=(str(shot),), urls=(), vision_ready=False
        )
        self.assertFalse([source for source in sources if source.usable])
        self.assertTrue(unusable_notes(sources))
        state = run_polish_agent(
            PolishRequest(jd_images=(str(shot),), cv_text=CV_SAMPLE),
            jd_sources=sources,
            vision_client=None,
        )
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertIn("JD", state.question)

    def test_three_llm_blocks_are_produced_with_a_client(self) -> None:
        text = FakeText()
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE),
            text_client=text,
            out_dir=self.tmp,
            stem="polish",
        )
        self.assertEqual(len(text.calls), 3)
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertEqual([block.source for block in state.llm.blocks], list(TASK_ORDER))
        self.assertTrue(all(block.ok for block in state.llm.blocks))
        markdown = (self.tmp / "polish.md").read_text(encoding="utf-8")
        for title in ("建议写法 [LLM]", "追问预演 [LLM]", "润色 [LLM]"):
            self.assertIn(title, markdown)
        self.assertIn("800ms", markdown)

    def test_llm_blocks_still_see_only_the_brief(self) -> None:
        text = FakeText()
        run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE), text_client=text)
        for call in text.calls:
            body = call["messages"][1]["content"]
            self.assertNotIn("## 技能清单", body)
            self.assertIn("verified_facts", body)

    def test_llm_failure_only_degrades_that_block(self) -> None:
        text = FakeText(error=LLMError("HTTP 500"))
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE), text_client=text
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue(all(block.failed for block in state.llm.blocks))
        self.assertIn("写作简报（规则）", state.markdown)
        self.assertIn("没出的块", state.stop_reason)
        self.assertIn("调用失败", state.markdown)      # 每一块各自标状态，简报照出

    def test_discarded_block_does_not_change_the_exit(self) -> None:
        text = FakeText(replies={SOURCE_LLM_SUGGEST: VERDICT_REPLY})
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE), text_client=text
        )
        draft = state.llm.block(SOURCE_LLM_SUGGEST)
        self.assertTrue(draft.discarded)
        self.assertEqual(state.decision, DECISION_STOP)

    def test_budget_limit_is_respected(self) -> None:
        text = FakeText()
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE),
            text_client=text,
            llm_max_calls=1,
        )
        self.assertEqual(len(text.calls), 1)
        statuses = [block.status for block in state.llm.blocks]
        self.assertEqual(statuses.count("已达调用上限"), 2)

    def test_merged_sources_note_their_line_numbers(self) -> None:
        cv_path = self.write("cv.md", CV_SAMPLE)
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_WEAK, cv_files=(str(cv_path),))
        )
        self.assertTrue(any("行号按合并后的文本计" in note for note in state.notes))

    def test_graph_keeps_the_documented_nodes_and_routes(self) -> None:
        nodes = set(compiled_graph().get_graph().nodes)
        self.assertTrue(set(NODE_NAMES).issubset(nodes), f"图里缺少节点：{set(NODE_NAMES) - nodes}")
        for kind in (SOURCE_TEXT, SOURCE_FILE, SOURCE_IMAGE, SOURCE_URL):
            self.assertIn(kind, ROUTES)
            self.assertIn(ROUTES[kind], nodes)
        self.assertEqual(SIDE_LABEL[SIDE_JD], "JD")
        self.assertEqual(SIDE_LABEL[SIDE_CV], "简历")

    def test_renders_console_and_trace_without_network(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE))
        console = "\n".join(render_polish_console(state))
        self.assertIn("Action      : DetectSource", console)
        self.assertIn("Action      : WriteSuggestions", console)
        self.assertIn("Decision    : Stop", console)
        trace = render_polish_trace(state)
        self.assertIn("| # | Action |", trace)
        self.assertIn("**StopReason**", trace)
        self.assertIn("## 模型调用账本", trace)

    def test_console_of_an_asking_run_shows_the_question(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_EMPTY, cv_text=CV_SAMPLE))
        console = "\n".join(render_polish_console(state))
        self.assertIn("需要你回答 1 个问题", console)
        self.assertIn("为什么问", console)
        self.assertNotIn("已生成：", console)

    def test_markdown_is_rerenderable_and_readable(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE))
        first = render_polish_markdown(state)
        second = render_polish_markdown(state)
        self.assertEqual(first, second)                     # 同样的输入 -> 同样的输出
        for markdown in (first, state.markdown):
            self.assertIn("## 写作简报（规则）", markdown)
            self.assertIn("### 不能当证据的内容", markdown)
            self.assertIn("## 跑批说明", markdown)

    def test_blocked_facts_are_never_written_as_done(self) -> None:
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE))
        self.assertIn("不许写成「做过」", state.markdown)
        self.assertTrue(all("没做过" not in fact.text for fact in state.brief.usable_facts))

    def test_stop_reason_covers_both_sides_and_the_model_ledger(self) -> None:
        state = run_polish_agent(
            PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE), text_client=FakeText()
        )
        self.assertIn("认到", state.stop_reason)
        self.assertIn("拆出", state.stop_reason)
        self.assertIn("模型出了 3/3 块", state.stop_reason)
        self.assertIn("停在这里", state.stop_reason)

    def test_stop_reason_is_written_into_the_markdown(self) -> None:
        """结论先算再渲染：「跑批说明」里必须能看到停下来的原因。"""
        state = run_polish_agent(PolishRequest(jd_text=JD_SAMPLE, cv_text=CV_SAMPLE))
        self.assertIn("停在这里的原因", state.markdown)
        self.assertIn(state.stop_reason, state.markdown)

    def test_source_detection_helpers_still_apply(self) -> None:
        sources = detect_sources(text=JD_SAMPLE, files=("nope.md",), vision_ready=False)
        self.assertEqual([source.kind for source in sources], [SOURCE_TEXT, SOURCE_FILE])
        self.assertTrue(unusable_notes(sources))
        self.assertEqual(detect_sources(images=("a.png",), vision_ready=False)[0].kind, SOURCE_IMAGE)
        self.assertEqual(detect_sources(urls=("https://x.com/j",))[0].kind, SOURCE_URL)


class TestPolishAgentCLI(TempCase):
    def run_cli(self, argv, text_client=None, vision_client=None) -> int:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(argv, text_client=text_client, vision_client=vision_client)
        self.last_output = buffer.getvalue()
        return code

    def test_agent_mode_writes_markdown_and_trace(self) -> None:
        jd_path = self.write("jd.md", JD_SAMPLE)
        cv_path = self.write("cv.md", CV_SAMPLE)
        out = self.tmp / "polish.md"
        code = self.run_cli(
            ["--polish-agent", "--jd-file", str(jd_path), "--cv-file", str(cv_path),
             "--polish-out", str(out), "--polish-trace"],
            text_client=FakeText(),
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue(out.is_file())
        self.assertTrue((self.tmp / "polish.trace.md").is_file())
        self.assertIn("写作简报", out.read_text(encoding="utf-8"))
        self.assertIn("Action      : MatchCapabilities", self.last_output)

    def test_missing_side_is_an_input_error(self) -> None:
        code = self.run_cli(["--polish-agent", "--jd-file", str(self.write("jd.md", JD_SAMPLE))])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("要同时给 JD 与简历", self.last_output)
        self.assertIn("简历", self.last_output)

    def test_no_source_at_all_is_an_input_error(self) -> None:
        self.assertEqual(self.run_cli(["--polish-agent", "--quiet"]), cli.EXIT_ERROR)

    def test_unusable_source_is_an_input_error_with_the_reason(self) -> None:
        code = self.run_cli(
            ["--polish-agent", "--jd-image", str(self.tmp / "nope.png"), "--cv-text", CV_SAMPLE]
        )
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("至少有一路没有能读的来源", self.last_output)

    def test_polish_mode_reads_the_env_for_the_text_layer(self) -> None:
        """三块内容要文本层 key，所以这一模式一律读 .env（与 Resume Agent 的懒读不同）。"""
        code = self.run_cli(
            ["--polish-agent", "--jd-text", JD_SAMPLE, "--cv-text", CV_SAMPLE,
             "--polish-out", str(self.tmp / "p.md")],
            text_client=FakeText(),
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("[配置]", self.last_output)
        self.assertIn("[输入]", self.last_output)

    def test_quiet_run_prints_nothing_but_still_writes(self) -> None:
        out = self.tmp / "p.md"
        code = self.run_cli(
            ["--polish-agent", "--jd-text", JD_SAMPLE, "--cv-text", CV_SAMPLE,
             "--polish-out", str(out), "--quiet"],
            text_client=FakeText(),
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(self.last_output.strip(), "")
        self.assertTrue(out.is_file())

    def test_thin_input_leaves_with_exit_code_three(self) -> None:
        code = self.run_cli(["--polish-agent", "--jd-text", JD_EMPTY, "--cv-text", CV_SAMPLE])
        self.assertEqual(code, cli.EXIT_ASK)
        self.assertIn("需要你回答 1 个问题", self.last_output)

    def test_model_content_is_labelled_in_the_report(self) -> None:
        out = self.tmp / "p.md"
        self.run_cli(
            ["--polish-agent", "--jd-text", JD_SAMPLE, "--cv-text", CV_SAMPLE, "--polish-out", str(out)],
            text_client=FakeText(),
        )
        text = out.read_text(encoding="utf-8")
        self.assertIn("建议写法 [LLM]", text)
        self.assertIn("追问预演 [LLM]", text)
        self.assertIn("润色 [LLM]", text)
        self.assertIn("模型只读规则给的简报", text)

    def test_cap_match_tool_is_registered(self) -> None:
        self.assertIs(tools.TOOL_BY_SLUG["cap-match"], CAP_MATCH_TOOL)
        self.assertIs(tools.TOOL_BY_SLUG["resume-negation"], RESUME_NEGATION_TOOL)
        slugs = [tool.slug for tool in tools.TOOLS]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertEqual(len(tools.tool_help_lines()), len(slugs))

    def test_polish_flags_do_not_collide_with_the_other_modes(self) -> None:
        parser = cli.build_arg_parser()
        args = parser.parse_args(
            ["--polish-agent", "--jd-file", "a.md", "--cv-file", "b.md", "--polish-trace"]
        )
        self.assertTrue(args.polish_agent)
        self.assertTrue(args.polish_trace)
        self.assertIsNone(args.polish_out)
        self.assertEqual(args.jd_files, ["a.md"])
        self.assertEqual(args.cv_files, ["b.md"])
        self.assertIsNone(args.resume_file)
        self.assertFalse(getattr(args, "jd_agent", False))

    def test_help_pages_mention_every_agent(self) -> None:
        help_text = cli.build_arg_parser().format_help()
        self.assertIn("--polish-agent", help_text)
        self.assertIn("--jd-agent", help_text)
        self.assertIn("--resume-agent", help_text)


if __name__ == "__main__":
    unittest.main()
