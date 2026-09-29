"""JD Agent（LangGraph）测试：三种输入类型 + 工具区 + Agent Loop 四要素。

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
from jd_agent.domain.jd import parse_jd_text  # noqa: E402
from jd_agent.agents.jd_graph import (  # noqa: E402
    MIN_JD_BULLETS,
    MIN_URL_LINES,
    NODE_NAMES,
    JDRequest,
    compiled_graph,
    render_jd_console,
    render_jd_trace,
    run_jd_agent,
)
from jd_agent.domain.jd_html import clean_lines, count_lines, html_to_markdown  # noqa: E402
from jd_agent.services.jd_source import (  # noqa: E402
    SOURCE_FILE,
    SOURCE_IMAGE,
    SOURCE_TEXT,
    SOURCE_URL,
    SourceError,
    build_image_messages,
    detect_sources,
    looks_like_url,
    normalize_url,
    unusable_notes,
)
from jd_agent.domain.lexicon import CAPABILITY_BY_KEY  # noqa: E402
from jd_agent.core.llm import LLMError  # noqa: E402
from jd_agent.core.schema import (  # noqa: E402
    DECISION_ADJUST,
    DECISION_ASK,
    DECISION_CONTINUE,
    DECISION_STOP,
    DECISIONS,
)
from jd_agent.tools.jd_files import JD_FILE_TOOL, parse_file_markdown  # noqa: E402
from jd_agent.tools.jd_struct import (  # noqa: E402
    JD_STRUCT_TOOL,
    render_jd_markdown,
    structure_jd_text,
)

SAMPLE_JD = """AI 大模型应用开发实习生

任职要求
- 熟练使用 Python，了解 FastAPI 等 Web 框架。
- 熟悉大模型基础原理，了解 RAG 与向量数据库。
- 本科及以上在读，每周能来 4 天。

加分项
- 熟悉 Docker 部署流程
"""

MULTI_JD = """某公司 2026 校招

## 岗位一：AI 算法实习生

**公司**：某某科技
**工作地点**：北京

**岗位职责**
- 参与模型训练与评测。

**任职要求**
- 熟悉 PyTorch 与深度学习基础。

## 岗位二：后端开发实习生

**任职要求**
- 熟悉 Java 与 MySQL。
"""

# 噪声多的页面：严格口径剩不到 MIN_URL_LINES 行，必须换放宽口径
NOISY_PAGE = """<html><head><title>职位详情</title><style>body{margin:0}</style></head><body>
<div class="nav">首页 | 登录</div>
<div>立即申请</div>
<div>分享</div>
<div>收藏</div>
<div>相似职位</div>
<div>返回顶部</div>
<div>版权所有</div>
<h2>任职要求</h2>
<ul><li>熟悉 Python</li><li>了解 RAG</li><li>熟悉 Docker</li></ul>
<script>var a = 1;</script>
</body></html>"""

# 干净页面：严格口径就够用，不需要 Adjust
CLEAN_PAGE = """<html><body><h2>岗位职责</h2><ul>
<li>参与大模型应用开发与调试。</li>
<li>协助完成知识库 RAG 功能。</li>
</ul><h2>任职要求</h2><ul>
<li>熟练使用 Python 与 FastAPI。</li>
<li>熟悉向量数据库与 Prompt 工程。</li>
<li>每周能到岗 4 天。</li>
</ul></body></html>"""

# 前端渲染的空壳页面：抓得到 HTML，但读不出正文
SHELL_PAGE = """<html><body><div id="app"></div><script>render()</script></body></html>"""

IMAGE_REPLY = """### 任职要求
- 熟悉 Python 与 FastAPI
- 了解 RAG 与向量数据库
- 每周实习 4 天以上
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


class TestHtmlToMarkdown(unittest.TestCase):
    def test_keeps_structure_and_drops_scripts(self) -> None:
        markdown = html_to_markdown(CLEAN_PAGE)
        self.assertIn("## 岗位职责", markdown)
        self.assertIn("- 参与大模型应用开发与调试。", markdown)
        self.assertNotIn("<li>", markdown)

    def test_script_and_style_never_reach_the_output(self) -> None:
        markdown = html_to_markdown(NOISY_PAGE)
        self.assertNotIn("var a = 1", markdown)
        self.assertNotIn("margin:0", markdown)
        self.assertIn("任职要求", markdown)

    def test_table_rows_become_markdown_rows(self) -> None:
        markdown = html_to_markdown("<table><tr><td>地点</td><td>北京</td></tr></table>")
        self.assertIn("| 地点 | 北京 |", markdown)

    def test_head_void_tags_do_not_swallow_the_body(self) -> None:
        """真实页面的 <head> 里都有 <meta> / <link>：它们没有闭合标签，不能吃掉整个 body。"""
        page = (
            '<html><head><meta charset="utf-8"><title>职位详情</title>'
            '<link rel="stylesheet" href="a.css"></head>'
            "<body><h2>任职要求</h2><ul><li>熟悉 Python</li></ul></body></html>"
        )
        markdown = html_to_markdown(page)
        self.assertIn("## 任职要求", markdown)
        self.assertIn("- 熟悉 Python", markdown)
        self.assertEqual(count_lines(markdown), 2)

    def test_body_level_text_is_not_lost(self) -> None:
        """正文直接写在 <body> 下、没有块级标签包着时，也要收得进来。"""
        self.assertEqual(html_to_markdown("<body>岗位职责：负责服务端开发</body>"), "岗位职责：负责服务端开发")

    def test_strict_drops_noise_relaxed_keeps_it(self) -> None:
        markdown = html_to_markdown(NOISY_PAGE)
        strict, dropped = clean_lines(markdown, strict=True)
        relaxed, _ = clean_lines(markdown, strict=False)
        self.assertNotIn("立即申请", strict)
        self.assertIn("立即申请", relaxed)
        self.assertGreater(dropped, 0)
        self.assertGreater(count_lines(relaxed), count_lines(strict))
        # 正常 JD 句子不该被当成噪声：严格口径也要留下
        self.assertIn("熟悉 Python", strict)


class TestSourceDetection(TempCase):
    def test_text_stays_text(self) -> None:
        sources = detect_sources(text=SAMPLE_JD)
        self.assertEqual([source.kind for source in sources], [SOURCE_TEXT])
        self.assertTrue(sources[0].usable)

    def test_pasted_url_is_treated_as_url(self) -> None:
        self.assertTrue(looks_like_url("https://jobs.example.com/p/1"))
        sources = detect_sources(text="https://jobs.example.com/p/1")
        self.assertEqual(sources[0].kind, SOURCE_URL)
        self.assertEqual(sources[0].label, "jobs.example.com")

    def test_normalize_url_fills_scheme_and_rejects_junk(self) -> None:
        self.assertEqual(normalize_url("jobs.example.com/p/1"), "https://jobs.example.com/p/1")
        self.assertEqual(normalize_url("ftp://x/y"), "")
        self.assertEqual(normalize_url("这不是网址"), "")

    def test_files_and_images_are_classified_by_suffix(self) -> None:
        jd_file = self.write("jd.md", SAMPLE_JD)
        image = self.write("shot.png", "not-a-real-png")
        sources = detect_sources(files=[str(jd_file), str(image)], vision_ready=False)
        self.assertEqual([source.kind for source in sources], [SOURCE_FILE, SOURCE_IMAGE])
        self.assertTrue(sources[0].usable)
        self.assertIn("DASHSCOPE_API_KEY", sources[1].note)     # 没配 key：图片分支跳过
        self.assertEqual(len(unusable_notes(sources)), 1)

    def test_missing_file_is_reported_not_raised(self) -> None:
        sources = detect_sources(files=[str(self.tmp / "nope.md")])
        self.assertFalse(sources[0].usable)
        self.assertIn("找不到文件", sources[0].note)

    def test_unknown_suffix_says_what_is_supported(self) -> None:
        path = self.write("jd.pdf", "%PDF-1.4")
        sources = detect_sources(files=[str(path)])
        self.assertFalse(sources[0].usable)
        self.assertIn("不认识", sources[0].note)


class TestFileTool(TempCase):
    def test_markdown_file_is_read_verbatim(self) -> None:
        path = self.write("jd.md", SAMPLE_JD)
        result = JD_FILE_TOOL.run(path=path, out_dir=self.tmp / "out", stem="jd")
        self.assertTrue(result.ok)
        written = self.tmp / "out" / "jd.md"
        self.assertTrue(written.is_file())
        self.assertIn("熟练使用 Python", written.read_text(encoding="utf-8"))

    def test_html_file_keeps_structure(self) -> None:
        path = self.write("jd.html", CLEAN_PAGE)
        parsed = parse_file_markdown(path)
        self.assertEqual(parsed.kind, "html")
        self.assertIn("## 任职要求", parsed.markdown)
        self.assertNotIn("<li>", parsed.markdown)

    def test_json_export_is_flattened(self) -> None:
        path = self.write("jd.json", '{"岗位名称": "AI 实习生", "要求": ["熟悉 Python", "了解 RAG"]}')
        parsed = parse_file_markdown(path)
        self.assertEqual(parsed.kind, "json")
        self.assertIn("- **岗位名称**：AI 实习生", parsed.markdown)
        self.assertIn("- 熟悉 Python", parsed.markdown)

    def test_bad_inputs_return_ok_false(self) -> None:
        cases = [
            self.tmp / "nope.md",
            self.write("jd.pdf", "x"),
            self.write("shot.png", "x"),
        ]
        for path in cases:
            result = JD_FILE_TOOL.run(path=path)
            self.assertFalse(result.ok, f"{path.name} 应该返回 ok=False")
            self.assertTrue(result.error)

    def test_json_syntax_error_is_explained(self) -> None:
        path = self.write("jd.json", "{不是 json")
        result = JD_FILE_TOOL.run(path=path)
        self.assertFalse(result.ok)
        self.assertIn("JSON 解析失败", result.error)


class TestStructTool(TempCase):
    def test_sections_meta_and_capabilities(self) -> None:
        structured = structure_jd_text(MULTI_JD)
        self.assertEqual(structured.position_count, 2)
        first = structured.positions[0]
        self.assertEqual(first.meta, [("公司", "某某科技"), ("工作地点", "北京")])
        self.assertEqual(first.duties, ["参与模型训练与评测。"])
        self.assertEqual(first.requirements, ["熟悉 PyTorch 与深度学习基础。"])

    def test_capability_hits_are_summarised(self) -> None:
        structured = structure_jd_text(SAMPLE_JD)
        names = [name for name, _ in structured.capabilities]
        self.assertIn(CAPABILITY_BY_KEY["python"].name, names)   # 词典命中要出现在摘要里

    def test_unknown_section_is_kept_as_is(self) -> None:
        structured = structure_jd_text(
            "## 团队介绍\n- 我们做 Agent 产品。\n\n**任职要求**\n- 熟悉 Python\n"
        )
        extras = structured.positions[0].extras
        self.assertEqual([title for title, _ in extras], ["团队介绍"])
        self.assertEqual(extras[0][1], ["我们做 Agent 产品。"])
        # 认不出的章节不能被塞进「任职要求」
        self.assertEqual(structured.positions[0].requirements, ["熟悉 Python"])

    def test_render_output_is_judge_ready(self) -> None:
        markdown = render_jd_markdown(structure_jd_text(MULTI_JD), source_line="测试")
        posting = parse_jd_text(markdown, source_file="output/jd.md")
        self.assertEqual(posting.position_count, 2)
        keys = {requirement.capability_key for requirement in posting.requirements}
        self.assertIn("dl_framework", keys)    # 岗位一的要求被主链路读到了
        self.assertTrue(posting.requirements)
        self.assertEqual(posting.company, "某某科技")

    def test_rerender_is_idempotent(self) -> None:
        first = render_jd_markdown(structure_jd_text(MULTI_JD), source_line="测试", generated_at="STAMP")
        again = structure_jd_text(first)
        second = render_jd_markdown(again, source_line="测试", generated_at="STAMP")
        self.assertEqual(first, second)

    def test_title_falls_back_to_first_clean_line(self) -> None:
        structured = structure_jd_text(SAMPLE_JD)
        self.assertEqual(structured.title, "AI 大模型应用开发实习生")
        self.assertTrue(any("第一行" in note for note in structured.notes))
        self.assertNotIn("AI 大模型应用开发实习生", structured.positions[0].sections()[0][1][0:1])

    def test_section_heading_never_becomes_the_doc_title(self) -> None:
        """多来源拼接时，后一段的 `### 岗位职责` 不能顶掉第一行的标题。"""
        text = "AI 算法实习生\n\n任职要求\n- 熟悉 Python\n\n### 岗位职责\n- 参与模型训练。\n"
        structured = structure_jd_text(text)
        self.assertEqual(structured.title, "AI 算法实习生")
        self.assertIn("参与模型训练。", structured.positions[0].duties)

    def test_doc_title_gives_up_when_only_sections_are_left(self) -> None:
        structured = structure_jd_text("### 任职要求\n- 熟悉 Python\n- 了解 RAG 与向量库\n- 熟悉 Docker\n")
        self.assertEqual(structured.title, "未命名岗位")

    def test_file_title_line_never_leaks_into_the_bullets(self) -> None:
        text = (
            "# AI 算法实习生（日常实习 / 秋招可投）\n\n**岗位类型**：在校实习\n\n"
            "**任职要求**\n- 熟悉 Python\n- 了解 RAG 与向量库\n- 熟悉 Docker\n"
        )
        structured = structure_jd_text(text)
        self.assertEqual(structured.title, "AI 算法实习生（日常实习 / 秋招可投）")
        self.assertEqual(structured.positions[0].meta, [("岗位类型", "在校实习")])
        self.assertEqual(structured.positions[0].sections()[0][0], "任职要求")
        self.assertNotIn("#", "".join(structured.positions[0].requirements))

    def test_tool_returns_ok_false_without_source(self) -> None:
        result = JD_STRUCT_TOOL.run()
        self.assertFalse(result.ok)
        self.assertTrue(result.error)


class TestJDAgentLoop(TempCase):
    def test_text_source_runs_the_whole_loop(self) -> None:
        state = run_jd_agent(JDRequest(text=SAMPLE_JD), out_dir=self.tmp, stem="jd")
        self.assertEqual(
            [step.action for step in state.trace],
            ["DetectSource", "ReadText", "StructureJD", "WriteMarkdown"],
        )
        self.assert_trace_shape(state)
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue(state.stop_reason)
        self.assertTrue((self.tmp / "jd.md").is_file())
        self.assertIn("**任职要求**", state.markdown)

    def test_text_source_never_touches_the_network(self) -> None:
        opener = FakeOpener(error=AssertionError("文本分支不该抓网页"))
        vision = FakeVision(error=AssertionError("文本分支不该调模型"))
        state = run_jd_agent(JDRequest(text=SAMPLE_JD), vision_client=vision, fetcher=opener)
        self.assertEqual(opener.calls, [])
        self.assertEqual(vision.calls, [])
        self.assertEqual(state.pages, {})
        self.assertEqual(state.decision, DECISION_STOP)

    def test_file_source_uses_the_file_tool(self) -> None:
        path = self.write("jd.md", MULTI_JD)
        state = run_jd_agent(JDRequest(files=(str(path),)), out_dir=self.tmp, stem="fromfile")
        self.assertEqual(
            [step.action for step in state.trace],
            ["DetectSource", "ParseFile", "StructureJD", "WriteMarkdown"],
        )
        self.assertIn(JD_FILE_TOOL.slug, state.trace[1].observation)
        self.assertEqual(state.structured.position_count, 2)

    def test_clean_url_page_needs_no_adjust(self) -> None:
        opener = FakeOpener({"https://x.com/job": CLEAN_PAGE})
        state = run_jd_agent(
            JDRequest(urls=("https://x.com/job",)), fetcher=opener, out_dir=self.tmp, stem="url"
        )
        self.assertEqual([step.action for step in state.trace][:2], ["DetectSource", "FetchUrl"])
        self.assertNotIn(DECISION_ADJUST, [step.decision for step in state.trace])
        self.assertEqual(opener.calls, ["https://x.com/job"])
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertIn("熟练使用 Python 与 FastAPI。", state.markdown)

    def test_noisy_url_page_triggers_one_adjust_without_refetching(self) -> None:
        opener = FakeOpener({"https://x.com/job": NOISY_PAGE})
        state = run_jd_agent(
            JDRequest(urls=("https://x.com/job",)), fetcher=opener, out_dir=self.tmp, stem="url"
        )
        actions = [step.action for step in state.trace]
        self.assertEqual(
            actions, ["DetectSource", "FetchUrl", "RelaxFetch", "StructureJD", "WriteMarkdown"]
        )
        adjusts = [step for step in state.trace if step.decision == DECISION_ADJUST]
        self.assertEqual(len(adjusts), 1)
        self.assertIn("放宽", adjusts[0].decision_reason)
        self.assertEqual(opener.calls, ["https://x.com/job"], "放宽口径不该重新抓一次")
        strict_lines = count_lines(clean_lines(html_to_markdown(NOISY_PAGE), strict=True)[0])
        self.assertLess(strict_lines, MIN_URL_LINES)
        self.assertGreater(state.chunks[0].line_count, strict_lines)

    def test_shell_page_asks_instead_of_inventing_content(self) -> None:
        opener = FakeOpener({"https://x.com/job": SHELL_PAGE})
        state = run_jd_agent(JDRequest(urls=("https://x.com/job",)), fetcher=opener)
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertTrue(state.question)
        self.assertTrue(state.question_reason)
        self.assertEqual(state.written, [])
        self.assertIn("截图", state.question)
        self.assertNotIn("任职要求", state.markdown)

    def test_image_source_is_transcribed_by_the_vision_client(self) -> None:
        shot = self.write("shot.png", "fake-png-bytes")
        vision = FakeVision()
        state = run_jd_agent(
            JDRequest(images=(str(shot),)), vision_client=vision, out_dir=self.tmp, stem="img"
        )
        self.assertEqual(
            [step.action for step in state.trace],
            ["DetectSource", "OcrImage", "StructureJD", "WriteMarkdown"],
        )
        self.assertEqual(len(vision.calls), 1)
        self.assertIn("data:image/png;base64,", str(vision.calls[0]["messages"]))
        self.assertEqual(state.structured.bullet_count, 3)
        self.assertEqual(state.decision, DECISION_STOP)

    def test_failed_image_call_is_degraded_then_asked(self) -> None:
        shot = self.write("shot.png", "fake-png-bytes")
        vision = FakeVision(error=LLMError("HTTP 500"))
        state = run_jd_agent(JDRequest(images=(str(shot),)), vision_client=vision)
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertTrue(state.notes)
        self.assertEqual([step.decision for step in state.trace][:3], [DECISION_CONTINUE] * 3)
        self.assert_trace_shape(state)

    def test_failed_fetch_is_degarded_and_still_uses_the_rest(self) -> None:
        opener = FakeOpener(error=SourceError("网络不可达：连接超时"))
        state = run_jd_agent(
            JDRequest(text=SAMPLE_JD, urls=("https://x.com/job",)), fetcher=opener
        )
        self.assertEqual(state.decision, DECISION_STOP)
        self.assertTrue(state.failed_chunks)
        self.assertIn("没取到", state.stop_reason)
        self.assertIn("熟练使用 Python", state.markdown)     # 文本来源照常出结果

    def test_thin_text_asks_a_question(self) -> None:
        state = run_jd_agent(JDRequest(text="招实习生一名。"))
        self.assertEqual(state.decision, DECISION_ASK)
        self.assertLess(state.structured.bullet_count, MIN_JD_BULLETS)
        self.assertTrue(state.question)

    def test_graph_keeps_the_documented_nodes(self) -> None:
        nodes = set(compiled_graph().get_graph().nodes)
        self.assertTrue(set(NODE_NAMES).issubset(nodes), f"图里缺少节点：{set(NODE_NAMES) - nodes}")

    def test_renders_console_and_trace_without_network(self) -> None:
        state = run_jd_agent(JDRequest(text=SAMPLE_JD))
        console = "\n".join(render_jd_console(state))
        self.assertIn("Action      : DetectSource", console)
        self.assertIn("Decision    : Stop", console)
        trace = render_jd_trace(state)
        self.assertIn("| # | Action |", trace)
        self.assertIn("**StopReason**", trace)


class TestJDAgentCLI(TempCase):
    def run_cli(self, argv) -> int:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(argv)
        self.last_output = buffer.getvalue()
        return code

    def test_agent_mode_writes_markdown_and_trace(self) -> None:
        out = self.tmp / "jd.md"
        code = self.run_cli(
            ["--jd-agent", "--jd-text", SAMPLE_JD, "--jd-out", str(out), "--jd-trace"]
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue(out.is_file())
        self.assertTrue((self.tmp / "jd.trace.md").is_file())
        self.assertIn("**任职要求**", out.read_text(encoding="utf-8"))
        self.assertIn("Action      : StructureJD", self.last_output)

    def test_agent_mode_without_source_is_an_input_error(self) -> None:
        self.assertEqual(self.run_cli(["--jd-agent", "--quiet"]), cli.EXIT_ERROR)

    def test_unusable_source_is_an_input_error_with_the_reason(self) -> None:
        code = self.run_cli(["--jd-agent", "--jd-image", str(self.tmp / "nope.png")])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("没有一个能读的 JD 来源", self.last_output)

    def test_quiet_run_prints_nothing(self) -> None:
        out = self.tmp / "jd.md"
        code = self.run_cli(["--jd-agent", "--jd-file", str(self.write("jd.md", SAMPLE_JD)), "--jd-out", str(out), "--quiet"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(self.last_output.strip(), "")
        self.assertTrue(out.is_file())

    def test_thin_input_leaves_with_exit_code_three(self) -> None:
        code = self.run_cli(["--jd-agent", "--jd-text", "招实习生一名。"])
        self.assertEqual(code, cli.EXIT_ASK)
        self.assertIn("需要你回答 1 个问题", self.last_output)

    def test_new_tools_are_registered(self) -> None:
        self.assertIs(tools.TOOL_BY_SLUG["jd-file"], tools.JD_FILE_TOOL)
        self.assertIs(tools.TOOL_BY_SLUG["jd-struct"], tools.JD_STRUCT_TOOL)
        slugs = [tool.slug for tool in tools.TOOLS]
        # 顺序有意义：简历排版在前，接着是 JD 两个工具，再是 Resume Agent 的三个工具
        self.assertEqual(
            slugs[:3], ["resume", "jd-file", "jd-struct"]
        )
        self.assertEqual(len(slugs), len(set(slugs)), "工具 slug 不能重复")
        self.assertEqual(len(tools.tool_help_lines()), len(slugs))

    def test_image_branch_reads_env_only_when_images_are_given(self) -> None:
        env = self.write("keys.env", "DASHSCOPE_API_KEY=\n")
        code = self.run_cli(
            ["--jd-agent", "--jd-text", SAMPLE_JD, "--jd-out", str(self.tmp / "jd.md"), "--env", str(env)]
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertNotIn("[配置]", self.last_output)          # 纯文本分支不读 .env、不打印配置


if __name__ == "__main__":
    unittest.main()
