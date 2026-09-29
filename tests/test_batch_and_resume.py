"""v1.1 的批量能力（默认读 input/jd 全部岗位）+ v2.0 的工具区（简历排版、三套风格）。

全部用临时目录，不碰真实 input/ 与 output/；大模型一律注入假 client，一个网络请求都不发。

运行方式（项目根目录）：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jd_agent import cli, tools  # noqa: E402
from jd_agent.domain.jd import load_jd_positions  # noqa: E402
from jd_agent.core.schema import STATUS_IDLE, STATUS_OK, STATUS_SKIPPED  # noqa: E402

JD_TWO_POSITIONS = """# 某公司 AI 岗

## 岗位一：大模型应用实习生

**任职要求**

- 熟练使用 Python，了解 FastAPI 等 Web 框架。
- 熟悉 Docker 基础部署流程。

## 岗位二：AI 产品实习生

**任职要求**

- 熟悉大模型应用场景，能写产品需求文档。
- 熟练使用 Python 做数据分析。
"""

JD_NO_REQUIREMENT = """# 只有一段说明，没有任何可识别的要求

这是一份没有任职要求章节的文件，用来验证批量模式会跳过它。
"""

PROJECT = """# 项目：AI 周报助手

## 我做了什么

- 独立完成采集与摘要流程，代码已开源到 GitHub。
- 用 Python 写了采集脚本，接入大模型 API 做摘要。

## 结果

- 摘要可用率从 62% 提升到 88%。
"""

RESUME = """# 周每每 · 个人简历

> 求职意向：**首选 AI 大模型应用开发实习生** ｜ 2027 届本科在读
> 可实习 6 个月以上 ｜ 每周 4 天

## 项目经历

### 智能旅行助手 Agent ｜ 全栈开发

**技术栈**：Vue3 + Python

- 负责后端接口与多智能体编排，代码已开源。

## 基本信息

| 项目 | 内容 |
|------|------|
| 姓名 | 周每每 |
| 联系方式 | [手机号] ｜ [邮箱] |

## 自我评价

- 踏实细心，执行力好。
"""

# 重构前（v1.1）用 classic 风格渲染出来的两份结果，用来锁住「默认风格输出没变」。
CLASSIC_MD_BASELINE = (
    "# 周每每\n\n"
    "> 求职意向：**首选 AI 大模型应用开发实习生** ｜ 2027 届本科在读\n"
    "> 可实习 6 个月以上 ｜ 每周 4 天\n\n"
    "## 项目经历\n\n"
    "### 智能旅行助手 Agent ｜ 全栈开发\n\n"
    "**技术栈**：Vue3 + Python\n\n"
    "- 负责后端接口与多智能体编排，代码已开源。\n\n"
    "## 基本信息\n\n"
    "| 项目 | 内容 |\n"
    "| --- | --- |\n"
    "| 姓名 | 周每每 |\n"
    "| 联系方式 | [手机号] ｜ [邮箱] |\n\n"
    "## 自我评价\n\n"
    "- 踏实细心，执行力好。\n"
)

CLASSIC_BODY_BASELINE = """
<div class="resume-doc">
<header>
<h1>周每每</h1>
<p class="tagline"><span>求职意向：<strong>首选 AI 大模型应用开发实习生</strong> ｜ 2027 届本科在读</span><span>可实习 6 个月以上 ｜ 每周 4 天</span></p>
</header>
<section><h2>项目经历</h2>
<h3>智能旅行助手 Agent ｜ 全栈开发</h3>
<p><strong>技术栈</strong>：Vue3 + Python</p>
<ul><li>负责后端接口与多智能体编排，代码已开源。</li></ul>
</section>
<section><h2>基本信息</h2>
<table><thead><tr><th>项目</th><th>内容</th></tr></thead><tbody><tr><td>姓名</td><td>周每每</td></tr><tr><td>联系方式</td><td>[手机号] ｜ [邮箱]</td></tr></tbody></table>
</section>
<section><h2>自我评价</h2>
<ul><li>踏实细心，执行力好。</li></ul>
</section>
</div>
</body>
</html>"""

FAKE_REPLY = json.dumps({"draft": "用 Python 写了采集脚本，摘要可用率从 62% 提升到 88%。"}, ensure_ascii=False)


class CountingClient:
    """假客户端：回一段合法内容，只记录被调用了几次（用来验证共享预算）。"""

    def __init__(self, reply: str = FAKE_REPLY):
        self.reply = reply
        self.calls: List[Any] = []

    def chat(self, messages, model=None, max_tokens=None, temperature=None) -> str:
        self.calls.append({"messages": messages, "model": model})
        return self.reply


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.jd_dir = self.tmp / "jd"
        self.project_dir = self.tmp / "project"
        self.resume_dir = self.tmp / "profile"
        for item in (self.jd_dir, self.project_dir, self.resume_dir):
            item.mkdir()
        self.env_file = self._write(self.tmp / "empty.env", "")
        self.project = self._write(self.project_dir / "project.md", PROJECT)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def _patch_dirs(self):
        return mock.patch.multiple(
            cli,
            JD_DIR=self.jd_dir,
            PROJECT_DIR=self.project_dir,
            RESUME_DIR=self.resume_dir,
            INPUT_DIR=self.tmp,
        )

    def _payload(self, out: str, name: str):
        return json.loads((self.tmp / out / name).read_text(encoding="utf-8"))


# ---- 1. 一个文件里的多个岗位 -------------------------------------------------


class TestJdPositions(TempCase):
    def test_all_positions_are_parsed_with_distinct_ids(self) -> None:
        path = self._write(self.jd_dir / "jd.md", JD_TWO_POSITIONS)
        postings = load_jd_positions(path)
        self.assertEqual([item.title for item in postings], ["岗位一：大模型应用实习生", "岗位二：AI 产品实习生"])
        self.assertEqual([item.jd_id for item in postings], ["jd#1", "jd#2"])
        self.assertTrue(all(item.requirements for item in postings))
        self.assertEqual(postings[0].position_count, 2)


# ---- 2. 批量：不指定 --jd 时把 input/jd 下所有岗位都跑一遍 ---------------------


class TestBatchRun(TempCase):
    def test_without_jd_flag_every_posting_gets_its_own_report(self) -> None:
        self._write(self.jd_dir / "jd.md", JD_TWO_POSITIONS)
        self._write(self.jd_dir / "noreq.md", JD_NO_REQUIREMENT)
        with self._patch_dirs():
            code = cli.main(["--project", str(self.project), "--out", str(self.tmp / "batch"), "--quiet"])
        self.assertEqual(code, cli.EXIT_OK)
        names = sorted(item.name for item in (self.tmp / "batch").glob("*"))
        self.assertEqual(
            names,
            [
                "resume_decision_jd_1.html",
                "resume_decision_jd_1.json",
                "resume_decision_jd_1.md",
                "resume_decision_jd_2.html",
                "resume_decision_jd_2.json",
                "resume_decision_jd_2.md",
            ],
        )
        first = self._payload("batch", "resume_decision_jd_1.json")
        second = self._payload("batch", "resume_decision_jd_2.json")
        self.assertEqual(first["jd"]["title"], "岗位一：大模型应用实习生")
        self.assertEqual(second["jd"]["title"], "岗位二：AI 产品实习生")

    def test_single_posting_keeps_the_v10_file_name(self) -> None:
        self._write(self.jd_dir / "noreq.md", JD_NO_REQUIREMENT)
        self._write(self.jd_dir / "single.md", JD_TWO_POSITIONS.split("## 岗位二")[0])
        with self._patch_dirs():
            code = cli.main(
                ["--project", str(self.project), "--out", str(self.tmp / "single"), "--formats", "json", "--quiet"]
            )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue((self.tmp / "single" / "resume_decision.json").is_file())

    def test_explicit_jd_file_still_runs_only_the_first_posting(self) -> None:
        jd = self._write(self.jd_dir / "jd.md", JD_TWO_POSITIONS)
        with self._patch_dirs():
            code = cli.main(
                ["--jd", str(jd), "--project", str(self.project), "--out", str(self.tmp / "one"), "--formats", "json", "--quiet"]
            )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue((self.tmp / "one" / "resume_decision.json").is_file())
        self.assertEqual(
            self._payload("one", "resume_decision.json")["jd"]["title"], "岗位一：大模型应用实习生"
        )

    def test_unknown_jd_title_is_an_input_error(self) -> None:
        jd = self._write(self.jd_dir / "jd.md", JD_TWO_POSITIONS)
        with self._patch_dirs():
            code = cli.main(["--jd", str(jd), "--jd-title", "岗位九", "--project", str(self.project), "--quiet"])
        self.assertEqual(code, cli.EXIT_ERROR)

    def test_postings_share_one_llm_budget(self) -> None:
        self._write(self.jd_dir / "jd.md", JD_TWO_POSITIONS)
        client = CountingClient()
        with self._patch_dirs():
            code = cli.main(
                [
                    "--llm",
                    "--llm-max-calls",
                    "3",
                    "--env",
                    str(self.env_file),
                    "--project",
                    str(self.project),
                    "--out",
                    str(self.tmp / "llm"),
                    "--formats",
                    "json",
                    "--quiet",
                ],
                text_client=client,
            )
        self.assertEqual(code, cli.EXIT_OK)
        # 两个岗位各自要 2 块内容（草稿 + 润色），但整次运行的预算是 3 次
        self.assertEqual(len(client.calls), 3)
        first = self._payload("llm", "resume_decision_jd_1.json")["llm"]
        second = self._payload("llm", "resume_decision_jd_2.json")["llm"]
        self.assertEqual([item["status"] for item in first["blocks"]], [STATUS_OK, STATUS_IDLE, STATUS_OK])
        self.assertEqual(second["calls"], 3)
        self.assertEqual([item["status"] for item in second["blocks"]], [STATUS_OK, STATUS_IDLE, STATUS_SKIPPED])

    def test_ask_result_still_exits_three(self) -> None:
        thin = self._write(self.project_dir / "thin.md", "# 项目：校园问答小助手\n\n- 前端用 Streamlit。\n")
        self._write(self.jd_dir / "jd.md", JD_TWO_POSITIONS)
        with self._patch_dirs():
            code = cli.main(["--project", str(thin), "--out", str(self.tmp / "ask"), "--formats", "json", "--quiet"])
        self.assertEqual(code, cli.EXIT_ASK)


# ---- 3. 简历排版 -------------------------------------------------------------


class TestResumeParsing(TempCase):
    def test_sections_are_ordered_and_content_is_preserved(self) -> None:
        resume = tools.parse_resume(RESUME, "profile/x.md")
        self.assertEqual(resume.name, "周每每")
        self.assertEqual(resume.section_titles, ["项目经历", "基本信息", "自我评价"])
        self.assertEqual(resume.tagline[0], "求职意向：**首选 AI 大模型应用开发实习生** ｜ 2027 届本科在读")
        self.assertFalse(any("章节顺序已按模板重排" in note for note in resume.notes))

    def test_markdown_output_is_normalised(self) -> None:
        resume = tools.parse_resume(RESUME, "profile/x.md")
        text = tools.render_markdown(resume)
        self.assertTrue(text.startswith("# 周每每\n"))
        self.assertIn("> 求职意向", text)
        self.assertIn("## 基本信息", text)
        self.assertIn("| 项目 | 内容 |", text)
        self.assertIn("| --- | --- |", text)
        self.assertIn("- 负责后端接口与多智能体编排，代码已开源。", text)
        self.assertNotIn("\n\n\n", text)
        self.assertTrue(text.endswith("\n"))

    def test_html_is_self_contained_and_scoped(self) -> None:
        resume = tools.parse_resume(RESUME, "profile/x.md")
        page = tools.render_html(resume)
        self.assertIn('<div class="resume-doc">', page)
        self.assertIn("<style>", page)
        self.assertIn("<strong>首选 AI 大模型应用开发实习生</strong>", page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        # 样式全部限定在 .resume-doc 里，嵌进别的页面时不会污染宿主
        classic_css = tools.resume_styles.get_style("classic").css
        self.assertTrue(classic_css.strip().startswith(".resume-doc"))
        self.assertNotIn("\nbody {", classic_css)
        self.assertIn("body { margin: 0;", page)
        preview = tools.render_html(resume, standalone=False)
        self.assertNotIn("body { margin: 0;", preview)

    def test_table_columns_are_padded(self) -> None:
        resume = tools.parse_resume("## 基本信息\n\n| 项目 | 内容 |\n| --- | --- |\n| 姓名 |\n", "x.md")
        self.assertIn("| 姓名 |  |", tools.render_markdown(resume))

    def test_project_description_can_be_merged_into_the_resume(self) -> None:
        resume = tools.build_resume(self._write(self.resume_dir / "r.md", RESUME), [self.project])
        self.assertIn("项目经历", resume.section_titles)
        self.assertTrue(any("并入项目描述" in note for note in resume.notes))
        text = tools.render_markdown(resume)
        self.assertIn("### 项目：AI 周报助手", text)
        self.assertIn("- 摘要可用率从 62% 提升到 88%。", text)

    def test_write_resume_files_writes_only_requested_formats(self) -> None:
        resume = tools.parse_resume(RESUME, "profile/x.md")
        written = tools.write_resume_files(resume, self.tmp / "out", ("md",), stem="resume")
        self.assertEqual([item.name for item in written], ["resume.md"])
        self.assertTrue(written[0].read_text(encoding="utf-8").startswith("# 周每每"))
        self.assertFalse((self.tmp / "out" / "resume.html").exists())

    def test_classic_output_matches_the_pre_refactor_baseline(self) -> None:
        resume = tools.parse_resume(RESUME, "profile/x.md")
        self.assertEqual(tools.render_markdown(resume), CLASSIC_MD_BASELINE)
        page = tools.render_html(resume)
        self.assertEqual(page.split("<body>", 1)[1], CLASSIC_BODY_BASELINE)


class TestResumeCli(TempCase):
    def _resume_file(self) -> Path:
        return self._write(self.resume_dir / "我的简历.md", RESUME)

    def test_build_resume_writes_both_files(self) -> None:
        self._resume_file()
        with self._patch_dirs():
            code = cli.main(["--build-resume", "--out", str(self.tmp / "docs"), "--quiet"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue((self.tmp / "docs" / "resume.md").is_file())
        self.assertTrue((self.tmp / "docs" / "resume.html").is_file())

    def test_example_files_are_skipped_when_picking_the_default(self) -> None:
        self._write(self.resume_dir / "示例简历-可替换.md", "# 示例\n\n- 占位\n")
        real = self._resume_file()
        with self._patch_dirs():
            self.assertEqual(cli.resolve_resume_path(""), real)
            self.assertEqual(cli.resolve_resume_path(str(real)), real)

    def test_explicit_resume_file_and_project_merge(self) -> None:
        other = self._write(self.resume_dir / "别人的简历.md", "# 张三\n\n- 一句话\n")
        with self._patch_dirs():
            code = cli.main(
                [
                    "--build-resume",
                    "--resume-file",
                    str(other),
                    "--resume-project",
                    str(self.project),
                    "--out",
                    str(self.tmp / "merge"),
                    "--resume-name",
                    "我的简历",
                    "--quiet",
                ]
            )
        self.assertEqual(code, cli.EXIT_OK)
        text = (self.tmp / "merge" / "我的简历.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# 张三"))
        self.assertIn("摘要可用率从 62% 提升到 88%", text)

    def test_bad_format_and_missing_file_are_input_errors(self) -> None:
        self._resume_file()
        with self._patch_dirs():
            self.assertEqual(cli.main(["--build-resume", "--resume-format", "pdf", "--quiet"]), cli.EXIT_ERROR)
            self.assertEqual(cli.main(["--build-resume", "--resume-file", str(self.tmp / "nope.md"), "--quiet"]), cli.EXIT_ERROR)
            self.assertEqual(cli.main(["--build-resume", "--resume-project", str(self.tmp / "nope.md"), "--quiet"]), cli.EXIT_ERROR)

    def test_style_flag_switches_style_and_default_file_name(self) -> None:
        self._resume_file()
        with self._patch_dirs():
            code = cli.main(
                ["--build-resume", "--resume-style", "accent", "--out", str(self.tmp / "accent"), "--quiet"]
            )
        self.assertEqual(code, cli.EXIT_OK)
        page = (self.tmp / "accent" / "resume-accent.html").read_text(encoding="utf-8")
        self.assertIn("--accent: #0f766e", page)                 # 强调竖线那套的配色
        markdown_text = (self.tmp / "accent" / "resume-accent.md").read_text(encoding="utf-8")
        self.assertIn("\n---\n", markdown_text)                  # 章节之间加分隔线

    def test_explicit_resume_name_wins_over_the_style_suffix(self) -> None:
        self._resume_file()
        with self._patch_dirs():
            code = cli.main(
                [
                    "--build-resume",
                    "--resume-style",
                    "structure",
                    "--resume-name",
                    "我的简历",
                    "--out",
                    str(self.tmp / "named"),
                    "--quiet",
                ]
            )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue((self.tmp / "named" / "我的简历.md").is_file())
        self.assertFalse((self.tmp / "named" / "resume-structure.md").exists())

    def test_unknown_style_is_an_input_error(self) -> None:
        self._resume_file()
        with self._patch_dirs():
            self.assertEqual(
                cli.main(["--build-resume", "--resume-style", "花哨", "--quiet"]), cli.EXIT_ERROR
            )

    def test_build_resume_never_touches_the_model(self) -> None:
        self._resume_file()

        class Bomb:
            def __init__(self, *args, **kwargs):
                raise AssertionError("简历排版不应该创建大模型客户端")

        with self._patch_dirs(), mock.patch("jd_agent.tools.resume.read_text", wraps=tools.resume.read_text):
            with mock.patch("jd_agent.agents.agent.OpenAICompatClient", Bomb):
                code = cli.main(["--build-resume", "--llm", "--out", str(self.tmp / "offline"), "--quiet"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertTrue((self.tmp / "offline" / "resume.html").is_file())


# ---- 4. 工具区：注册表 + 三套风格 --------------------------------------------


class TestToolRegistry(unittest.TestCase):
    def test_resume_tool_is_registered(self) -> None:
        self.assertIs(tools.get_tool("resume"), tools.RESUME_TOOL)
        self.assertIs(tools.TOOL_BY_SLUG["resume"], tools.RESUME_TOOL)
        self.assertEqual(tools.RESUME_TOOL.slug, "resume")
        self.assertTrue(tools.RESUME_TOOL.title)
        self.assertTrue(tools.RESUME_TOOL.usage.startswith("python main.py"))
        self.assertEqual(tools.TOOLS[0], tools.RESUME_TOOL)      # 简历排版仍是第一个注册的工具
        slugs = [tool.slug for tool in tools.TOOLS]
        self.assertEqual(len(slugs), len(set(slugs)), "工具 slug 不能重复")
        self.assertIsNone(tools.get_tool("没有这个工具"))

    def test_help_lines_are_ready_for_the_cli(self) -> None:
        lines = tools.tool_help_lines()
        self.assertEqual(len(lines), len(tools.TOOLS))
        self.assertIn("简历排版", lines[0])
        self.assertIn("resume", lines[0])

    def test_tool_result_is_serialisable(self) -> None:
        result = tools.ToolResult(ok=False, error="坏了", summary=[("姓名", "张三")], notes=["n"])
        self.assertFalse(result.ok)
        self.assertEqual(result.files, [])
        self.assertEqual(
            result.as_dict(),
            {
                "ok": False,
                "error": "坏了",
                "summary": [{"label": "姓名", "value": "张三"}],
                "notes": ["n"],
                "files": [],
            },
        )


class TestResumeStyles(TempCase):
    def test_three_styles_render_and_differ(self) -> None:
        resume_file = self._write(self.resume_dir / "r.md", RESUME)
        styles = tools.resume_styles.all_styles()
        self.assertEqual([style.key for style in styles], ["classic", "structure", "accent"])
        blocks = set()
        for style in styles:
            resume = tools.build_resume(resume_file, style=style.key)
            self.assertEqual(resume.style, style.key)
            page = tools.render_html(resume)
            blocks.add(page.split("<style>", 1)[1].split("</style>", 1)[0])
            self.assertIn('<div class="resume-doc">', page)
            self.assertIn("body { margin: 0;", page)                  # 单独打开时才有纸面样式
            self.assertNotIn("body { margin: 0;", tools.render_html(resume, standalone=False))
            self.assertTrue(style.css.strip().startswith(".resume-doc"))
            self.assertNotIn("\nbody {", style.css)                   # 样式都限定在 .resume-doc 里
        self.assertEqual(len(blocks), len(styles))                    # 三套样式互不相同

    def test_markdown_only_changes_where_the_style_says_so(self) -> None:
        resume_file = self._write(self.resume_dir / "r.md", RESUME)
        classic = tools.render_markdown(tools.build_resume(resume_file))
        structure = tools.render_markdown(tools.build_resume(resume_file, style="structure"))
        accent = tools.render_markdown(tools.build_resume(resume_file, style="accent"))
        self.assertEqual(classic, structure)                          # 架构清晰只换 CSS
        self.assertNotIn("\n---\n", classic)
        self.assertIn("\n---\n", accent)                              # 强调竖线：章节之间加分隔线
        self.assertIn("### 智能旅行助手 Agent ｜ 全栈开发", classic)
        self.assertIn("**智能旅行助手 Agent ｜ 全栈开发**", accent)     # 子条目改成加粗行

    def test_default_stem_follows_the_style(self) -> None:
        self.assertEqual(tools.default_stem("classic"), "resume")
        self.assertEqual(tools.default_stem("structure"), "resume-structure")
        self.assertEqual(tools.default_stem("accent"), "resume-accent")

    def test_unknown_style_is_an_error_not_an_exception(self) -> None:
        resume_file = self._write(self.resume_dir / "r.md", RESUME)
        result = tools.RESUME_TOOL.run(resume_file=resume_file, out_dir=self.tmp / "o", style="花哨")
        self.assertFalse(result.ok)
        self.assertIn("没有这种排版风格", result.error)
        self.assertIn("classic", result.error)
        self.assertEqual(result.written, [])


class TestResumeToolRun(TempCase):
    def test_run_writes_files_and_reports_a_summary(self) -> None:
        resume_file = self._write(self.resume_dir / "r.md", RESUME)
        result = tools.RESUME_TOOL.run(
            resume_file=resume_file, out_dir=self.tmp / "docs", formats=("html",), style="structure"
        )
        self.assertTrue(result.ok, result.error)
        self.assertEqual([item.name for item in result.written], ["resume-structure.html"])
        self.assertTrue(result.written[0].is_file())
        labels = dict(result.summary)
        self.assertEqual(labels["姓名"], "周每每")
        self.assertIn("架构清晰", labels["排版风格"])
        # 风格只写在摘要里；notes 说的是「这篇简历被怎么规整过」
        self.assertFalse(any("排版风格" in note for note in result.notes))
        self.assertEqual(result.as_dict()["files"], result.files)

    def test_bad_input_is_reported_instead_of_raised(self) -> None:
        resume_file = self._write(self.resume_dir / "r.md", RESUME)
        cases = [
            ("文件不存在", dict(resume_file=self.tmp / "nope.md")),
            ("格式不认识", dict(resume_file=resume_file, formats=("pdf",))),
            ("格式为空", dict(resume_file=resume_file, formats=())),
            ("项目文件缺失", dict(resume_file=resume_file, projects=(self.tmp / "nope.md",))),
        ]
        for label, case in cases:
            with self.subTest(case=label):
                result = tools.RESUME_TOOL.run(out_dir=self.tmp / "o", **case)
                self.assertFalse(result.ok)
                self.assertTrue(result.error)
                self.assertEqual(result.written, [])


if __name__ == "__main__":
    unittest.main()
