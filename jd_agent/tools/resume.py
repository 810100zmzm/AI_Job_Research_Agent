"""简历排版工具：把手写的简历 Markdown 重排成「简约大方」的 Markdown / HTML。

命令行（jd_agent/cli.py）与 Streamlit 前端（streamlit_app.py）共用这一份实现：
    python main.py --build-resume [--resume-style classic|compact|accent] [--resume-file ...]

工具类 `ResumeTool` 负责「解析 + 校验 + 落盘」，给命令行用；页面只调下面的纯函数，
不写磁盘。两者都只做规则，不联网、不读 .env、不调用大模型。

输入是 input/profile 下那种结构：

    # 姓名 · 个人简历
    > 求职意向：…… ｜ ……
    ## 基本信息 / ## 教育背景 / ## 项目经历 ……

输出两份自包含文件：
    * Markdown —— 层级、列表符号、表格全部统一，方便再编辑或粘贴到投递平台；
    * HTML     —— 内嵌 CSS、A4 宽度、无外链，浏览器里可直接打印成 PDF。

排版规则（纯规则，全部写在这里，可直接检查）：
    1. 章节顺序按模板固定（基本信息 / 教育背景 / 技能清单 / 项目经历 / 实习经历 /
       荣誉奖项 / 竞赛与获奖 / 校园经历 / 自我评价），没写的章节不出现，
       文件里多出来的章节按原顺序排在后面；
    2. 列表统一成 `- `；段落之间只留一个空行；行尾空白与重复空格清掉；
    3. 表格按最大列数补齐，分隔行统一成 `| --- |`，单元格里的 `|` 换成全角；
    4. 只做结构重排与空白规整：不新增、不改写任何事实，也不判断「该不该写」；
    5. 「怎么排」另由 resume_styles 里的三套风格决定（CSS + 章节分隔线 + 子条目写法）；
       默认的经典简约与重构前那份 resume_doc.py 的输出逐字一致。
"""
from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..jd import display_path, read_text
from ..project import parse_project_text
from .base import Tool, ToolResult
from .resume_styles import DEFAULT_STYLE, PAGE_CSS, ResumeStyle, get_style

HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.*\S)\s*$")
BULLET_RE = re.compile(r"^\s*(?:[-*+•·▪]|\d{1,2}\s*[.、)])\s+(?P<text>.*\S)\s*$")
RULE_RE = re.compile(r"^\s*(?:[-=*_—]{3,}|<[^>]+>)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|(?P<cells>.*)\|\s*$")
TABLE_DIVIDER_RE = re.compile(r"^:?-{2,}:?$")
QUOTE_RE = re.compile(r"^\s*>\s?(?P<text>.*)$")
NUMBER_PREFIX_RE = re.compile(r"^[一二三四五六七八九十\d]+\s*[、.．)）]\s*")
TITLE_SUFFIX_RE = re.compile(r"\s*[·・|｜]\s*(?:个人)?(?:简历|经历|resume)\s*$", re.IGNORECASE)

# 章节模板顺序；不在表里的章节按原文顺序排在后面
SECTION_ORDER = (
    "基本信息",
    "教育背景",
    "技能清单",
    "专业技能",
    "项目经历",
    "实习经历",
    "工作经历",
    "荣誉奖项",
    "竞赛与获奖",
    "校园经历",
    "自我评价",
)

@dataclass
class Table:
    """一张表格：第一行当表头，其余当数据行。"""

    header: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    @property
    def width(self) -> int:
        return max([len(self.header)] + [len(row) for row in self.rows] + [0])

    @property
    def empty(self) -> bool:
        return not self.header and not self.rows


@dataclass
class Node:
    """章节里的一个内容块。"""

    kind: str
    text: str = ""
    level: int = 0
    lines: List[str] = field(default_factory=list)
    table: Optional[Table] = None


@dataclass
class Section:
    title: str
    nodes: List[Node] = field(default_factory=list)


@dataclass
class Resume:
    """一份排好版的简历。"""

    name: str
    source_file: str
    tagline: List[str] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    style: str = DEFAULT_STYLE          # 排版风格 key，见 resume_styles

    @property
    def section_titles(self) -> List[str]:
        return [section.title for section in self.sections]

    @property
    def item_count(self) -> int:
        """列表要点 + 段落 + 表格行，用来在终端/前端给一个「简历有多厚」的直觉。"""
        total = 0
        for section in self.sections:
            for node in section.nodes:
                if node.kind == "bullets":
                    total += len(node.lines)
                elif node.kind == "table" and node.table:
                    total += len(node.table.rows)
                elif node.kind == "text":
                    total += 1
        return total


def _flat(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _resume_name(title: str) -> str:
    """`周每每 · 个人简历` -> `周每每`；没有这层后缀就原样保留。"""
    cleaned = _flat(title)
    return TITLE_SUFFIX_RE.sub("", cleaned).strip() or cleaned


def _strip_number(title: str) -> str:
    return NUMBER_PREFIX_RE.sub("", _flat(title)).strip()


def _split_row(line: str) -> List[str]:
    match = TABLE_ROW_RE.match(line)
    return [_flat(cell) for cell in match.group("cells").split("|")]


def _is_divider(cells: Sequence[str]) -> bool:
    return bool(cells) and all(TABLE_DIVIDER_RE.match(cell or "---") for cell in cells)


def parse_resume(text: str, source_file: str = "") -> Resume:
    """把手写的简历 Markdown 读成结构化简历（只做整理，不改写）。"""
    resume = Resume(name="", source_file=source_file)
    lines = text.splitlines()
    current: Optional[Section] = None
    pending: List[str] = []
    index = 0

    def flush_text() -> None:
        nonlocal pending
        if pending and current is not None:
            current.nodes.append(Node(kind="text", text=" ".join(pending)))
        pending = []

    def section() -> Section:
        nonlocal current
        if current is None:
            current = Section(title="其他")
            resume.sections.append(current)
        return current

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        index += 1
        if not stripped or RULE_RE.match(line):
            flush_text()
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_text()
            level = len(heading.group("level"))
            title = _flat(heading.group("title"))
            if level == 1 and not resume.name:
                resume.name = _resume_name(title)
                continue
            if level == 2:
                current = Section(title=_strip_number(title))
                resume.sections.append(current)
                continue
            section().nodes.append(Node(kind="heading", level=3, text=title))
            continue

        table_row = TABLE_ROW_RE.match(line)
        if table_row:
            flush_text()
            block = [line]
            while index < len(lines) and TABLE_ROW_RE.match(lines[index]):
                block.append(lines[index].rstrip())
                index += 1
            rows = [row for row in (_split_row(item) for item in block) if not _is_divider(row)]
            if rows:
                section().nodes.append(
                    Node(kind="table", table=Table(header=rows[0], rows=rows[1:]))
                )
            continue

        quote = QUOTE_RE.match(line)
        if quote:
            flush_text()
            text_value = _flat(quote.group("text"))
            if not text_value:
                continue
            if current is None and not resume.sections:
                resume.tagline.append(text_value)
            else:
                section().nodes.append(Node(kind="quote", text=text_value))
            continue

        bullet = BULLET_RE.match(line)
        if bullet:
            flush_text()
            text_value = _flat(bullet.group("text"))
            last_node = current.nodes[-1] if current and current.nodes else None
            if last_node is not None and last_node.kind == "bullets":
                last_node.lines.append(text_value)
            else:
                section().nodes.append(Node(kind="bullets", lines=[text_value]))
            continue

        pending.append(_flat(line))

    flush_text()
    if not resume.name:
        resume.name = Path(source_file).stem if source_file else "个人简历"
    resume.sections = order_sections(resume.sections, resume)
    return resume


def order_sections(sections: Sequence[Section], resume: Optional[Resume] = None) -> List[Section]:
    """按模板顺序重排章节；模板外的章节保持原文相对顺序，排在后面。"""
    rank = {title: index for index, title in enumerate(SECTION_ORDER)}
    indexed = list(enumerate(sections))
    ordered = sorted(indexed, key=lambda pair: (rank.get(pair[1].title, len(rank)), pair[0]))
    result = [item for _, item in ordered]
    titles = [item.title for item in sections]
    if resume is not None and titles != [item.title for item in result]:
        resume.notes.append("章节顺序已按模板重排：" + " → ".join(item.title for item in result))
    return result


def merge_project_text(resume: Resume, text: str, source_file: str) -> int:
    """把一份项目描述并入「项目经历」章节，返回新增的要点条数。"""
    project = parse_project_text(text, source_file=source_file)
    section = next((item for item in resume.sections if item.title == "项目经历"), None)
    if section is None:
        section = Section(title="项目经历")
        resume.sections.append(section)
    bullets = [fact.text for fact in project.facts]
    section.nodes.append(Node(kind="heading", level=3, text=project.title))
    if bullets:
        section.nodes.append(Node(kind="bullets", lines=bullets))
    resume.notes.append(f"并入项目描述：{source_file}（新增 {len(bullets)} 条要点）")
    return len(bullets)


def build_resume(path, projects: Sequence = (), style: str = DEFAULT_STYLE) -> Resume:
    """读一份简历 md；projects 里的项目描述会被并进「项目经历」。

    style 只记在结果上（不问对错），校验交给 ResumeTool —— 工具出问题要返回 ok=False，不抛异常。
    """
    resume = parse_resume(read_text(path), source_file=display_path(path))
    resume.style = style
    for project_path in projects:
        source = display_path(project_path)
        merge_project_text(resume, read_text(project_path), source)
    return resume


# ---- Markdown 输出 ----------------------------------------------------------

RESUME_FORMATS = ("md", "html")
RESUME_STEM = "resume"


def default_stem(style: str = DEFAULT_STYLE) -> str:
    """默认文件名：经典简约还是 resume，其余风格带后缀，免得来回换风格互相覆盖。"""
    return RESUME_STEM if style == DEFAULT_STYLE else f"{RESUME_STEM}-{style}"


def _pad(row: Sequence[str], width: int) -> List[str]:
    cells = [_flat(cell).replace("|", "｜") for cell in row]
    return cells + [""] * max(0, width - len(cells))


def _table_markdown(table: Table) -> List[str]:
    width = table.width or 1
    lines = [
        "| " + " | ".join(_pad(table.header, width)) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(_pad(row, width)) + " |" for row in table.rows]
    return lines


def _node_markdown(node: Node, style: ResumeStyle) -> List[str]:
    if node.kind == "heading":
        if style.bold_sub_entries:
            return [f"**{node.text}**", ""]
        return [f"{'#' * max(3, node.level)} {node.text}", ""]
    if node.kind == "bullets":
        return [f"- {line}" for line in node.lines] + [""]
    if node.kind == "table" and node.table:
        return _table_markdown(node.table) + [""]
    if node.kind == "quote":
        return [f"> {node.text}", ""]
    return [node.text, ""]


def render_markdown(resume: Resume) -> str:
    """排好版的简历 Markdown：章节有序、列表与表格统一，可直接投递或再编辑。

    章节之间插不插 `---`、子条目用 `###` 还是加粗行，都由 resume.style 决定。
    """
    style = get_style(resume.style)
    lines: List[str] = [f"# {resume.name}", ""]
    if resume.tagline:
        lines += [f"> {item}" for item in resume.tagline] + [""]
    for index, section in enumerate(resume.sections):
        if index and style.rule_between_sections:
            lines += ["---", ""]
        lines += [f"## {section.title}", ""]
        for node in section.nodes:
            lines += _node_markdown(node, style)
    cleaned: List[str] = []
    for line in lines:
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).rstrip() + "\n"


# ---- HTML 输出 --------------------------------------------------------------

# 样式全部来自 resume_styles（PAGE_CSS + style.css），并且都限定在 .resume-doc 里：
# 整份 HTML 既能单独打开，也能直接嵌进宿主页面预览，不会污染宿主。
def _escape(text) -> str:
    return html_lib.escape(str(text))


def _inline(text: str) -> str:
    """极小的行内语法：**加粗** 与 `代码`；先转义再替换，不引入外链。"""
    out = _escape(text)
    out = re.sub(r"\*\*(?P<inner>[^*]+)\*\*", r"<strong>\g<inner></strong>", out)
    out = re.sub(r"`(?P<inner>[^`]+)`", r"<code>\g<inner></code>", out)
    return out


def _table_html(table: Table) -> str:
    parts = ["<table>"]
    if table.header:
        parts.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in table.header) + "</tr></thead>")
    parts.append("<tbody>")
    for row in table.rows:
        parts.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in _pad(row, table.width)) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _node_html(node: Node) -> str:
    if node.kind == "heading":
        return f"<h3>{_inline(node.text)}</h3>"
    if node.kind == "bullets":
        items = "".join(f"<li>{_inline(line)}</li>" for line in node.lines)
        return f"<ul>{items}</ul>"
    if node.kind == "table" and node.table:
        return _table_html(node.table)
    if node.kind == "quote":
        return f"<blockquote>{_inline(node.text)}</blockquote>"
    return f"<p>{_inline(node.text)}</p>"


def render_html(resume: Resume, notes: bool = True, standalone: bool = True) -> str:
    """自包含的简历 HTML：内嵌 CSS、A4 宽度、无外链，可直接打印。

    standalone=False 时不写 body 级样式，方便直接嵌进宿主页面预览（样式都限定在 .resume-doc 里）。
    具体样式取自 resume.style 对应的那套风格。
    """
    style = get_style(resume.style)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(resume.name)} · 简历</title>",
        f"<style>{(PAGE_CSS if standalone else '') + style.css}</style>",
        "</head>",
        "<body>",
        '<div class="resume-doc">',
        "<header>",
        f"<h1>{_escape(resume.name)}</h1>",
    ]
    if resume.tagline:
        spans = "".join(f"<span>{_inline(item)}</span>" for item in resume.tagline)
        parts.append(f'<p class="tagline">{spans}</p>')
    parts.append("</header>")
    for section in resume.sections:
        parts.append(f"<section><h2>{_inline(section.title)}</h2>")
        parts += [_node_html(node) for node in section.nodes]
        parts.append("</section>")
    if notes and resume.notes:
        items = "".join(f"<li>{_inline(note)}</li>" for note in resume.notes)
        parts.append(f'<div class="notes"><ul>{items}</ul></div>')
    parts += ["</div>", "</body>", "</html>"]
    return "\n".join(parts)


RESUME_RENDERERS = {"md": render_markdown, "html": render_html}


def write_resume_files(
    resume: Resume, out_dir, formats: Sequence[str] = RESUME_FORMATS, stem: str = "resume"
) -> List[Path]:
    """把简历写到 out_dir，返回写出的文件路径（顺序与 formats 一致）。"""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name in formats:
        renderer = RESUME_RENDERERS[name]
        path = directory / f"{stem}.{name}"
        path.write_text(renderer(resume), encoding="utf-8")
        written.append(path)
    return written


def resume_summary(resume: Resume) -> List[Tuple[str, str]]:
    """给终端与前端的摘要行（标签, 值）。"""
    return [
        ("姓名", resume.name),
        ("来源", resume.source_file),
        ("章节", "、".join(resume.section_titles) or "（没有识别到章节）"),
        ("内容量", f"{resume.item_count} 条要点/段落"),
    ]


# ---- 工具入口 ---------------------------------------------------------------


class ResumeTool(Tool):
    """简历排版：解析 → 校验 → 落盘。留给调用方的是一份 ToolResult，不是异常。"""

    slug = "resume"
    title = "简历排版"
    summary = "把简历 md 重排成简约大方的 md / html（三种风格，纯规则、不联网）"
    usage = (
        "python main.py --build-resume [--resume-style classic|compact|accent] "
        "[--resume-file input/profile/我的简历.md] [--resume-project input/project/示例项目1-AI周报助手.md] "
        "[--resume-format md,html] [--resume-name 我的简历] [--out output]"
    )

    def run(
        self,
        resume_file,
        out_dir,
        formats: Sequence[str] = RESUME_FORMATS,
        stem: Optional[str] = None,
        style: str = DEFAULT_STYLE,
        projects: Sequence = (),
    ) -> ToolResult:
        """读简历 md，写出排好版的文件。输入有问题就返回 ok=False —— 不抛异常、不打印。"""
        try:
            chosen = get_style(style)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))

        picked = [str(item).strip().lower() for item in formats if str(item).strip()]
        unknown = [item for item in picked if item not in RESUME_FORMATS]
        if unknown:
            return ToolResult(
                ok=False, error=f"简历不支持的输出格式：{'、'.join(unknown)}（可选 md / html）"
            )
        if not picked:
            return ToolResult(ok=False, error="至少要选一种简历输出格式：md / html")

        source = Path(resume_file).expanduser()
        if not source.is_file():
            return ToolResult(ok=False, error=f"没有找到简历文件：{source}")

        merge_paths = [Path(item).expanduser() for item in projects]
        missing = [str(item) for item in merge_paths if not item.is_file()]
        if missing:
            return ToolResult(ok=False, error=f"找不到要并入的项目描述文件：{'、'.join(missing)}")

        resume = build_resume(source, merge_paths, style=chosen.key)
        written = write_resume_files(resume, out_dir, picked, stem=stem or default_stem(chosen.key))

        # 风格只放摘要里，不塞进 notes —— notes 说的是「这篇简历被怎么规整过」
        return ToolResult(
            ok=True,
            summary=list(resume_summary(resume)) + [("排版风格", chosen.label)],
            notes=list(resume.notes),
            written=written,
        )


RESUME_TOOL = ResumeTool()
