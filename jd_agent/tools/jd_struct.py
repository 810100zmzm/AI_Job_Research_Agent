"""JD 结构化 Tool：把 JD 原文（粘贴的文本 / 图片转录 / 网页正文）规整成统一模板的 Markdown。

一次转换做三件事，全部是规则：
    1. 切岗位：按「岗位一 / 岗位二」标题切分（复用 jd.py 的 split_positions），没有就当一个岗位；
    2. 分章节：认「岗位职责 / 任职要求 / 加分项」等常见写法（复用 jd.py 的 SECTION_LEVEL），
       公司 / 地点这类信息收成 `**公司**：xx` 元信息行，认不出的章节原样保留、不乱归类；
    3. 出 Markdown：统一成「岗位标题 + 元信息 + 标准章节」模板，judge 主链路（main.py --jd）能直接吃。

保真三条：不翻译、不总结、不改字；丢掉的只有分隔线 / 引用行 / 空行 / 重复行，丢了多少行如实报出来；
章节标题认不出来时原样保留，而不是塞进「任职要求」。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..domain.jd import (
    BULLET_RE,
    HEADING_RE,
    JOB_HEADING_RE,
    META_KEYS,
    META_RE,
    RULE_RE,
    SECTION_LEVEL,
    SECTION_RE,
    clean_text,
    norm_section,
    split_positions,
)
from ..domain.lexicon import CAPABILITY_BY_KEY, match_capabilities
from ..core.schema import LEVEL_DUTY, LEVEL_MUST, LEVEL_PLUS
from .base import Tool, ToolResult
from .jd_files import parse_file_markdown

BUCKET_INTRO = "intro"
BUCKET_BY_LEVEL = {LEVEL_DUTY: "duties", LEVEL_MUST: "requirements", LEVEL_PLUS: "plus"}
BUCKET_TITLE = {
    BUCKET_INTRO: "岗位描述",
    "duties": "岗位职责",
    "requirements": "任职要求",
    "plus": "加分项",
}
BUCKET_ORDER = (BUCKET_INTRO, "duties", "requirements", "plus")
CN_NUM = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")

# 「认不出的章节标题」也要像标题才收：`# 关于我们` / `**团队介绍**` / `团队介绍：`
HEADING_LINE_RE = re.compile(r"^\*{0,2}[^：:*#]{1,16}\*{0,2}\s*[:：]\s*$")
UNSAFE_BUCKETS = ("duties", "requirements", "plus")


@dataclass
class JDPosition:
    """结构化后的一个岗位。"""

    title: str = ""
    explicit: bool = False                 # 原文里有「岗位N：」这类标题
    meta: List[Tuple[str, str]] = field(default_factory=list)
    intro: List[str] = field(default_factory=list)
    duties: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    plus: List[str] = field(default_factory=list)
    extras: List[Tuple[str, List[str]]] = field(default_factory=list)

    def bucket(self, name: str) -> List[str]:
        return {
            BUCKET_INTRO: self.intro,
            "duties": self.duties,
            "requirements": self.requirements,
            "plus": self.plus,
        }[name]

    def sections(self) -> List[Tuple[str, List[str]]]:
        """按模板顺序给出非空章节：(章节名, 条目)。"""
        result = [(BUCKET_TITLE[name], self.bucket(name)) for name in BUCKET_ORDER if self.bucket(name)]
        result += [(title, items) for title, items in self.extras if items]
        return result

    @property
    def bullet_count(self) -> int:
        return sum(len(items) for _, items in self.sections())

    @property
    def requirement_count(self) -> int:
        return len(self.requirements)


@dataclass
class StructuredJD:
    """一份结构化好的 JD（也是 Markdown 渲染与摘要的数据来源）。"""

    title: str
    source_label: str = ""
    positions: List[JDPosition] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    dropped: int = 0
    capabilities: List[Tuple[str, int]] = field(default_factory=list)   # (能力名, 命中条数)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def bullet_count(self) -> int:
        return sum(position.bullet_count for position in self.positions)

    @property
    def requirement_count(self) -> int:
        return sum(position.requirement_count for position in self.positions)

    @property
    def titles(self) -> List[str]:
        return [position.title for position in self.positions if position.title]


def _is_heading_line(line: str) -> bool:
    """这行像不像一个章节标题（用于「认不出的章节原样保留」这一步）。"""
    stripped = str(line or "").strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.startswith("**") and stripped.endswith("**"):
        return True
    return bool(HEADING_LINE_RE.match(stripped))


def _open_extra(position: JDPosition, heading: str) -> List[str]:
    """认不出的章节：原样保留成一个小节，返回它的条目列表（后面的内容写进这里）。"""
    entry: Tuple[str, List[str]] = (heading, [])
    position.extras.append(entry)
    return entry[1]


def _parse_position(title: str, body: Sequence[Tuple[int, str]]) -> Tuple[JDPosition, int, Dict[str, int]]:
    """把一段正文拆进各个章节；返回 (岗位, 丢掉的行数, 命中的能力计数)。"""
    position = JDPosition(
        title=clean_text(title),
        explicit=bool(JOB_HEADING_RE.match(f"## {title}")),
    )
    current: List[str] = position.intro
    dropped = 0
    caps: Dict[str, int] = {}
    seen: Set[str] = set()

    for _line_no, raw in body:
        line = str(raw).rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if RULE_RE.match(line) or stripped.startswith(">") or stripped.startswith("```"):
            dropped += 1
            continue

        section = SECTION_RE.match(line)
        if section:
            name = norm_section(section.group("name"))
            level = SECTION_LEVEL.get(name)
            if level:
                current = position.bucket(BUCKET_BY_LEVEL[level])
                continue
            if _is_heading_line(line):
                heading = clean_text(section.group("name")) or stripped
                current = _open_extra(position, heading)
                continue
        if stripped.startswith("#"):        # 剩下的标题行（文件 H1 / 长章节名）原样保留，别当成正文条目
            heading = clean_text(stripped.lstrip("#").strip())
            if heading:
                current = _open_extra(position, heading)
                continue

        text = clean_text(stripped)
        meta = META_RE.match(text) if text else None
        if meta and norm_section(meta.group("key")) in META_KEYS:
            key = clean_text(meta.group("key"))
            value = clean_text(meta.group("value"))
            if any(key == existing for existing, _ in position.meta) or not key:
                dropped += 1
            else:
                position.meta.append((key, value))
            continue

        bullet = BULLET_RE.match(line)
        content = clean_text(bullet.group("text") if bullet else line)
        if not content or content in seen:
            dropped += 1
            continue
        seen.add(content)
        current.append(content)
        for key in match_capabilities(content):
            caps[key] = caps.get(key, 0) + 1
    return position, dropped, caps


MAX_TITLE_CHARS = 30


def _usable_title(text: str) -> str:
    """章节名当不了标题：「岗位职责 / 任职要求 / 加分项」这类一律让位。"""
    candidate = clean_text(text)
    if not candidate or norm_section(candidate) in SECTION_LEVEL:
        return ""
    return candidate


def _first_doc_heading(lines: Sequence[str]) -> str:
    """原文里第一个「文档级」标题：`## 岗位一：xxx` 与「任职要求」这类章节名都不算。"""
    for raw in lines:
        match = HEADING_RE.match(str(raw))
        if not match or JOB_HEADING_RE.match(str(raw)):
            continue
        candidate = _usable_title(match.group("title"))
        if candidate:
            return candidate
    return ""


def _first_line_title(lines: Sequence[str]) -> Tuple[str, int]:
    """原文第一行够不够格当标题：够就返回 (标题, 行号)，不够返回 ("", -1)。"""
    for index, raw in enumerate(lines):
        stripped = str(raw).strip()
        if not stripped or RULE_RE.match(str(raw)) or stripped.startswith((">", "|")):
            continue
        if stripped.startswith("#"):        # 这一行是标题：交给「第一个文档级标题」那条路径
            return "", -1
        candidate = _usable_title(stripped)
        if (
            not candidate
            or BULLET_RE.match(str(raw))
            or len(candidate) > MAX_TITLE_CHARS
            or "：" in candidate
        ):
            return "", -1
        return candidate, index
    return "", -1


def _split_title(lines: Sequence[str], title: str) -> Tuple[str, List[str], str]:
    """定标题：调用方给的 > 原文第一行（够短、且不像章节 / 要求）> 原文第一个标题。

    返回 (标题, 去掉标题行之后的原文, 标题是从哪来的)。「取自第一行」这一步会把这行从正文里拿走，
    并且会写进 notes —— 有改动就要看得见。第一行不是那种干净短行时不做任何猜测，标题留空。
    """
    given = str(title or "").strip()
    if given:
        return given, list(lines), "调用方指定"
    candidate, index = _first_line_title(lines)
    if candidate:
        remaining = list(lines[:index]) + list(lines[index + 1 :])
        return candidate, remaining, "原文第一行"
    heading = _first_doc_heading(lines)
    if heading:
        return heading, list(lines), "原文标题"
    return "", list(lines), ""


def _fallback_title(positions: Sequence[JDPosition]) -> str:
    """没有可用标题时的兜底：单岗位用岗位名，多岗位说清有几个岗位。"""
    if len(positions) == 1:
        return _usable_title(positions[0].title) or "未命名岗位"
    return f"{len(positions)} 个岗位"


def structure_jd_text(text: str, source_label: str = "", title: str = "") -> StructuredJD:
    """把 JD 原文结构化；原文再乱也会返回一份结果，不做任何「猜测补全」。"""
    lines = str(text or "").splitlines()
    resolved, body_lines, title_from = _split_title(lines, title)
    blocks = split_positions(body_lines)
    positions: List[JDPosition] = []
    notes: List[str] = []
    dropped = 0
    caps: Dict[str, int] = {}

    for block_title, body in blocks:
        position, block_dropped, block_caps = _parse_position(block_title, body)
        dropped += block_dropped
        for key, count in block_caps.items():
            caps[key] = caps.get(key, 0) + count
        positions.append(position)
    if not positions:
        positions.append(JDPosition())

    if not resolved:
        resolved = _fallback_title(positions)
    if not _usable_title(positions[0].title):     # 纯章节名当不了标题，换成文档标题
        positions[0].title = resolved
    if title_from == "原文第一行":
        notes.append(f"标题取自原文第一行并已从正文移出：{resolved}")
    if len(positions) > 1:
        notes.append(f"识别到 {len(positions)} 个岗位（按原文的「岗位N / 标题」切分）")
    if not any(position.bucket(name) for position in positions for name in UNSAFE_BUCKETS):
        notes.append("原文没有分章节标题：内容都归到「岗位描述」，下游按岗位职责一档处理")
    if any(position.extras for position in positions):
        names = "、".join(title for position in positions for title, _ in position.extras)
        notes.append(f"认不出的章节原样保留：{names}")
    if dropped:
        notes.append(f"丢掉 {dropped} 行空行 / 分隔线 / 引用行 / 重复行（字一个没改）")

    ordered = sorted(
        ((CAPABILITY_BY_KEY[key].name, count) for key, count in caps.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return StructuredJD(
        title=resolved,
        source_label=str(source_label or ""),
        positions=positions,
        notes=notes,
        dropped=dropped,
        capabilities=ordered,
    )


def render_jd_markdown(structured: StructuredJD, source_line: str = "", generated_at: str = "") -> str:
    """把结构化结果渲染成统一模板的 JD Markdown（judge 主链路能直接解析这一份）。"""
    stamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    header = "> 由 JD Agent（LangGraph）转换"
    if source_line:
        header += f"｜来源：{source_line}"
    header += f"｜生成时间：{stamp}｜章节由规则识别，一字未改"

    blocks: List[List[str]] = [[f"# {structured.title or '未命名岗位'}", header]]
    multiple = len(structured.positions) > 1
    for index, position in enumerate(structured.positions, start=1):
        body: List[str] = []
        if multiple or position.explicit:
            name = position.title or f"岗位{CN_NUM[min(index, len(CN_NUM)) - 1]}"
            body += [f"## {name}", ""]
        for key, value in position.meta:
            body.append(f"**{key}**：{value}".rstrip("："))
        if position.meta:
            body.append("")
        for section_title, items in position.sections():
            body.append(f"**{section_title}**")
            body += [f"- {item}" for item in items]
            body.append("")
        blocks.append(body)
    return "\n\n".join("\n".join(block).strip() for block in blocks).strip() + "\n"


def jd_summary(structured: StructuredJD) -> List[Tuple[str, str]]:
    """给终端与前端的摘要行（标签, 值）。"""
    return [
        ("标题", structured.title),
        ("岗位", "、".join(structured.titles) or "（没有识别到岗位标题）"),
        (
            "要点",
            f"{structured.bullet_count} 条（其中任职要求 {structured.requirement_count} 条）",
        ),
        (
            "命中能力",
            "、".join(f"{name}×{count}" for name, count in structured.capabilities[:6])
            or "（没命中能力词典里的项）",
        ),
        ("清理", f"丢掉 {structured.dropped} 行空行 / 分隔线 / 重复行"),
    ]


class JDStructTool(Tool):
    """JD 结构化：原文（text 或 path）-> 统一模板的 JD Markdown，可选落盘。"""

    slug = "jd-struct"
    title = "JD 结构化"
    summary = "把 JD 原文规整成统一模板的 Markdown（岗位标题 + 元信息 + 岗位职责/任职要求/加分项，纯规则）"
    usage = "python main.py --jd-agent --jd-file input/jd/xx.md --jd-out output/jd.md"

    def run(
        self,
        text: str = "",
        path=None,
        out_dir=None,
        stem: Optional[str] = None,
        title: str = "",
        source_label: str = "",
    ) -> ToolResult:
        """原文可以给 text，也可以给 path（走文件解析 Tool）；有问题返回 ok=False。"""
        if path is not None and not str(text or "").strip():
            try:
                parsed = parse_file_markdown(path)
            except ValueError as exc:
                return ToolResult(ok=False, error=str(exc))
            text, source_label = parsed.markdown, source_label or parsed.path
        if not str(text or "").strip():
            return ToolResult(ok=False, error="没有 JD 原文：请给 path（文件）或 text（文本）")

        structured = structure_jd_text(text, source_label=source_label, title=title)
        markdown = render_jd_markdown(structured, source_line=source_label)
        written: List[Path] = []
        if out_dir is not None:
            directory = Path(out_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{stem or 'jd'}.md"
            target.write_text(markdown, encoding="utf-8")
            written.append(target)
        return ToolResult(
            ok=True,
            summary=jd_summary(structured),
            notes=list(structured.notes),
            written=written,
        )


JD_STRUCT_TOOL = JDStructTool()
