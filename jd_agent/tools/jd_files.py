"""文件解析 Tool：把本地 JD 文件读成 Markdown 草稿（纯规则、不联网）。

JD Agent 主链路与命令行共用同一份实现：
    * `parse_file_markdown(path)` 是纯函数，Agent 取原文时直接调它；
    * `JDFileTool` 是薄壳：解析 + 校验 + 落盘，返回 ToolResult（输入有问题不抛异常）。

按后缀分派：
    .md/.markdown/.txt/.text  直接读取（去 BOM、统一换行、压掉空行与重复行）
    .html/.htm/.xhtml         HTML -> Markdown（保留标题 / 列表 / 段落 / 表格结构）
    .json                     招聘平台导出的 JSON 摊平成 Markdown
    .yaml/.yml                key: value 摊平成 Markdown（只做扁平化，不做完整 YAML 解析）
    图片后缀                  规则读不出像素内容：交给 Agent 的图片分支（Qwen-VL），工具区不联网

工具区三条底线在这份实现里同样成立：不联网、不读 .env、不调用大模型。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..domain.jd import display_path, read_text
from ..domain.jd_html import clean_lines, count_lines, markdown_from_html
from .base import Tool, ToolResult

TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".text")
HTML_SUFFIXES = (".html", ".htm", ".xhtml")
JSON_SUFFIXES = (".json",)
YAML_SUFFIXES = (".yaml", ".yml")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

KIND_TEXT = "text"
KIND_HTML = "html"
KIND_JSON = "json"
KIND_YAML = "yaml"
KIND_IMAGE = "image"

KIND_LABEL = {
    KIND_TEXT: "纯文本",
    KIND_HTML: "HTML 网页",
    KIND_JSON: "JSON 导出",
    KIND_YAML: "YAML 导出",
    KIND_IMAGE: "图片",
}

SCALAR_TYPES = (str, int, float, bool)


@dataclass
class ParsedFile:
    """一次文件解析的结果（Agent 与 Tool 都拿它说话）。"""

    path: str
    kind: str
    markdown: str
    dropped: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return count_lines(self.markdown)

    @property
    def char_count(self) -> int:
        return len(self.markdown)

    @property
    def empty(self) -> bool:
        return not self.markdown.strip()


def classify_file(path) -> str:
    """按后缀判断文件类型：text / html / json / yaml / image；认不出返回空串。"""
    suffix = Path(str(path)).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return KIND_TEXT
    if suffix in HTML_SUFFIXES:
        return KIND_HTML
    if suffix in JSON_SUFFIXES:
        return KIND_JSON
    if suffix in YAML_SUFFIXES:
        return KIND_YAML
    if suffix in IMAGE_SUFFIXES:
        return KIND_IMAGE
    return ""


def _flat(text: Any) -> str:
    return " ".join(str(text).split())


def json_to_markdown(payload: Any, level: int = 2) -> str:
    """把 JSON 摊平成 Markdown：标量写成 `- **键**：值`，嵌套结构升成标题。"""
    lines: List[str] = []

    def walk(node: Any, depth: int) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, SCALAR_TYPES) or value is None:
                    text = "" if value is None else _flat(value)
                    lines.append(f"- **{_flat(key)}**：{text}".rstrip("："))
                else:
                    lines.append(f"{'#' * max(1, min(6, depth))} {_flat(key)}")
                    walk(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node:
                if isinstance(item, SCALAR_TYPES) or item is None:
                    lines.append(f"- {_flat(item or '')}".rstrip())
                else:
                    walk(item, depth)
        else:
            lines.append(f"- {_flat(node)}".rstrip())

    walk(payload, level)
    return "\n".join(line for line in lines if line.strip())


YAML_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^:#\-][^:]*?)\s*:\s*(?P<value>.*)$")


def yaml_to_markdown(text: str) -> str:
    """把 YAML 摊平成 Markdown（只做扁平化：key: value 变条目，缩进变缩进）。"""
    lines: List[str] = []
    for raw in str(text or "").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = " " * (len(raw) - len(raw.lstrip()))
        stripped = raw.strip()
        if stripped.startswith("- "):
            lines.append(f"{indent}- {_flat(stripped[2:])}")
            continue
        match = YAML_LINE_RE.match(raw)
        if not match:
            lines.append(f"{indent}- {_flat(stripped)}")
            continue
        key = _flat(match.group("key"))
        value = _flat(match.group("value"))
        if value:
            lines.append(f"{indent}- **{key}**：{value}")
        else:
            lines.append(f"{indent}- **{key}**")
    return "\n".join(lines)


def parse_file_markdown(path, strict: bool = True) -> ParsedFile:
    """读一个本地 JD 文件，转成 Markdown 草稿；读不出来抛 ValueError（不返回半成品）。"""
    candidate = Path(str(path)).expanduser()
    if not candidate.is_file():
        raise ValueError(f"找不到文件：{candidate}")
    kind = classify_file(candidate)
    if not kind:
        suffix = candidate.suffix or "无后缀"
        raise ValueError(f"不认识的文件类型：{suffix}（支持 md / txt / html / json / yaml）")
    if kind == KIND_IMAGE:
        raise ValueError(
            f"{candidate.name} 是图片：规则工具读不出像素内容，请走图片分支（Qwen-VL），工具区不联网"
        )

    notes: List[str] = []
    raw = read_text(candidate)
    if kind == KIND_TEXT:
        markdown, dropped = clean_lines(raw, strict=strict)
    elif kind == KIND_HTML:
        markdown, dropped = markdown_from_html(raw, strict=strict)
        notes.append("HTML 只保留标题 / 列表 / 段落 / 表格结构，script 与样式一并丢弃")
    elif kind == KIND_JSON:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败（{exc.msg}，第 {exc.lineno} 行）") from exc
        markdown, dropped = clean_lines(json_to_markdown(payload), strict=False)
        notes.append("JSON 按字段摊平成条目，层级用标题表示")
    else:
        markdown, dropped = clean_lines(yaml_to_markdown(raw), strict=False)
        notes.append("YAML 只做扁平化（key: value -> 条目），不做类型转换")
    return ParsedFile(
        path=display_path(candidate), kind=kind, markdown=markdown, dropped=dropped, notes=notes
    )


def parsed_summary(parsed: ParsedFile, strict: bool = True) -> List[Tuple[str, str]]:
    """给终端与前端的摘要行（标签, 值）。"""
    scope = "严格口径（含 UI 噪声过滤）" if strict else "基础口径（只去空行 / 重复行）"
    return [
        ("文件", parsed.path),
        ("类型", KIND_LABEL.get(parsed.kind, parsed.kind)),
        ("有效行", f"{parsed.line_count} 行（{parsed.char_count} 字）"),
        ("清理", f"丢掉 {parsed.dropped} 行｜{scope}"),
    ]


class JDFileTool(Tool):
    """文件解析：读本地 JD 文件 -> Markdown 草稿，可选落盘。"""

    slug = "jd-file"
    title = "JD 文件解析"
    summary = "把本地 JD 文件（md/txt/html/json/yaml）读成 Markdown 草稿（纯规则、不联网）"
    usage = "python main.py --jd-agent --jd-file input/jd/xx.md [--jd-file xx.html] [--jd-out output/jd.md]"

    def run(
        self,
        path,
        out_dir=None,
        stem: Optional[str] = None,
        strict: bool = True,
    ) -> ToolResult:
        """读文件 -> Markdown；输入有问题返回 ok=False，不抛异常、不打印。"""
        try:
            parsed = parse_file_markdown(path, strict=strict)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        if parsed.empty:
            return ToolResult(ok=False, error=f"{parsed.path} 里没有读到任何内容")

        written: List[Path] = []
        if out_dir is not None:
            directory = Path(out_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            name = stem or Path(parsed.path).stem or "jd"
            target = directory / f"{name}.md"
            target.write_text(parsed.markdown, encoding="utf-8")
            written.append(target)
        return ToolResult(
            ok=True,
            summary=parsed_summary(parsed, strict=strict),
            notes=list(parsed.notes),
            written=written,
        )


JD_FILE_TOOL = JDFileTool()
