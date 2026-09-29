"""简历结构化 Tool：把简历原文拆成「章节 + 事实条目」（纯规则、不联网）。

Resume Agent 主链路与命令行共用同一份实现：
    * `structure_resume_text(text, ...)` 是纯函数，Agent 拆事实时直接调它；
    * `ResumeStructTool` 是薄壳：解析 + 校验 + 落盘，返回 ToolResult（输入有问题不抛异常）。

这一层只管**结构**：章节怎么切、抬头信息怎么认、一条事实的原文与行号。
「这条算不算证据」（证据等级判定，resume-evidence）与「写了但没做过」（否定/背景识别，
resume-negation）分别在另外两个 Tool 里，三个 Tool 合起来才是完整口径。

工具区三条底线在这份实现里同样成立：不联网、不读 .env、不调用大模型。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..domain.jd import display_path, read_text
from ..domain.resume_facts import ROLE_LABEL, ResumeDoc, extract_resume
from .base import Tool, ToolResult

STEM = "resume_facts"


def structure_resume_text(text: str, source_file: str = "", title: str = "") -> ResumeDoc:
    """简历原文 -> 结构化结果（章节 + 抬头 + 事实条目）。"""
    return extract_resume(text, source_file=source_file, title=title)


def load_resume_doc(path, title: str = "") -> ResumeDoc:
    """从文件读一份简历并结构化（md / txt / html 请先用 jd-file Tool 转成 Markdown）。"""
    candidate = Path(path)
    return extract_resume(read_text(candidate), source_file=display_path(candidate), title=title)


def struct_summary(doc: ResumeDoc) -> List[Tuple[str, str]]:
    """给终端与前端的摘要行（标签, 值）。"""
    sections = "、".join(section.title for section in doc.sections) or "（没有识别到章节）"
    roles = "、".join(
        dict.fromkeys(ROLE_LABEL.get(section.role, section.role) for section in doc.sections)
    )
    return [
        ("简历", doc.name),
        ("来源", doc.source_file or "（粘贴的文本）"),
        ("章节", f"{len(doc.sections)} 个：{sections}"),
        ("事实条目", f"{doc.bullet_count} 条（单列了 {len(doc.meta)} 项抬头信息）"),
        ("章节角色", roles or "—"),
    ]


def render_struct_markdown(doc: ResumeDoc, source_line: str = "", generated_at: str = "") -> str:
    """结构化简历：章节 + 条目 + 原文行号（不带等级，判等级是证据 Tool 的事）。"""
    lines = [f"# {doc.name} · 结构化简历", ""]
    note = ["由 Resume Agent 的简历结构化 Tool（resume-struct）生成"]
    if source_line:
        note.append(f"来源：{source_line}")
    if generated_at:
        note.append(f"生成时间：{generated_at}")
    note.append("章节与条目按原文，一字未改")
    lines += ["> " + "｜".join(note), ""]

    for key, value in doc.meta:
        lines.append(f"**{key}**：{value}")
    if doc.meta:
        lines.append("")

    for section in doc.sections:
        if not section.facts and not section.background:
            continue
        lines.append(f"## {section.title}")
        if section.background:
            lines.append("")
            lines.append("> 整节是背景 / 简介：这里的条目按「背景」处理，不算「我做过什么」。")
        subsection = ""
        for fact in section.facts:
            if fact.subsection and fact.subsection != subsection:
                subsection = fact.subsection
                lines += ["", f"### {subsection}"]
            lines.append(f"- {fact.text}（L{fact.line_no}）")
        lines.append("")

    for note_line in doc.notes:
        lines.append(f"> 备注：{note_line}")
    return "\n".join(lines).rstrip() + "\n"


class ResumeStructTool(Tool):
    """简历结构化：原文 -> 章节 + 事实条目，可选落盘。"""

    slug = "resume-struct"
    title = "简历结构化"
    summary = "把简历拆成章节与事实条目（带原文行号，纯规则、不联网）"
    usage = (
        "python main.py --resume-agent --cv-file input/profile/xx.md "
        "[--resume-out output/resume_facts.md]"
    )

    def run(
        self,
        text: Optional[str] = None,
        path=None,
        source_file: str = "",
        title: str = "",
        out_dir=None,
        stem: Optional[str] = None,
    ) -> ToolResult:
        """拆一份简历；输入有问题返回 ok=False，不抛异常、不打印。"""
        if bool(str(text or "").strip()) == bool(path):
            return ToolResult(ok=False, error="简历结构化要且只要给一个来源：text= 或 path=")
        try:
            doc = (
                load_resume_doc(path, title=title)
                if path
                else structure_resume_text(str(text), source_file=source_file or "文本", title=title)
            )
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, error=f"简历读不出来：{exc}")
        if not doc.facts:
            return ToolResult(ok=False, error="这份简历里没有读到任何事实条目")

        notes = list(doc.notes)
        if not doc.meta:
            notes.append("没有读到抬头信息（姓名 / 求职意向这类），只做了章节与条目")

        written: List[Path] = []
        if out_dir is not None:
            directory = Path(out_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            name = stem or STEM
            target = directory / f"{name}.md"
            target.write_text(render_struct_markdown(doc), encoding="utf-8")
            written.append(target)
        return ToolResult(ok=True, summary=struct_summary(doc), notes=notes, written=written)


RESUME_STRUCT_TOOL = ResumeStructTool()
