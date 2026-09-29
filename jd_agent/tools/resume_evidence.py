"""证据等级判定 Tool：给简历里的事实条目逐条定级（纯规则、不联网）。

三档等级，标记词与主线 project.py 逐字相同：

    有结果 result —— 量化指标（「准确率从 72% 提升到 85%」）/ 交付动作（已开源 / 已上线 / 已发布）
                     / 荣誉名次（获奖、奖学金、省级一等奖 —— 已经拿到的外部认定）
    有动作 action —— 有「搭建 / 实现 / 调试 / 优化」这类动手信号，但没写结果
    仅提及 mention —— 只写了「了解 / 熟悉 / 使用过」，或者只有技术栈罗列

两个口径（对应 Agent Loop 里的一次 Adjust）：

    严格 strict  —— 只认「有动作 / 有结果」的硬证据
    放宽 relaxed —— 允许把「仅提及」计为弱证据，避免把空白误判成「没做过」

**但否定与背景两条线不在放宽范围内**：被否定 / 背景识别挡掉的条目，两个口径下都不算证据 ——
「没做过」写多少遍都不会变成「做过」。

工具区三条底线：不联网、不读 .env、不调用大模型。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..domain.resume_facts import (
    EVIDENCE_ACTION,
    EVIDENCE_LABEL,
    EVIDENCE_MENTION,
    EVIDENCE_RESULT,
    EvidenceSummary,
    ResumeDoc,
    ResumeFact,
    extract_resume,
    summarize,
    table_lines,
)
from .base import Tool, ToolResult
from .resume_negation import render_blocked_table
from .resume_struct import load_resume_doc

STEM = "resume_evidence"

LEVEL_ORDER = (EVIDENCE_RESULT, EVIDENCE_ACTION, EVIDENCE_MENTION)


def summarize_evidence(facts: Sequence[ResumeFact], relaxed: bool = False) -> EvidenceSummary:
    """给事实条目逐条定级并汇总（判定本身在 resume_facts.make_fact 里已经做完）。"""
    return summarize(facts, relaxed=relaxed)


def render_evidence_table(facts: Sequence[ResumeFact]) -> List[str]:
    """能当证据的条目表：# / 事实 / 等级 / 命中能力 / 依据 / 出处。"""
    if not facts:
        return ["（没有）"]
    rows = [
        [
            fact.index,
            fact.text,
            fact.level_label,
            "、".join(fact.capability_names) or "—",
            fact.basis,
            fact.ref,
        ]
        for fact in facts
    ]
    return table_lines(("#", "事实", "等级", "命中能力", "依据", "出处"), rows)


def evidence_summary_rows(doc: ResumeDoc, summary: EvidenceSummary) -> List[Tuple[str, str]]:
    """给终端与前端的摘要行（标签, 值）。"""
    counts = summary.counts
    return [
        ("简历", doc.name),
        ("事实条目", f"{len(summary.facts)} 条"),
        (
            "可当证据",
            f"{len(summary.usable)} 条（硬证据 {len(summary.solid)} / 弱证据 {len(summary.weak)}）",
        ),
        (
            "等级分布",
            " / ".join(f"{EVIDENCE_LABEL[level]} {counts.get(level, 0)}" for level in LEVEL_ORDER),
        ),
        ("被挡住", f"{len(summary.blocked)} 条（否定 {len(summary.negated)} / 背景 {len(summary.background)}）"),
        (
            "本次计入",
            f"{len(summary.evidence)} 条（{'放宽' if summary.relaxed else '严格'}口径："
            + ("仅提及也算弱证据）" if summary.relaxed else "只认有动作 / 有结果）"),
        ),
    ]


def render_evidence_markdown(
    doc: ResumeDoc, summary: EvidenceSummary, source_line: str = "", generated_at: str = ""
) -> str:
    """证据等级清单：按原文章节列出每条事实的等级与依据，最后附被挡掉的条目。"""
    counts = summary.counts
    lines = [f"# {doc.name} · 证据等级清单", ""]
    note = ["由 Resume Agent 的证据等级判定 Tool（resume-evidence）生成"]
    if source_line:
        note.append(f"来源：{source_line}")
    if generated_at:
        note.append(f"生成时间：{generated_at}")
    note.append("事实与等级由规则判定，一字未改")
    lines += ["> " + "｜".join(note), ""]

    lines += [f"**结论**：{summary.conclusion}", ""]
    lines += [
        "**口径**："
        + ("放宽（「仅提及」按弱证据计入）" if summary.relaxed else "严格（只认「有动作 / 有结果」）")
        + "；否定与背景永远不算证据。",
        "",
    ]
    lines += [
        "**等级分布**："
        + "｜".join(f"{EVIDENCE_LABEL[level]} {counts.get(level, 0)}" for level in LEVEL_ORDER)
        + f"｜被挡 {len(summary.blocked)}",
        "",
    ]

    for section in doc.sections:
        if not section.facts:
            continue
        kept = [fact for fact in section.facts if fact.usable]
        lines.append(f"## {section.title}")
        lines.append("")
        if section.background:
            lines.append("> 整节是背景 / 简介：这里的条目都不算「我做过什么」。")
            lines.append("")
        lines += render_evidence_table(kept)
        lines.append("")

    lines += ["## 不能当证据的内容（挡住「写了但没做过」）", ""]
    lines += render_blocked_table(summary.blocked)
    return "\n".join(lines).rstrip() + "\n"


class ResumeEvidenceTool(Tool):
    """证据等级判定：逐条定级 + 汇总，可选落盘。"""

    slug = "resume-evidence"
    title = "证据等级判定"
    summary = "给简历事实条目逐条判「有结果 / 有动作 / 仅提及」，严格与放宽两个口径（纯规则、不联网）"
    usage = "python main.py --resume-agent --cv-file input/profile/xx.md --resume-out output/resume_facts.md"

    def run(
        self,
        text: Optional[str] = None,
        path=None,
        source_file: str = "",
        title: str = "",
        relaxed: bool = False,
        out_dir=None,
        stem: Optional[str] = None,
    ) -> ToolResult:
        """判一份简历的等级；输入有问题返回 ok=False，不抛异常、不打印。"""
        if bool(str(text or "").strip()) == bool(path):
            return ToolResult(ok=False, error="证据等级判定要且只要给一个来源：text= 或 path=")
        try:
            doc = (
                load_resume_doc(path, title=title)
                if path
                else extract_resume(str(text or ""), source_file=source_file or "文本", title=title)
            )
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, error=f"简历读不出来：{exc}")
        if not doc.facts:
            return ToolResult(ok=False, error="这份简历里没有读到任何事实条目")

        summary = summarize_evidence(doc.facts, relaxed=relaxed)
        notes = [
            "等级标记词与主线 project.py 一致：量化指标 / 交付动作 / 动手动作 / 归属词",
            "荣誉与名次（获奖、奖学金、名次）算「有结果」：已经拿到的外部认定",
        ]
        if summary.blocked:
            notes.append(
                f"{len(summary.blocked)} 条被否定 / 背景识别挡住，两个口径下都不算证据"
            )

        written: List[Path] = []
        if out_dir is not None:
            directory = Path(out_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{stem or STEM}.md"
            target.write_text(render_evidence_markdown(doc, summary), encoding="utf-8")
            written.append(target)
        return ToolResult(
            ok=True,
            summary=evidence_summary_rows(doc, summary),
            notes=notes,
            written=written,
        )


RESUME_EVIDENCE_TOOL = ResumeEvidenceTool()
