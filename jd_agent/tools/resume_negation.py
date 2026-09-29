"""否定 / 背景识别 Tool：挡住「简历里写了，但其实是没做过 / 只是背景」的内容。

**这个 Tool 存在的唯一理由**：不让「简历里出现过」被当成「已具备的证据」。

它挡两种东西（判定规则全部来自 resume_facts，原文照抄在下面，可直接核对）：

    否定 negated —— 「未接触过向量数据库」「没做过 Docker 部署」「尚未有互联网公司实习经历」。
                    哪怕同一行里出现了「部署」这种动作词，也不算做过：它是反向证据。
    背景 background —— 「本项目旨在解决…」「计划学习 Go」「项目简介」这类：写的是课题、行业与打算，
                      不是「我做过什么」。

两条纪律：
  * **放宽口径也不放**：否认与背景不参与「证据强弱」的放宽 —— 「没做过」写多少遍都不会变成「做过」；
  * **挡住不等于删掉**：被挡的条目连同原文与行号一起列出来，你能一眼核对它挡得对不对。

工具区三条底线：不联网、不读 .env、不调用大模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..domain.resume_facts import (
    CAPABILITY_BY_KEY,
    ABSENCE_RE,
    BACKGROUND_SENTENCE_RE,
    COURSE_RE,
    FACT_BLOCK_BACKGROUND,
    FACT_BLOCK_LABEL,
    FACT_BLOCK_NEGATED,
    FACT_BLOCK_REASON_HINT,
    PLAN_RE,
    STRONG_NEGATION_RE,
    ResumeDoc,
    ResumeFact,
    background_hit,
    extract_resume,
    negated_capability_keys,
    negation_hit,
    plan_hit,
    table_lines,
)
from .base import Tool, ToolResult
from .resume_struct import load_resume_doc

STEM = "resume_negation"

# 判定规则原文：工具自解释，命令行与前端都能直接展示
RULE_LINES = (
    f"否定（{FACT_BLOCK_LABEL[FACT_BLOCK_NEGATED]}）：整行命中 {STRONG_NEGATION_RE.pattern}，"
    f"或命中 {ABSENCE_RE.pattern}，或能力关键词前后 24 字内出现否定词（未 / 没 / 无 / 尚未 …）"
    f" → {FACT_BLOCK_REASON_HINT[FACT_BLOCK_NEGATED]}；",
    f"背景（{FACT_BLOCK_LABEL[FACT_BLOCK_BACKGROUND]}）：整行命中 {BACKGROUND_SENTENCE_RE.pattern}，"
    f"或命中 {PLAN_RE.pattern}（「计划学习 X」只是想学），"
    "或所在章节名命中「背景 / 简介 / 概述 / 痛点 / 项目介绍」 → "
    f"{FACT_BLOCK_REASON_HINT[FACT_BLOCK_BACKGROUND]}；",
    f"课程（不打标签但会降级）：命中 {COURSE_RE.pattern} 且这一行没有任何指标"
    " → 等级封顶「仅提及」（上过课不等于做过事）。",
)


@dataclass
class ScreenReport:
    """一次否定 / 背景识别的结果。"""

    facts: List[ResumeFact] = field(default_factory=list)

    @property
    def negated(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.blocked == FACT_BLOCK_NEGATED]

    @property
    def background(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.blocked == FACT_BLOCK_BACKGROUND]

    @property
    def blocked(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.blocked]

    @property
    def kept(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.usable]

    def facts_of(self, kind: str) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.blocked == kind]

    @property
    def rules(self) -> Tuple[str, ...]:
        return RULE_LINES

    @property
    def headline(self) -> str:
        if not self.facts:
            return "没有可判定的条目。"
        if not self.blocked:
            return f"{len(self.facts)} 条事实里没有发现否定或背景写法。"
        return (
            f"{len(self.facts)} 条事实里有 {len(self.blocked)} 条被挡住"
            f"（否定 {len(self.negated)} / 背景 {len(self.background)}），"
            "它们不能算「已具备的证据」。"
        )


def screen_facts(facts: Sequence[ResumeFact]) -> ScreenReport:
    """把事实条目过一遍否定 / 背景识别（判定在 resume_facts.make_fact 里已经做完了）。"""
    return ScreenReport(facts=list(facts))


def explain_line(text: str) -> Tuple[str, str]:
    """单看一句话：返回 (挡住的类型, 原因)；没挡住就是 ("", "")。"""
    negated_keys, marker = negated_capability_keys(text)
    whole = negation_hit(text)
    if whole or negated_keys:
        names = "、".join(
            CAPABILITY_BY_KEY[key].name for key in negated_keys if key in CAPABILITY_BY_KEY
        )
        reason = f"原文写了「{whole or marker}」：这是反向证据，不能当「已具备」"
        if names:
            reason += f"；被否定的能力：{names}"
        return FACT_BLOCK_NEGATED, reason
    background = background_hit(text) or plan_hit(text)
    if background:
        return FACT_BLOCK_BACKGROUND, f"写的是背景 / 目标 / 打算（{background}），不是个人动作"
    return "", ""


def negation_summary(report: ScreenReport, source_label: str = "") -> List[Tuple[str, str]]:
    """给终端与前端的摘要行（标签, 值）。"""
    rows = [
        ("来源", source_label or "（粘贴的文本）"),
        ("事实条目", f"{len(report.facts)} 条"),
        ("挡住", f"{len(report.blocked)} 条（否定 {len(report.negated)} / 背景 {len(report.background)}）"),
    ]
    if report.negated:
        rows.append(("否定的原文", "；".join(fact.quote_short for fact in report.negated[:3])))
    if report.background:
        rows.append(("背景的原文", "；".join(fact.quote_short for fact in report.background[:3])))
    return rows


def render_blocked_table(facts: Sequence[ResumeFact]) -> List[str]:
    """被挡掉的条目表：原文 + 判定 + 为什么不算 + 出处。"""
    if not facts:
        return ["（没有）"]
    rows = [
        [fact.index, fact.text, fact.level_label, fact.block_reason, fact.ref] for fact in facts
    ]
    return table_lines(("#", "原文", "判定", "为什么不算证据", "出处"), rows)


def render_negation_markdown(
    doc: ResumeDoc, report: ScreenReport, source_line: str = "", generated_at: str = ""
) -> str:
    """否定 / 背景识别报告（--resume-agent 的输出里也会带上这一段）。"""
    lines = [f"# {doc.name} · 否定与背景识别", ""]
    note = ["由 Resume Agent 的否定 / 背景识别 Tool（resume-negation）生成"]
    if source_line:
        note.append(f"来源：{source_line}")
    if generated_at:
        note.append(f"生成时间：{generated_at}")
    note.append("作用：挡住「简历里写了，但其实是没做过 / 只是背景」的内容，不让它被当成「已具备的证据」")
    lines += ["> " + "｜".join(note), ""]
    lines += [f"**结论**：{report.headline}", ""]

    for kind in (FACT_BLOCK_NEGATED, FACT_BLOCK_BACKGROUND):
        group = report.facts_of(kind)
        lines.append(f"## {FACT_BLOCK_LABEL[kind]}：{len(group)} 条")
        lines.append("")
        lines.append(f"> {FACT_BLOCK_REASON_HINT[kind]}。")
        lines += render_blocked_table(group)
        lines.append("")

    lines += ["## 判定规则（原文照抄，可直接核对）", ""]
    lines += [f"- {rule}" for rule in RULE_LINES]
    return "\n".join(lines).rstrip() + "\n"


class ResumeNegationTool(Tool):
    """否定 / 背景识别：把「写了但没做过」的条目挑出来，可选落盘。"""

    slug = "resume-negation"
    title = "否定 / 背景识别"
    summary = "挑出简历里「写了但没做过 / 只是背景」的条目，不许它们当证据（纯规则、不联网）"
    usage = "python main.py --resume-agent --cv-file input/profile/xx.md --resume-trace"

    def run(
        self,
        text: Optional[str] = None,
        path=None,
        source_file: str = "",
        title: str = "",
        out_dir=None,
        stem: Optional[str] = None,
    ) -> ToolResult:
        """识别一份简历里的否定与背景；输入有问题返回 ok=False，不抛异常、不打印。"""
        if bool(str(text or "").strip()) == bool(path):
            return ToolResult(ok=False, error="否定 / 背景识别要且只要给一个来源：text= 或 path=")
        try:
            doc = (
                load_resume_doc(path, title=title)
                if path
                else extract_resume(str(text or ""), source_file=source_file or "文本", title=title)
            )
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, error=f"简历读不出来：{exc}")

        report = screen_facts(doc.facts)
        notes = []
        if not report.blocked:
            notes.append("没有发现「写了但没做过」的写法：这份简历里的条目都能拿去判等级")
        else:
            notes.append("被挡掉的条目只影响「能不能当证据」，原文一律保留在报告里")

        written: List[Path] = []
        if out_dir is not None:
            directory = Path(out_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{stem or STEM}.md"
            target.write_text(render_negation_markdown(doc, report), encoding="utf-8")
            written.append(target)
        return ToolResult(
            ok=True,
            summary=negation_summary(report, source_label=source_file or doc.source_file),
            notes=notes,
            written=written,
        )
RESUME_NEGATION_TOOL = ResumeNegationTool()
