"""渲染：把一次 Agent 运行的结果输出成 Markdown / JSON / HTML / 终端文本。

四种输出说同一件事：结论 -> 理由 / 风险 ->（可选）LLM 生成内容 -> 岗位要求对照表 -> 完整 Trace。
Trace 是这份产物里最该被检查的部分：每一步都能看到「看到了什么、状态怎么变、下一步为什么这么走」。

来源标记：规则算出来的段落一律标 [规则]，大模型生成的一律标 [LLM] 并带上 source
（llm-suggest / llm-interview / llm-polish）；不带 --llm 时 LLM 那一节整体不出现。
"""
from __future__ import annotations

import html as html_lib
import json
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from .agent import COVERAGE_MIN, COVERAGE_OK, MAX_ROUNDS, MIN_FACTS
from ..core.schema import (
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_NONE,
    EVIDENCE_RESULT,
    SOURCE_LABEL,
    SOURCE_RULE,
    VERDICT_EDIT,
    VERDICT_NO,
    VERDICT_YES,
    AgentState,
    ContextItem,
    AgentObservation,
    EvidenceMatch,
    ImageRef,
    JobPosting,
    LlmBlock,
    LlmReport,
    Project,
    ProjectFact,
    Reason,
    Requirement,
    RetrievedContext,
    Risk,
    TraceStep,
    Verdict,
    VisionReport,
)

LEVEL_ICON = {
    EVIDENCE_RESULT: "✅",
    EVIDENCE_ACTION: "🟠",
    EVIDENCE_MENTION: "🟡",
    EVIDENCE_NONE: "❌",
}
VERDICT_ICON = {VERDICT_YES: "✅", VERDICT_EDIT: "🟠", VERDICT_NO: "❌"}
CELL_LIMIT = 78
CN_NUM = ("一", "二", "三", "四", "五", "六")


def _has_llm(state: AgentState) -> bool:
    """这份报告里有没有 LLM 生成内容（不带 --llm 时整节都不出现）。"""
    report = state.llm_report
    return bool(report is not None and not report.is_empty)


def _sections(state: AgentState) -> Dict[str, str]:
    """章节编号：有 LLM 段时多一节，后面的编号自动顺延。"""
    keys = ["verdict"]
    if _has_llm(state):
        keys.append("llm")
    if not state.context.empty:
        keys.append("context")
    keys += ["evidence", "trace", "params"]
    return {key: CN_NUM[index] for index, key in enumerate(keys)}


def _source_label(source: str) -> str:
    return SOURCE_LABEL.get(source, "[LLM]")


def _stamp(generated_at: str) -> str:
    return generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")


def _cell(text: str, limit: int = CELL_LIMIT) -> str:
    text = " ".join(str(text).split()).replace("|", "｜")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _refs(refs: Sequence[str]) -> str:
    return "、".join(f"`{ref}`" for ref in refs) if refs else "—"


# ---- Markdown --------------------------------------------------------------

def render_markdown(state: AgentState, generated_at: str = "") -> str:
    lines: List[str] = [
        "# 这个项目要不要写进简历？",
        "",
        f"- JD：`{state.jd.source_file}`（岗位：{state.jd.name}）",
        f"- 项目：`{state.project.source_file}`（{state.project.name}）",
        f"- 生成时间：{_stamp(generated_at)}　|　轮次：{state.rounds}　|　最终 Decision：**{state.decision}**",
    ]
    if state.needs_answer:
        lines += _md_question(state)
    else:
        lines += _md_verdict(state)
        lines += _md_llm(state)
    lines += _md_context(state)
    lines += _md_evidence(state)
    lines += _md_trace(state)
    lines += _md_params(state)
    return "\n".join(lines).rstrip() + "\n"


def _md_verdict(state: AgentState) -> List[str]:
    verdict = state.verdict
    if verdict is None:
        return []
    sec = _sections(state)
    lines = [
        "",
        f"## {sec['verdict']}、结论 {_source_label(SOURCE_RULE)}",
        "",
        f"### {VERDICT_ICON.get(verdict.call, '')} {verdict.call} {_source_label(SOURCE_RULE)}",
        "",
        verdict.headline,
        "",
        f"**StopReason**：{verdict.stop_reason}",
        "",
        f"### 为什么（{len(verdict.reasons)} 条） {_source_label(SOURCE_RULE)}",
        "",
    ]
    for index, reason in enumerate(verdict.reasons, start=1):
        lines.append(f"{index}. {reason.text}")
        lines.append(f"    - JD 出处：`{reason.jd_ref}`　项目出处：{_refs(reason.project_refs)}")
    lines += ["", f"### 风险（{len(verdict.risks)} 条） {_source_label(SOURCE_RULE)}", ""]
    for risk in verdict.risks:
        lines.append(f"- **[{risk.kind}]** {risk.text}")
    if verdict.rewrite:
        lines += ["", f"### 建议写法 {_source_label(SOURCE_RULE)}", "", f"`{verdict.rewrite}`"]
    return lines


def _md_llm(state: AgentState) -> List[str]:
    """LLM 装饰层：报告已生成之后追加的三块内容，逐块标 [LLM] 与 source。"""
    report = state.llm_report
    if report is None or report.is_empty:
        return []
    sec = _sections(state)
    lines = [
        "",
        f"## {sec['llm']}、LLM 生成内容（报告已生成之后追加，不是判定依据）",
        "",
        "> 每块都带 `source` 标记，与上面的 [规则] 段落严格分开；"
        "「值不值得写」这个结论、以及 StopReason，始终只由规则给出。",
        "",
        f"- 模型：`{report.model or '未指定'}`　|　调用：{report.calls}/{report.call_limit} 次　|　"
        f"单次超时：{report.timeout:g}s　|　失败重试：{report.retries} 次",
    ]
    if report.note:
        lines.append(f"- 说明：{report.note}")
    for block in report.blocks:
        lines += _md_llm_block(block)
    return lines


def _md_llm_block(block: LlmBlock) -> List[str]:
    lines = [
        "",
        f"### {block.label} {block.title}　`source: {block.source}`",
        "",
        f"- 状态：**{block.status}**（本块调用 {block.calls} 次，模型 `{block.model or '—'}`）",
    ]
    if block.verified:
        lines.append(f"- Verify：已校验；证据 {len(block.evidence_refs)} 条，丢弃 {len(block.dropped_claims)} 句")
    if block.items:
        lines.append("")
        for index, item in enumerate(block.items, start=1):
            lines.append(f"{index}. {item.get('question', '')}")
            extra: List[str] = []
            if item.get("target"):
                extra.append(f"追问点：{item['target']}")
            if item.get("prepare"):
                extra.append(f"怎么准备：{item['prepare']}")
            if extra:
                lines.append(f"    - {'　|　'.join(extra)}")
    if block.text:
        lines += ["", block.text]
    if block.error:
        lines += ["", f"> {block.error}"]
    for note in block.notes:
        lines.append(f"- {note}")
    return lines


# ---- HTML ------------------------------------------------------------------

HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0 auto;
       max-width: 1080px; padding: 32px 20px 64px; line-height: 1.7; }
h1 { font-size: 25px; margin-bottom: 4px; }
h2 { margin-top: 40px; border-bottom: 1px solid #8883; padding-bottom: 6px; font-size: 19px; }
h3 { margin-top: 22px; font-size: 16px; }
.meta { color: #7a7a7a; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13.5px; }
th, td { border: 1px solid #8883; padding: 7px 9px; text-align: left; vertical-align: top; }
th { background: #8881; }
code { background: #8882; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.card { border: 1px solid #8883; border-left: 4px solid #8882; border-radius: 6px;
        padding: 14px 18px; margin: 14px 0; }
.card.no { border-left-color: #c0392b; } .card.edit { border-left-color: #b78103; }
.card.yes { border-left-color: #14892c; } .card.ask { border-left-color: #2b6cb0; }
.card.ai { border-left-color: #6b46c1; }
.card h3 { margin: 0 0 6px; }
ol, ul { padding-left: 22px; }
.step { border: 1px solid #8883; border-radius: 6px; margin: 10px 0; overflow: hidden; }
.step-head { display: flex; gap: 10px; align-items: center; background: #8881;
             padding: 7px 12px; font-weight: 600; font-size: 14px; }
.step-head .idx { opacity: .55; }
.decision { margin-left: auto; font-size: 12px; padding: 1px 8px; border-radius: 10px;
            border: 1px solid currentColor; font-weight: 600; }
.d-Continue { color: #2b6cb0; } .d-Adjust { color: #b78103; }
.d-Ask { color: #b7791f; } .d-Stop { color: #14892c; }
.step dl { margin: 0; padding: 10px 14px; font-size: 13.5px; }
.step dt { font-weight: 600; opacity: .7; font-size: 12px; margin-top: 8px; }
.step dd { margin: 2px 0 0; }
.lv-result { color: #14892c; font-weight: 600; }
.lv-action { color: #b78103; font-weight: 600; }
.lv-mention { color: #7a7a7a; font-weight: 600; }
.lv-none { color: #c0392b; font-weight: 600; }
.ref { color: #7a7a7a; font-size: 12px; }
"""


def _e(text) -> str:
    return html_lib.escape(str(text))


def _ref_html(refs: Sequence[str]) -> str:
    return "、".join(f"<code>{_e(ref)}</code>" for ref in refs) if refs else "—"


def _md_context(state: AgentState) -> List[str]:
    if state.context.empty:
        return []
    sec = _sections(state)
    lines = [
        "",
        f"## {sec['context']}、记忆与知识库补充（只作参考，不参与判定）",
        "",
        f"- 记忆命中：{len(state.context.memory)} 条",
        f"- 知识库命中：{len(state.context.knowledge)} 条",
        "- 这些内容只进入 `state.context`，结论、理由、风险与 Decision 仍只由规则证据产生。",
    ]
    for title, items in (("记忆", state.context.memory), ("知识库", state.context.knowledge)):
        if not items:
            continue
        lines += ["", f"### {title}", ""]
        for index, item in enumerate(items, start=1):
            label = item.metadata.get("title") or item.source
            lines.append(f"{index}. **{_cell(str(label), 80)}**（score {item.score:.3f}）")
            lines.append(f"    - 来源：`{item.source}`")
            lines.append(f"    - {_cell(item.text, 260)}")
    return lines


def _html_context(state: AgentState) -> List[str]:
    if state.context.empty:
        return []
    sec = _sections(state)
    parts = [
        f"<h2>{sec['context']}、记忆与知识库补充（只作参考，不参与判定）</h2>",
        f'<p class="ref">记忆命中 {len(state.context.memory)} 条；知识库命中 {len(state.context.knowledge)} 条。'
        "这些内容只进入 <code>state.context</code>，结论与 Decision 仍只由规则证据产生。</p>",
    ]
    for title, items in (("记忆", state.context.memory), ("知识库", state.context.knowledge)):
        if not items:
            continue
        parts += [f"<h3>{title}</h3>", "<ol>"]
        for item in items:
            label = item.metadata.get("title") or item.source
            parts.append(
                f"<li><strong>{_e(label)}</strong>（score {item.score:.3f}）<br>"
                f'<span class="ref">来源：<code>{_e(item.source)}</code></span><br>'
                f"{_e(item.text)}</li>"
            )
        parts.append("</ol>")
    return parts


def _html_meta(state: AgentState, generated_at: str) -> str:
    return (
        '<p class="meta">'
        f"JD：<code>{_e(state.jd.source_file)}</code>（{_e(state.jd.name)}）　|　"
        f"项目：<code>{_e(state.project.source_file)}</code>（{_e(state.project.name)}）　|　"
        f"生成时间：{_e(_stamp(generated_at))}　|　轮次：{state.rounds}　|　"
        f"最终 Decision：<strong>{_e(state.decision)}</strong>"
        "</p>"
    )


def _html_verdict(state: AgentState) -> List[str]:
    verdict = state.verdict
    if verdict is None:
        return []
    css = {VERDICT_YES: "yes", VERDICT_EDIT: "edit"}.get(verdict.call, "no")
    sec = _sections(state)
    parts = [
        f"<h2>{sec['verdict']}、结论 {_source_label(SOURCE_RULE)}</h2>",
        f'<div class="card {css}">',
        f"<h3>{VERDICT_ICON.get(verdict.call, '')} {_e(verdict.call)} {_source_label(SOURCE_RULE)}</h3>",
        f"<p>{_e(verdict.headline)}</p>",
        f'<p class="ref">StopReason：{_e(verdict.stop_reason)}</p>',
        "</div>",
        f"<h3>为什么（{len(verdict.reasons)} 条） {_source_label(SOURCE_RULE)}</h3>",
        "<ol>",
    ]
    for reason in verdict.reasons:
        parts.append(
            f"<li>{_e(reason.text)}<br>"
            f'<span class="ref">JD 出处：<code>{_e(reason.jd_ref)}</code>　'
            f"项目出处：{_ref_html(reason.project_refs)}</span></li>"
        )
    parts += ["</ol>", f"<h3>风险（{len(verdict.risks)} 条） {_source_label(SOURCE_RULE)}</h3>", "<ul>"]
    for risk in verdict.risks:
        parts.append(f"<li><strong>[{_e(risk.kind)}]</strong> {_e(risk.text)}</li>")
    parts.append("</ul>")
    if verdict.rewrite:
        parts += [
            f"<h3>建议写法 {_source_label(SOURCE_RULE)}</h3>",
            f"<p><code>{_e(verdict.rewrite)}</code></p>",
        ]
    return parts


def _html_llm(state: AgentState) -> List[str]:
    """LLM 装饰层：三块内容各一张卡片，逐块标 [LLM] 与 source。"""
    report = state.llm_report
    if report is None or report.is_empty:
        return []
    sec = _sections(state)
    parts = [
        f"<h2>{sec['llm']}、LLM 生成内容（报告已生成之后追加，不是判定依据）</h2>",
        '<p class="ref">每块都带 <code>source</code> 标记，与上面的 [规则] 段落严格分开；'
        "「值不值得写」这个结论、以及 StopReason，始终只由规则给出。</p>",
        f'<p class="ref">模型：<code>{_e(report.model or "未指定")}</code>　|　'
        f"调用：{report.calls}/{report.call_limit} 次　|　单次超时：{report.timeout:g}s　|　"
        f"失败重试：{report.retries} 次</p>",
    ]
    if report.note:
        parts.append(f'<p class="ref">说明：{_e(report.note)}</p>')
    for block in report.blocks:
        parts += _html_llm_block(block)
    return parts


def _html_llm_block(block: LlmBlock) -> List[str]:
    parts = [
        '<div class="card ai">',
        f"<h3>{_e(block.label)} {_e(block.title)} "
        f'<span class="ref"><code>source: {_e(block.source)}</code></span></h3>',
        f'<p class="ref">状态：<strong>{_e(block.status)}</strong>　|　本块调用 {block.calls} 次　|　'
        f"模型 <code>{_e(block.model or '—')}</code></p>",
    ]
    if block.items:
        parts.append("<ol>")
        for item in block.items:
            parts.append(f"<li>{_e(item.get('question', ''))}")
            extra = []
            if item.get("target"):
                extra.append(f"追问点：{_e(item['target'])}")
            if item.get("prepare"):
                extra.append(f"怎么准备：{_e(item['prepare'])}")
            if extra:
                parts.append(f'<br><span class="ref">{"　|　".join(extra)}</span>')
            parts.append("</li>")
        parts.append("</ol>")
    if block.text:
        parts.append(f"<p>{_e(block.text)}</p>")
    if block.error:
        parts.append(f'<p class="ref">{_e(block.error)}</p>')
    for note in block.notes:
        parts.append(f'<p class="ref">{_e(note)}</p>')
    parts.append("</div>")
    return parts


def _html_question(state: AgentState) -> List[str]:
    last = state.trace[-1] if state.trace else None
    return [
        f"<h2>{_sections(state)['verdict']}、需要你回答一个问题</h2>",
        '<div class="card ask">',
        f"<h3>{_e(state.question)}</h3>",
        f"<p>为什么问：{_e(state.question_reason)}</p>",
        f'<p class="ref">当前资料为什么不够：{_e(last.observation if last else "—")}</p>',
        "</div>",
        "<h3>怎么继续</h3>",
        "<ul>",
        '<li>方式一（推荐）：<code>python main.py --answer "你的回答"</code> —— 回答会作为新材料并入证据池，Agent 重新检索一轮；</li>',
        f"<li>方式二：把这句话补进 <code>{_e(state.project.source_file)}</code> 后重跑；</li>",
        "<li>需要它继续追问：加 <code>--max-rounds 3</code>。</li>",
        "</ul>",
    ]


def _html_evidence(state: AgentState) -> List[str]:
    parts = [f"<h2>{_sections(state)['evidence']}、岗位要求 vs 项目证据 {_source_label(SOURCE_RULE)}</h2>"]
    if state.skipped_requirements:
        skipped = "、".join(f"{item.name}（{item.level_label}）" for item in state.skipped_requirements)
        parts.append(f'<p class="ref">项目无法证明、已移出对比的要求：{_e(skipped)}</p>')
    parts.append("<table><thead><tr>")
    for header in ("#", "岗位要求", "层级", "JD 原文", "项目证据", "证据等级", "备注"):
        parts.append(f"<th>{_e(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for index, match in enumerate(state.matches, start=1):
        fact = match.best_fact
        evidence = "—" if fact is None else _e(fact.quote_short) + f' <span class="ref">{_e(fact.ref)}</span>'
        note = match.note
        if match.missing_subs:
            extra = f"岗位强调但项目没提：{'、'.join(match.missing_subs)}"
            note = f"{note}；{extra}" if note else extra
        parts.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><strong>{_e(match.requirement_name)}</strong></td>"
            f"<td>{_e(match.requirement.level_label)}</td>"
            f'<td>{_e(match.requirement.quote_short)} <span class="ref">L{match.requirement.line_no}</span></td>'
            f"<td>{evidence}</td>"
            f'<td class="lv-{_e(match.level)}">{_e(match.level_label)}</td>'
            f"<td>{_e(note or '—')}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return parts


def _html_trace(state: AgentState) -> List[str]:
    parts = [f"<h2>{_sections(state)['trace']}、运行 Trace（每一步都可检查）</h2>"]
    for step in state.trace:
        parts += [
            '<div class="step">',
            '<div class="step-head">'
            f'<span class="idx">#{step.index}</span><span>{_e(step.action)}</span>'
            f'<span class="decision d-{_e(step.decision)}">{_e(step.decision)}</span>'
            "</div>",
            "<dl>",
            f"<dt>Observation（实际看到什么）</dt><dd>{_e(step.observation)}</dd>",
            f"<dt>State Update（状态怎么变）</dt><dd>{_e(step.state_update)}</dd>",
            f"<dt>Decision（下一步）</dt><dd>{_e(step.decision)} —— {_e(step.decision_reason)}</dd>",
            "</dl>",
            "</div>",
        ]
    return parts


def _html_params(state: AgentState) -> List[str]:
    rows = [
        ("证据等级", "有结果＝有量化指标或「已开源 / 已上线」这类交付；有动作＝有搭建 / 实现 / 调试等动词但没有结果；仅提及＝只出现在技术栈里"),
        ("Agent 策略", f"{state.strategy}（rule 离线可复现；llm 有界 plan-act-observe + verify）"),
        ("值得写", f"核心要求覆盖 ≥ {COVERAGE_OK:.0%}，且有量化结果，且必备项无缺口"),
        ("值得写但要改写", f"覆盖在 {COVERAGE_MIN:.0%} ~ {COVERAGE_OK:.0%} 之间，或必备项有缺口"),
        ("暂不建议写", f"覆盖 < {COVERAGE_MIN:.0%}，或没有任何动作级证据"),
        ("提问条件", f"看不出个人贡献 / 与岗位零交集 / 事实少于 {MIN_FACTS} 条 / 没有结果且覆盖 < 50%"),
        ("轮次上限", f"最多 Ask 一轮（max_rounds = {MAX_ROUNDS}），补充后仍不足就给保守结论"),
    ]
    rows += _llm_param_rows(state)
    parts = [
        f"<h2>{_sections(state)['params']}、判定口径（全部写死，可直接检查）</h2>",
        "<table><thead><tr><th>项</th><th>规则</th></tr></thead><tbody>",
    ]
    for key, value in rows:
        parts.append(f"<tr><td>{_e(key)}</td><td>{_e(value)}</td></tr>")
    parts.append("</tbody></table>")
    return parts


def render_html(state: AgentState, generated_at: str = "") -> str:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>项目简历判断 · {_e(state.project.name)}</title>",
        f"<style>{HTML_CSS}</style>",
        "</head>",
        "<body>",
        "<h1>这个项目要不要写进简历？</h1>",
        _html_meta(state, generated_at),
    ]
    if state.needs_answer:
        parts += _html_question(state)
    else:
        parts += _html_verdict(state)
        parts += _html_llm(state)
    parts += _html_context(state)
    parts += _html_evidence(state)
    parts += _html_trace(state)
    parts += _html_params(state)
    parts += ["</body>", "</html>"]
    return "\n".join(parts)


def _md_question(state: AgentState) -> List[str]:
    last = state.trace[-1] if state.trace else None
    return [
        "",
        f"## {_sections(state)['verdict']}、需要你回答一个问题",
        "",
        f"> {state.question}",
        "",
        f"**为什么问这个**：{state.question_reason}",
        "",
        f"**当前资料为什么不够**：{last.observation if last else '—'}",
        "",
        "### 怎么继续",
        "",
        "- 方式一（推荐）：`python main.py --answer \"你的回答\"` —— 回答会作为新材料并入证据池，Agent 重新检索一轮；",
        f"- 方式二：把这句话补进 `{state.project.source_file}` 后重跑；",
        "- 需要它继续追问：加 `--max-rounds 3`。",
    ]


def _md_evidence(state: AgentState) -> List[str]:
    sec = _sections(state)
    lines = ["", f"## {sec['evidence']}、岗位要求 vs 项目证据 {_source_label(SOURCE_RULE)}", ""]
    if state.skipped_requirements:
        skipped = "、".join(f"{item.name}（{item.level_label}）" for item in state.skipped_requirements)
        lines += [f"> 项目无法证明、已移出对比的要求：{skipped}", ""]
    lines += [
        "| # | 岗位要求 | 层级 | JD 原文 | 项目证据 | 证据等级 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, match in enumerate(state.matches, start=1):
        fact = match.best_fact
        if fact is None:
            evidence = "—"
        elif fact.line_no > 0:
            evidence = f"{fact.quote_short}（L{fact.line_no}）"
        elif fact.source_file.startswith("图片:"):
            evidence = f"{fact.quote_short}（{fact.source_file}）"
        else:
            evidence = f"{fact.quote_short}（补充说明）"
        note = match.note
        if match.missing_subs:
            extra = f"岗位强调但项目没提：{'、'.join(match.missing_subs)}"
            note = f"{note}；{extra}" if note else extra
        lines.append(
            "| "
            + " | ".join(
                _cell(item)
                for item in (
                    str(index),
                    match.requirement_name,
                    match.requirement.level_label,
                    f"{match.requirement.quote_short}（L{match.requirement.line_no}）",
                    evidence,
                    f"{LEVEL_ICON.get(match.level, '')}{match.level_label}",
                    note or "—",
                )
            )
            + " |"
        )
    return lines


def _md_trace(state: AgentState) -> List[str]:
    lines = [
        "",
        f"## {_sections(state)['trace']}、运行 Trace（每一步都可检查）",
        "",
        "| # | Action | Observation（实际看到什么） | State Update（状态怎么变） | Decision（下一步） |",
        "|---|---|---|---|---|",
    ]
    for step in state.trace:
        lines.append("| " + " | ".join(_cell(item) for item in step.as_row()) + " |")
    return lines


def _llm_param_rows(state: AgentState) -> List[Tuple[str, str]]:
    """只有真的用了可选层时，才在「判定口径」里补上对应说明。"""
    rows: List[Tuple[str, str]] = []
    report = state.llm_report
    if report is not None:
        if report.strategy == "llm" or state.strategy == "llm":
            rows.append(
                (
                    "v4 LLM 策略（--strategy llm）",
                    "Planner 每一步只选择一个白名单动作；PolicyGuard 校验 action / args / reason；"
                    "Observe 把结果回灌 state；Verify 丢弃无法回到 L0 证据的文字。",
                )
            )
        tasks = "、".join(f"{item.title}（{item.source}）" for item in report.blocks)
        rows.append(
            (
                "LLM 装饰层（--llm）",
                f"只在规则报告生成之后介入，做三件规则做不好的事：{tasks}；"
                f"每块都带 source 标记，报告里标 [LLM]，与 [规则] 段落严格分开",
            )
        )
        rows.append(
            (
                "调用纪律（--llm）",
                f"单次超时 {report.timeout:g}s、失败重试 {report.retries} 次、"
                f"一次运行最多 {report.call_limit} 次调用（本次已用 {report.calls} 次）；"
                "不带 --llm 时这一层完全不创建，也不会读 .env",
            )
        )
        rows.append(
            (
                "LLM 不做结论",
                "模型输出里出现「值得写 / 不建议写」这类档位词的内容会被整块丢弃；"
                "结论档位、StopReason 与 Decision 只由规则给出",
            )
        )
        rows.append(
            (
                "失败可降级（--llm）",
                "任何一块调用失败只显示「未启用 / 调用失败」，规则报告、Trace 与退出码不受影响",
            )
        )
    if state.vision.enabled:
        rows.append(
            (
                "图片解析（--vision）",
                f"由 {state.vision.model} 把引用的图片读成文字事实（source_file = 「图片:相对路径」），"
                "证据等级仍由 classify() 判定，解析结果需人工复核",
            )
        )
    return rows


def _md_params(state: AgentState) -> List[str]:
    lines = [
        "",
        f"## {_sections(state)['params']}、判定口径（全部写死，可直接检查）",
        "",
        "| 项 | 规则 |",
        "|---|---|",
        f"| Agent 策略 | `{state.strategy}`（rule = 离线可复现；llm = 有界 plan-act-observe + verify） |",
        "| 证据等级 | 有结果＝有量化指标或「已开源 / 已上线」这类交付；有动作＝有搭建 / 实现 / 调试等动词但没有结果；仅提及＝只出现在技术栈里 |",
        f"| 值得写 | 核心要求覆盖 ≥ {COVERAGE_OK:.0%}，且有量化结果，且必备项无缺口 |",
        f"| 值得写但要改写 | 覆盖在 {COVERAGE_MIN:.0%} ~ {COVERAGE_OK:.0%} 之间，或必备项有缺口 |",
        f"| 暂不建议写 | 覆盖 < {COVERAGE_MIN:.0%}，或没有任何动作级证据 |",
        f"| 提问条件 | 看不出个人贡献 / 与岗位零交集 / 事实少于 {MIN_FACTS} 条 / 没有结果且覆盖 < 50% |",
        f"| 轮次上限 | 最多 Ask 一轮（max_rounds = {MAX_ROUNDS}），补充后仍不足就给保守结论 |",
    ]
    lines += [f"| {key} | {value} |" for key, value in _llm_param_rows(state)]
    return lines


# ---- JSON ------------------------------------------------------------------

def _context_payload(state: AgentState) -> Dict:
    def rows(items):
        return [
            {
                "source": item.source,
                "text": item.text,
                "score": item.score,
                "category": item.category,
                "metadata": item.metadata,
            }
            for item in items
        ]

    return {
        "memory": rows(state.context.memory),
        "knowledge": rows(state.context.knowledge),
    }

def build_payload(state: AgentState, generated_at: str = "") -> Dict:
    verdict = state.verdict
    return {
        "generated_at": _stamp(generated_at),
        "decision": state.decision,
        "rounds": state.rounds,
        "sufficiency": state.sufficiency,
        "strategy": state.strategy,
        "jd": {
            "source_file": state.jd.source_file,
            "title": state.jd.title,
            "name": state.jd.name,
            "company": state.jd.company,
            "job_type": state.jd.job_type,
            "position_count": state.jd.position_count,
            "position_titles": state.jd.position_titles,
            "requirement_count": len(state.jd.requirements),
            "line_count": state.jd.line_count,
            "bullet_count": state.jd.bullet_count,
        },
        "project": {
            "source_file": state.project.source_file,
            "title": state.project.title,
            "name": state.project.name,
            "line_count": state.project.line_count,
            "fact_count": len(state.project.facts),
            "result_facts": len(state.project.result_facts),
            "facts": [
                {
                    "text": fact.text,
                    "section": fact.section,
                    "line_no": fact.line_no,
                    "source_file": fact.source_file,
                    "ref": fact.ref,
                    "level": fact.level,
                    "metrics": fact.metrics,
                    "capabilities": fact.capabilities,
                    "negated": fact.negated,
                }
                for fact in state.project.facts
            ],
        },
        "vision": {
            "enabled": state.vision.enabled,
            "model": state.vision.model,
            "parsed": state.vision.parsed,
            "facts_added": state.vision.facts_added,
            "error": state.vision.error,
            "notes": state.vision.notes,
            "images": [
                {"raw": item.raw, "path": item.path, "ref": item.ref, "line_no": item.line_no}
                for item in state.vision.images
            ],
        },
        "llm": _llm_payload(state),
        **({"context": _context_payload(state)} if not state.context.empty else {}),
        "core_requirements": [
            _requirement_payload(item) for item in state.core_requirements
        ],
        "skipped_requirements": [
            _requirement_payload(item) for item in state.skipped_requirements
        ],
        "evidence_matches": [
            {
                "requirement": match.requirement_name,
                "requirement_level": match.requirement.level_label,
                "jd_ref": match.requirement.ref,
                "level": match.level,
                "relaxed": match.relaxed,
                "matched_subs": match.matched_subs,
                "missing_subs": match.missing_subs,
                "note": match.note,
                "project_evidence": [
                    {"text": fact.text, "line_no": fact.line_no, "level": fact.level, "ref": fact.ref}
                    for fact in match.facts
                ],
            }
            for match in state.matches
        ],
        "question": state.question or None,
        "question_reason": state.question_reason or None,
        "observations": [item.to_dict() for item in state.observations],
        "verdict": None
        if verdict is None
        else {
            "call": verdict.call,
            "headline": verdict.headline,
            "reasons": [
                {
                    "text": reason.text,
                    "requirement": reason.requirement_name,
                    "jd_ref": reason.jd_ref,
                    "project_refs": reason.project_refs,
                }
                for reason in verdict.reasons
            ],
            "risks": [{"text": risk.text, "kind": risk.kind, "refs": risk.refs} for risk in verdict.risks],
            "rewrite": verdict.rewrite,
            "stop_reason": verdict.stop_reason,
            "source": SOURCE_RULE,
        },
        "trace": [
            {
                "step": step.index,
                "action": step.action,
                "observation": step.observation,
                "state_update": step.state_update,
                "decision": step.decision,
                "decision_reason": step.decision_reason,
            }
            for step in state.trace
        ],
        "params": {
            "max_rounds": MAX_ROUNDS,
            "strategy": state.strategy,
            "min_facts": MIN_FACTS,
            "coverage_ok": COVERAGE_OK,
            "coverage_min": COVERAGE_MIN,
            "llm": state.llm_report is not None,
            "llm_calls": _llm_stat(state, "calls"),
            "llm_call_limit": _llm_stat(state, "call_limit"),
            "llm_timeout": _llm_stat(state, "timeout"),
            "llm_retries": _llm_stat(state, "retries"),
            "llm_blocks": _llm_status_map(state),
            "vision": state.vision.enabled,
        },
    }


def _llm_payload(state: AgentState) -> Optional[Dict]:
    """LLM 装饰层的结构化输出；不带 --llm 时是 None（与 v1.0 的 JSON 完全一致）。"""
    report = state.llm_report
    if report is None:
        return None
    return {
        "enabled": report.enabled,
        "model": report.model,
        "calls": report.calls,
        "call_limit": report.call_limit,
        "timeout": report.timeout,
        "retries": report.retries,
        "strategy": report.strategy,
        "planner_calls": report.planner_calls,
        "verified_blocks": report.verified_blocks,
        "dropped_claims": report.dropped_claims,
        "note": report.note,
        "ok_count": report.ok_count,
        "blocks": [
            {
                "source": item.source,
                "label": item.label,
                "title": item.title,
                "status": item.status,
                "model": item.model,
                "calls": item.calls,
                "text": item.text,
                "items": item.items,
                "error": item.error,
                "enabled": item.enabled,
                "idle": item.idle,
                "skipped": item.skipped,
                "failed": item.failed,
                "discarded": item.discarded,
                "verified": item.verified,
                "evidence_refs": item.evidence_refs,
                "dropped_claims": item.dropped_claims,
                "notes": item.notes,
                "note": "模型生成，不是判定依据；规则结论见 verdict",
            }
            for item in report.blocks
        ],
    }


def _llm_status_map(state: AgentState) -> Dict[str, str]:
    report = state.llm_report
    return {item.source: item.status for item in report.blocks} if report else {}


def _llm_stat(state: AgentState, name: str):
    report = state.llm_report
    return getattr(report, name) if report else None


def render_json(state: AgentState, generated_at: str = "") -> str:
    return json.dumps(build_payload(state, generated_at), ensure_ascii=False, indent=2)


def _requirement_payload(item: Requirement) -> Dict:
    """一条岗位要求的结构化形态：保留旧键（name / capability）并补齐重建所需字段。"""
    return {
        "name": item.name,
        "capability": item.capability_key,
        "capability_key": item.capability_key,
        "capability_name": item.name,
        "category": item.category,
        "level": item.level,
        "section": item.section,
        "line_no": item.line_no,
        "source_file": item.source_file,
        "quote": item.quote,
    }


def _requirement_from_payload(data: Dict) -> Requirement:
    return Requirement(
        capability_key=str(data.get("capability_key") or data.get("capability") or ""),
        capability_name=str(data.get("capability_name") or data.get("name") or ""),
        category=str(data.get("category") or ""),
        level=str(data.get("level") or ""),
        section=str(data.get("section") or ""),
        line_no=int(data.get("line_no") or 0),
        quote=str(data.get("quote") or ""),
        source_file=str(data.get("source_file") or ""),
    )


def _fact_parts(ref: str, line_no: int) -> Tuple[str, int]:
    if "::L" in str(ref):
        source_file, _, line = str(ref).rpartition("::L")
        return source_file, int(line) if line.isdigit() else int(line_no or 0)
    return str(ref), int(line_no or 0)


def _fact_from_payload(data: Dict) -> ProjectFact:
    source_file, line_no = _fact_parts(data.get("ref"), data.get("line_no"))
    metrics = list(data.get("metrics") or [])
    return ProjectFact(
        text=str(data.get("text") or ""),
        section=str(data.get("section") or ""),
        line_no=line_no,
        source_file=source_file or str(data.get("source_file") or ""),
        capabilities=list(data.get("capabilities") or []),
        negated=list(data.get("negated") or []),
        level=str(data.get("level") or EVIDENCE_MENTION),
        has_metric=bool(metrics or data.get("has_metric")),
        metrics=metrics,
    )


def _llm_report_from_payload(data: Optional[Dict]) -> Optional[LlmReport]:
    if not data:
        return None
    blocks = [
        LlmBlock(
            source=str(item.get("source") or ""),
            title=str(item.get("title") or ""),
            model=str(item.get("model") or ""),
            enabled=bool(item.get("enabled", True)),
            calls=int(item.get("calls") or 0),
            text=str(item.get("text") or ""),
            items=list(item.get("items") or []),
            error=str(item.get("error") or ""),
            failed=bool(item.get("failed")),
            discarded=bool(item.get("discarded")),
            idle=bool(item.get("idle")),
            skipped=bool(item.get("skipped")),
            verified=bool(item.get("verified")),
            evidence_refs=list(item.get("evidence_refs") or []),
            dropped_claims=list(item.get("dropped_claims") or []),
            notes=list(item.get("notes") or []),
        )
        for item in data.get("blocks") or []
    ]
    return LlmReport(
        enabled=bool(data.get("enabled")),
        model=str(data.get("model") or ""),
        calls=int(data.get("calls") or 0),
        call_limit=int(data.get("call_limit") or 10),
        timeout=float(data.get("timeout") or 30.0),
        retries=int(data.get("retries") or 0),
        blocks=blocks,
        note=str(data.get("note") or ""),
        strategy=str(data.get("strategy") or ""),
        planner_calls=int(data.get("planner_calls") or 0),
        verified_blocks=int(data.get("verified_blocks") or 0),
        dropped_claims=int(data.get("dropped_claims") or 0),
    )


def state_from_payload(payload: Dict) -> AgentState:
    """把持久化的 JSON payload 还原成可以再次渲染的 AgentState。

    output/ 只存 JSON；Markdown / HTML 报告在读取时由这份 payload 重新渲染，
    因此需要把证据、结论、Trace 这些都还原回对象。
    """
    jd_data = payload.get("jd") or {}
    project_data = payload.get("project") or {}
    core = [_requirement_from_payload(item) for item in payload.get("core_requirements") or []]
    by_name = {}
    by_line = {}
    for requirement in core:
        by_name.setdefault(requirement.capability_name, requirement)
        by_line.setdefault((requirement.capability_name, requirement.line_no), requirement)
    matches: List[EvidenceMatch] = []
    for item in payload.get("evidence_matches") or []:
        name = str(item.get("requirement") or "")
        source_file, line_no = _fact_parts(item.get("jd_ref"), 0)
        requirement = by_line.get((name, line_no)) or by_name.get(name)
        if requirement is None:
            requirement = Requirement(
                capability_key="",
                capability_name=name,
                category="",
                level="",
                section="",
                line_no=line_no,
                quote=name,
                source_file=source_file,
            )
        matches.append(
            EvidenceMatch(
                requirement=requirement,
                facts=[_fact_from_payload(fact) for fact in item.get("project_evidence") or []],
                level=str(item.get("level") or EVIDENCE_NONE),
                matched_subs=list(item.get("matched_subs") or []),
                missing_subs=list(item.get("missing_subs") or []),
                relaxed=bool(item.get("relaxed")),
                note=str(item.get("note") or ""),
            )
        )
    jd = JobPosting(
        jd_id=str(jd_data.get("name") or jd_data.get("source_file") or ""),
        title=str(jd_data.get("title") or ""),
        source_file=str(jd_data.get("source_file") or ""),
        company=str(jd_data.get("company") or ""),
        job_type=str(jd_data.get("job_type") or ""),
        line_count=int(jd_data.get("line_count") or 0),
        bullet_count=int(jd_data.get("bullet_count") or 0),
        position_count=int(jd_data.get("position_count") or 1),
        position_titles=list(jd_data.get("position_titles") or []),
        requirements=core,
    )
    project = Project(
        title=str(project_data.get("title") or ""),
        source_file=str(project_data.get("source_file") or ""),
        line_count=int(project_data.get("line_count") or 0),
        facts=[_fact_from_payload(fact) for fact in project_data.get("facts") or []],
    )
    vision_data = payload.get("vision") or {}
    vision = VisionReport(
        enabled=bool(vision_data.get("enabled")),
        model=str(vision_data.get("model") or ""),
        images=[
            ImageRef(
                raw=str(item.get("raw") or ""),
                path=str(item.get("path") or ""),
                ref=str(item.get("ref") or ""),
                line_no=int(item.get("line_no") or 0),
            )
            for item in vision_data.get("images") or []
        ],
        parsed=int(vision_data.get("parsed") or 0),
        facts_added=int(vision_data.get("facts_added") or 0),
        error=str(vision_data.get("error") or ""),
        notes=list(vision_data.get("notes") or []),
    )
    context_data = payload.get("context") or {}
    context = RetrievedContext(
        memory=[_context_item(item) for item in context_data.get("memory") or []],
        knowledge=[_context_item(item) for item in context_data.get("knowledge") or []],
    )
    verdict_data = payload.get("verdict")
    verdict = None
    if verdict_data:
        verdict = Verdict(
            call=str(verdict_data.get("call") or ""),
            headline=str(verdict_data.get("headline") or ""),
            reasons=[
                Reason(
                    text=str(item.get("text") or ""),
                    requirement_name=str(item.get("requirement") or ""),
                    jd_ref=str(item.get("jd_ref") or ""),
                    project_refs=list(item.get("project_refs") or []),
                )
                for item in verdict_data.get("reasons") or []
            ],
            risks=[
                Risk(
                    text=str(item.get("text") or ""),
                    kind=str(item.get("kind") or ""),
                    refs=list(item.get("refs") or []),
                )
                for item in verdict_data.get("risks") or []
            ],
            rewrite=str(verdict_data.get("rewrite") or ""),
            stop_reason=str(verdict_data.get("stop_reason") or ""),
            llm_report=_llm_report_from_payload(payload.get("llm")),
        )
    return AgentState(
        jd=jd,
        project=project,
        core_requirements=core,
        skipped_requirements=[
            _requirement_from_payload(item) for item in payload.get("skipped_requirements") or []
        ],
        matches=matches,
        trace=[
            TraceStep(
                index=int(item.get("step") or 0),
                action=str(item.get("action") or ""),
                observation=str(item.get("observation") or ""),
                state_update=str(item.get("state_update") or ""),
                decision=str(item.get("decision") or ""),
                decision_reason=str(item.get("decision_reason") or ""),
            )
            for item in payload.get("trace") or []
        ],
        rounds=int(payload.get("rounds") or 1),
        question=str(payload.get("question") or ""),
        question_reason=str(payload.get("question_reason") or ""),
        sufficiency=str(payload.get("sufficiency") or ""),
        verdict=verdict,
        vision=vision,
        context=context,
        observations=[
            AgentObservation(
                action=str(item.get("action") or ""),
                summary=str(item.get("summary") or ""),
                data=dict(item.get("data") or {}),
                evidence_refs=list(item.get("evidence_refs") or []),
                reason=str(item.get("reason") or ""),
            )
            for item in payload.get("observations") or []
        ],
        strategy=str(payload.get("strategy") or payload.get("params", {}).get("strategy") or "rule"),
    )


def _context_item(data: Dict) -> ContextItem:
    return ContextItem(
        source=str(data.get("source") or ""),
        text=str(data.get("text") or ""),
        score=float(data.get("score") or 0.0),
        category=str(data.get("category") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


def render_markdown_from_payload(payload: Dict) -> str:
    """从持久化 JSON 还原出 Markdown 报告（output/ 不再落盘 Markdown）。"""
    return render_markdown(state_from_payload(payload), str(payload.get("generated_at") or ""))


def render_html_from_payload(payload: Dict) -> str:
    """从持久化 JSON 还原出 HTML 报告（output/ 不再落盘 HTML）。"""
    return render_html(state_from_payload(payload), str(payload.get("generated_at") or ""))


# ---- 终端 ------------------------------------------------------------------

def render_console(state: AgentState) -> List[str]:
    lines: List[str] = ["", "=" * 72, "运行 Trace", "=" * 72]
    for step in state.trace:
        lines += [
            f"[{step.index}] Action      : {step.action}",
            f"    Observation : {step.observation}",
            f"    State Update: {step.state_update}",
            f"    Decision    : {step.decision} —— {step.decision_reason}",
            "",
        ]
    if not state.context.empty:
        lines += ["=" * 72, "补充上下文（只作参考，不参与判定）", "=" * 72]
        for label, items in (("记忆", state.context.memory), ("知识库", state.context.knowledge)):
            for item in items:
                name = item.metadata.get("title") or item.source
                lines.append(f"  [{label}] {name}（score {item.score:.3f}）：{item.text}")
        lines.append("")
    lines.append("=" * 72)
    if state.needs_answer:
        lines += [
            "需要你回答 1 个问题（资料不足，不猜）",
            "=" * 72,
            f"  {state.question}",
            f"  为什么问：{state.question_reason}",
            "",
            "  继续方式：python main.py --answer \"你的回答\"",
        ]
    elif state.verdict:
        verdict = state.verdict
        lines += [
            f"结论 {_source_label(SOURCE_RULE)}：{verdict.call}",
            "=" * 72,
            f"  {verdict.headline}",
            "",
            f"为什么（{len(verdict.reasons)} 条）：",
        ]
        for index, reason in enumerate(verdict.reasons, start=1):
            lines.append(f"  {index}. {reason.text}")
            lines.append(f"     JD {reason.jd_ref}｜项目 {_refs(reason.project_refs)}")
        lines += ["", f"风险（{len(verdict.risks)} 条）："]
        for risk in verdict.risks:
            lines.append(f"  - [{risk.kind}] {risk.text}")
        if verdict.rewrite:
            lines += ["", f"建议写法：{verdict.rewrite}"]
        lines += _console_llm(state)
        lines += ["", f"StopReason：{verdict.stop_reason}"]
    return lines


def _console_llm(state: AgentState) -> List[str]:
    """LLM 装饰层（可选）：三个块各一行小标题，逐块标 [LLM] 与 source。"""
    report = state.llm_report
    if report is None or report.is_empty:
        return []
    lines = [
        "",
        f"LLM 生成内容（报告已生成之后追加，不是判定依据；模型 {report.model or '未指定'}，"
        f"调用 {report.calls}/{report.call_limit} 次，超时 {report.timeout:g}s，重试 {report.retries} 次）：",
    ]
    if report.note:
        lines.append(f"  说明：{report.note}")
    for block in report.blocks:
        lines.append(f"  {block.label} {block.title}（source: {block.source}｜状态：{block.status}）：")
        for index, item in enumerate(block.items, start=1):
            lines.append(f"    {index}. {item.get('question', '')}")
            if item.get("target"):
                lines.append(f"       追问点：{item['target']}")
            if item.get("prepare"):
                lines.append(f"       怎么准备：{item['prepare']}")
        if block.text:
            lines.append(f"    {block.text}")
        if block.error:
            lines.append(f"    （{block.error}；以上结论、理由与风险仍全部来自规则）")
    return lines
