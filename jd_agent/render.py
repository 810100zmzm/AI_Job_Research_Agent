"""渲染：把一次 Agent 运行的结果输出成 Markdown / JSON / HTML / 终端文本。

四种输出说同一件事：结论 -> 理由 / 风险 -> 岗位要求对照表 -> 完整 Trace。
Trace 是这份产物里最该被检查的部分：每一步都能看到「看到了什么、状态怎么变、下一步为什么这么走」。
"""
from __future__ import annotations

import html as html_lib
import json
from datetime import datetime
from typing import Dict, List, Sequence

from .agent import COVERAGE_MIN, COVERAGE_OK, MAX_ROUNDS, MIN_FACTS
from .schema import (
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_NONE,
    EVIDENCE_RESULT,
    VERDICT_EDIT,
    VERDICT_NO,
    VERDICT_YES,
    AgentState,
)

LEVEL_ICON = {
    EVIDENCE_RESULT: "✅",
    EVIDENCE_ACTION: "🟠",
    EVIDENCE_MENTION: "🟡",
    EVIDENCE_NONE: "❌",
}
VERDICT_ICON = {VERDICT_YES: "✅", VERDICT_EDIT: "🟠", VERDICT_NO: "❌"}
CELL_LIMIT = 78


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
    lines += _md_evidence(state)
    lines += _md_trace(state)
    lines += _md_params()
    return "\n".join(lines).rstrip() + "\n"


def _md_verdict(state: AgentState) -> List[str]:
    verdict = state.verdict
    if verdict is None:
        return []
    lines = [
        "",
        "## 一、结论",
        "",
        f"### {VERDICT_ICON.get(verdict.call, '')} {verdict.call}",
        "",
        verdict.headline,
        "",
        f"**StopReason**：{verdict.stop_reason}",
        "",
        f"### 为什么（{len(verdict.reasons)} 条）",
        "",
    ]
    for index, reason in enumerate(verdict.reasons, start=1):
        lines.append(f"{index}. {reason.text}")
        lines.append(f"    - JD 出处：`{reason.jd_ref}`　项目出处：{_refs(reason.project_refs)}")
    lines += ["", f"### 风险（{len(verdict.risks)} 条）", ""]
    for risk in verdict.risks:
        lines.append(f"- **[{risk.kind}]** {risk.text}")
    if verdict.rewrite:
        lines += ["", "### 建议写法", "", f"`{verdict.rewrite}`"]
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
    parts = [
        "<h2>一、结论</h2>",
        f'<div class="card {css}">',
        f"<h3>{VERDICT_ICON.get(verdict.call, '')} {_e(verdict.call)}</h3>",
        f"<p>{_e(verdict.headline)}</p>",
        f'<p class="ref">StopReason：{_e(verdict.stop_reason)}</p>',
        "</div>",
        f"<h3>为什么（{len(verdict.reasons)} 条）</h3>",
        "<ol>",
    ]
    for reason in verdict.reasons:
        parts.append(
            f"<li>{_e(reason.text)}<br>"
            f'<span class="ref">JD 出处：<code>{_e(reason.jd_ref)}</code>　'
            f"项目出处：{_ref_html(reason.project_refs)}</span></li>"
        )
    parts += ["</ol>", f"<h3>风险（{len(verdict.risks)} 条）</h3>", "<ul>"]
    for risk in verdict.risks:
        parts.append(f"<li><strong>[{_e(risk.kind)}]</strong> {_e(risk.text)}</li>")
    parts.append("</ul>")
    if verdict.rewrite:
        parts += ["<h3>建议写法</h3>", f"<p><code>{_e(verdict.rewrite)}</code></p>"]
    return parts


def _html_question(state: AgentState) -> List[str]:
    last = state.trace[-1] if state.trace else None
    return [
        "<h2>一、需要你回答一个问题</h2>",
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
    parts = ["<h2>二、岗位要求 vs 项目证据</h2>"]
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
    parts = ["<h2>三、运行 Trace（每一步都可检查）</h2>"]
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


def _html_params() -> List[str]:
    rows = [
        ("证据等级", "有结果＝有量化指标或「已开源 / 已上线」这类交付；有动作＝有搭建 / 实现 / 调试等动词但没有结果；仅提及＝只出现在技术栈里"),
        ("值得写", f"核心要求覆盖 ≥ {COVERAGE_OK:.0%}，且有量化结果，且必备项无缺口"),
        ("值得写但要改写", f"覆盖在 {COVERAGE_MIN:.0%} ~ {COVERAGE_OK:.0%} 之间，或必备项有缺口"),
        ("暂不建议写", f"覆盖 < {COVERAGE_MIN:.0%}，或没有任何动作级证据"),
        ("提问条件", f"看不出个人贡献 / 与岗位零交集 / 事实少于 {MIN_FACTS} 条 / 没有结果且覆盖 < 50%"),
        ("轮次上限", f"最多 Ask 一轮（max_rounds = {MAX_ROUNDS}），补充后仍不足就给保守结论"),
    ]
    parts = ["<h2>四、判定口径（全部写死，可直接检查）</h2>", "<table><thead><tr><th>项</th><th>规则</th></tr></thead><tbody>"]
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
    parts += _html_question(state) if state.needs_answer else _html_verdict(state)
    parts += _html_evidence(state)
    parts += _html_trace(state)
    parts += _html_params()
    parts += ["</body>", "</html>"]
    return "\n".join(parts)


def _md_question(state: AgentState) -> List[str]:
    last = state.trace[-1] if state.trace else None
    return [
        "",
        "## 一、需要你回答一个问题",
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
    lines = ["", "## 二、岗位要求 vs 项目证据", ""]
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
        "## 三、运行 Trace（每一步都可检查）",
        "",
        "| # | Action | Observation（实际看到什么） | State Update（状态怎么变） | Decision（下一步） |",
        "|---|---|---|---|---|",
    ]
    for step in state.trace:
        lines.append("| " + " | ".join(_cell(item) for item in step.as_row()) + " |")
    return lines


def _md_params() -> List[str]:
    return [
        "",
        "## 四、判定口径（全部写死，可直接检查）",
        "",
        "| 项 | 规则 |",
        "|---|---|",
        "| 证据等级 | 有结果＝有量化指标或「已开源 / 已上线」这类交付；有动作＝有搭建 / 实现 / 调试等动词但没有结果；仅提及＝只出现在技术栈里 |",
        f"| 值得写 | 核心要求覆盖 ≥ {COVERAGE_OK:.0%}，且有量化结果，且必备项无缺口 |",
        f"| 值得写但要改写 | 覆盖在 {COVERAGE_MIN:.0%} ~ {COVERAGE_OK:.0%} 之间，或必备项有缺口 |",
        f"| 暂不建议写 | 覆盖 < {COVERAGE_MIN:.0%}，或没有任何动作级证据 |",
        f"| 提问条件 | 看不出个人贡献 / 与岗位零交集 / 事实少于 {MIN_FACTS} 条 / 没有结果且覆盖 < 50% |",
        f"| 轮次上限 | 最多 Ask 一轮（max_rounds = {MAX_ROUNDS}），补充后仍不足就给保守结论 |",
    ]


# ---- JSON ------------------------------------------------------------------

def build_payload(state: AgentState, generated_at: str = "") -> Dict:
    verdict = state.verdict
    return {
        "generated_at": _stamp(generated_at),
        "decision": state.decision,
        "rounds": state.rounds,
        "sufficiency": state.sufficiency,
        "jd": {
            "source_file": state.jd.source_file,
            "title": state.jd.title,
            "company": state.jd.company,
            "job_type": state.jd.job_type,
            "position_count": state.jd.position_count,
            "position_titles": state.jd.position_titles,
            "requirement_count": len(state.jd.requirements),
        },
        "project": {
            "source_file": state.project.source_file,
            "title": state.project.title,
            "fact_count": len(state.project.facts),
            "result_facts": len(state.project.result_facts),
            "facts": [
                {
                    "text": fact.text,
                    "section": fact.section,
                    "line_no": fact.line_no,
                    "level": fact.level,
                    "metrics": fact.metrics,
                    "capabilities": fact.capabilities,
                    "negated": fact.negated,
                }
                for fact in state.project.facts
            ],
        },
        "core_requirements": [
            {
                "name": item.name,
                "capability": item.capability_key,
                "category": item.category,
                "level": item.level,
                "line_no": item.line_no,
                "source_file": item.source_file,
                "quote": item.quote,
            }
            for item in state.core_requirements
        ],
        "skipped_requirements": [
            {"name": item.name, "level": item.level, "line_no": item.line_no, "quote": item.quote}
            for item in state.skipped_requirements
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
            "min_facts": MIN_FACTS,
            "coverage_ok": COVERAGE_OK,
            "coverage_min": COVERAGE_MIN,
        },
    }


def render_json(state: AgentState, generated_at: str = "") -> str:
    return json.dumps(build_payload(state, generated_at), ensure_ascii=False, indent=2)


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
            f"结论：{verdict.call}",
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
        lines += ["", f"StopReason：{verdict.stop_reason}"]
    return lines
