"""Polish Agent 的大模型层：基于「写作简报」生成建议写法 / 追问预演 / 润色。

三块内容彼此独立，一块失败不影响另外两块。三条红线与主线完全一致：

  * 规则优先：模型只在写作简报（`polish_brief.PolishBrief`）生成之后介入，而且只读这份简报 ——
    不是 JD / 简历全文，只有规则挑出来的句子、等级与出处；词库能判的绝不问模型；
  * 来源可区分：每块内容都带 source（llm-suggest / llm-interview / llm-polish），一律标 [LLM]，
    与规则写的「写作简报」严格分开；模型输出的「值不值得写」这类结论性表述会被整块丢弃；
  * 失败可降级：没有 Key → 三块「未启用」；调用失败 → 该块「调用失败」；内容越界 → 「已丢弃」。
    三种情况都只改这一块的文字，不改 StopReason、不改退出码。

调用纪律直接复用主线的账本（`generate.CallBudget`）：单次超时 30 秒、失败重试 1 次、
一次运行最多 `--llm-max-calls` 次（默认 10）。这一层不 new 客户端，也不读 `.env`。

与主线的一处差别：主线里「面试追问预演」是闲置状态（待改版），而 Polish Agent 的存在理由
就是这三块内容，所以这里三块都开启。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .generate import CallBudget, LLMBudgetError, verdict_word
from ..core.llm import parse_json_reply
from .polish_brief import PolishBrief
from ..core.schema import (
    SOURCE_LLM_INTERVIEW,
    SOURCE_LLM_POLISH,
    SOURCE_LLM_SUGGEST,
    LlmBlock,
    LlmReport,
)

INTERVIEW_COUNT = 3          # 追问预演固定 3 条
MAX_REQUIREMENTS = 8         # 送给模型的要求最多几条
MAX_FACTS = 10               # 送给模型的已核验事实最多几条
MAX_FACT_CHARS = 160
MAX_QUOTE_CHARS = 80
MAX_DRAFT_CHARS = 320
MAX_POLISH_CHARS = 200
MAX_LINE_CHARS = 120
MAX_REFS = 3

# source -> (Trace 里的 Action 名, 报告里的标题, 一句话职责)
TASK_SPECS = (
    (SOURCE_LLM_SUGGEST, "SuggestWording", "建议写法", "用规则已核验的事实拼一版可编辑的简历写法"),
    (SOURCE_LLM_INTERVIEW, "RehearseInterview", "追问预演", "针对要求、缺口与风险预演最可能被追问的 3 个问题"),
    (SOURCE_LLM_POLISH, "PolishWording", "润色", "把已核验事实的原文逐句润顺，不新增任何事实"),
)
TASK_TITLE = {source: title for source, _, title, _ in TASK_SPECS}
TASK_ACTION = {source: action for source, action, _, _ in TASK_SPECS}
TASK_DUTY = {source: duty for source, _, _, duty in TASK_SPECS}
TASK_ORDER: Tuple[str, ...] = tuple(source for source, _, _, _ in TASK_SPECS)


# ---- 三块任务的提示词（边界写在提示里，再用下面的校验兜底）------------------

DRAFT_SYSTEM_PROMPT = (
    "你是简历写作教练。用户会给你一份「规则系统已经核验过」的写作简报 JSON：里面有 JD 要求"
    "（requirements，带 jd_ref 与 evidence_level）、已核验事实（verified_facts，带 ref 与 level）、"
    "被挡掉的内容（blocked_facts）以及风险（risks）。你只做一件事：写一版可编辑的简历写法（3~5 句，中文）。\n"
    "硬性要求：\n"
    "1) 只能使用 verified_facts 里出现的事实、数字与说法，一个字都不能新增或放大；\n"
    "2) level 是「仅提及」的条目，只能用「了解 / 接触过 / 用过」这类词，不许写成「熟练 / 精通 / 主导」；\n"
    "3) level 是「有动作」的条目，写清做了什么，不要编结果；有数字的条目照抄数字，不要改小数点；\n"
    "4) blocked_facts 里的内容一律不许写成「做过」；\n"
    "5) 不要输出「值得写 / 不值得写」这类判断，也不要复述覆盖率和结论；\n"
    "6) 只输出严格 JSON：{\"draft\": \"...\"}。"
)

INTERVIEW_SYSTEM_PROMPT = (
    "你是技术面试官教练。用户会给你一份「规则系统已经核验过」的写作简报 JSON：里面有 JD 要求"
    "（带 jd_ref 与 evidence_level）、已核验事实（带 ref 与 level）、被挡掉的内容（blocked_facts）"
    "以及风险（risks）。你只做一件事：预演面试官最可能追问的 3 个问题。\n"
    "硬性要求：\n"
    "1) 每个问题必须针对 JSON 里已有的要求、缺口、反向证据或风险，越具体越好，别问泛泛的「介绍一下你自己」；\n"
    "2) 每条给出 target（对着哪条要求或哪句原文，带上 JSON 里的出处写法）"
    "与 prepare（一句话：候选人该怎么准备，只能基于已核验事实）；\n"
    "3) 不许新增 JSON 之外的事实或数字，也不许替候选人编造经历；\n"
    "4) 不要重新判断「值不值得写」；\n"
    "5) 只输出严格 JSON：{\"questions\": [{\"question\": \"...\", \"target\": \"...\", \"prepare\": \"...\"}]}。"
)

POLISH_SYSTEM_PROMPT = (
    "你是简历文字编辑。用户会给你一份「规则系统已经核验过」的写作简报 JSON，里面有已核验事实"
    "（verified_facts：ref / level / text）。你只做一件事：把这些原文逐句润顺，给出可以直接粘进简历的写法。\n"
    "硬性要求：\n"
    "1) 一句原文对一句润色：ref 与 original 必须与 JSON 里一模一样，不许合并、不许漏、不许加条；\n"
    "2) 信息不增不减：不许新增事实、数字、公司、时间、规模，也不许删掉已有的数字；\n"
    "3) level 是「仅提及」的，润色后仍要弱（「了解 / 接触过」），不许升级成「熟练 / 精通 / 负责」；\n"
    "4) 不要写形容词堆砌的营销文案，也不要输出「值得写 / 不值得写」这类判断；\n"
    "5) 只输出严格 JSON：{\"items\": [{\"ref\": \"...\", \"original\": \"...\", \"polished\": \"...\"}]}。"
)

TASK_PROMPT = {
    SOURCE_LLM_SUGGEST: DRAFT_SYSTEM_PROMPT,
    SOURCE_LLM_INTERVIEW: INTERVIEW_SYSTEM_PROMPT,
    SOURCE_LLM_POLISH: POLISH_SYSTEM_PROMPT,
}


# ---- 送进模型的那份 JSON（严格来自规则层）-----------------------------------

def _jd_source(brief: PolishBrief) -> str:
    """JD 的来源写法：优先用调用方给的来源行，其次用岗位自己的来源文件。"""
    if brief.source_line:
        return brief.source_line
    return brief.posting.source_file if brief.posting else ""


def build_brief_payload(brief: PolishBrief) -> Dict[str, Any]:
    """把写作简报整理成给模型看的 JSON（不含 JD / 简历全文，只有规则挑出来的句子与出处）。"""
    return {
        "stage": "polish-brief-ready",
        "jd": {"title": brief.jd_name, "source": _jd_source(brief)},
        "resume": {"title": brief.doc.name, "source": brief.doc.source_file},
        "scope": "放宽" if brief.summary.relaxed else "严格",
        "requirements": [
            {
                "name": link.name,
                "level": link.requirement.level_label,
                "jd_ref": link.requirement.ref,
                "quote": _clip(link.requirement.quote, MAX_QUOTE_CHARS),
                "evidence_level": link.level_label,
                "evidence_refs": link.refs,
                "missing_parts": link.subs_missing,
            }
            for link in brief.core_links[:MAX_REQUIREMENTS]
        ],
        "verified_facts": [
            {
                "ref": fact.ref,
                "level": fact.level_label,
                "text": _clip(fact.text, MAX_FACT_CHARS),
                "supports": fact.capability_names,
            }
            for fact in brief.usable_facts[:MAX_FACTS]
        ],
        "blocked_facts": [
            {
                "ref": fact.ref,
                "text": _clip(fact.text, MAX_FACT_CHARS),
                "why": _clip(fact.block_reason, MAX_QUOTE_CHARS),
            }
            for fact in brief.blocked_facts[:MAX_FACTS]
        ],
        "coverage": {
            "matched": len(brief.matched),
            "total": len(brief.core_links),
            "ratio": round(brief.coverage, 3),
        },
        "gaps": [link.name for link in brief.gaps[:MAX_REQUIREMENTS]],
        "risks": [
            {"kind": risk.kind, "text": _clip(risk.text, MAX_FACT_CHARS), "refs": risk.refs[:MAX_REFS]}
            for risk in brief.risks()
        ],
    }


def build_user_message(brief: PolishBrief) -> str:
    """三块任务共用的 user 消息：同一份简报，不同的系统提示词。"""
    return "规则系统已经定稿的写作简报 JSON：\n" + json.dumps(
        build_brief_payload(brief), ensure_ascii=False
    )


# ---- 报告与单块任务 ---------------------------------------------------------

def new_report(budget: CallBudget) -> LlmReport:
    """建一份报告账本：调用上限 / 超时 / 重试都跟着调用账本走。"""
    return LlmReport(
        enabled=True,
        model=budget.model,
        call_limit=budget.limit,
        timeout=budget.timeout,
        retries=budget.retries,
    )


def unavailable_report(model: str, note: str) -> LlmReport:
    """没有 Key：三块都标「未启用」，一次请求都不发。"""
    report = LlmReport(enabled=True, model=model, note=note)
    report.blocks = [
        LlmBlock(source=source, title=TASK_TITLE[source], model=model, enabled=False, error=note)
        for source in TASK_ORDER
    ]
    return report


def failed_report(model: str, note: str) -> LlmReport:
    """整层出问题（不该发生）时的兜底：三块都标「调用失败」，简报照出。"""
    report = LlmReport(enabled=True, model=model, note=note)
    report.blocks = [
        LlmBlock(source=source, title=TASK_TITLE[source], model=model, enabled=True, error=note, failed=True)
        for source in TASK_ORDER
    ]
    return report


def run_task(report: LlmReport, budget: CallBudget, user: str, source: str) -> LlmBlock:
    """跑一块任务并把它挂进报告（按调用顺序追加，方便 Trace 一步步落）。"""
    block = LlmBlock(source=source, title=TASK_TITLE[source], model=budget.model)
    before = budget.calls
    messages = [
        {"role": "system", "content": TASK_PROMPT[source]},
        {"role": "user", "content": user},
    ]
    try:
        reply = budget.ask(messages)
    except LLMBudgetError as exc:
        block.calls = budget.calls - before
        block.error = str(exc)
        block.skipped = True
    except Exception as exc:                         # 失败只降级：这一块没内容，别的块照跑
        block.calls = budget.calls - before
        block.error = f"调用失败：{exc}"
        block.failed = True
    else:
        block.calls = budget.calls - before
        _PARSERS[source](block, reply)
    report.blocks.append(block)
    report.calls = budget.calls
    return block


def task_observation(block: LlmBlock, report: LlmReport) -> str:
    """这一步的 Observation：真实发生了什么，失败也照样写清楚。"""
    duty = TASK_DUTY.get(block.source, block.title)
    if block.skipped:
        return f"{duty}：没有调用（{block.error}），该块标「已达调用上限」"
    if block.error and not block.calls:
        return f"{duty}：没有发出调用（{block.error}）"
    if block.error:
        return (
            f"{duty}：调用 {block.calls} 次后{block.status}（{block.error}）；"
            "其它两块不受影响，简报里的规则内容也没有被替换"
        )
    return (
        f"{duty}：产出 {block.summary()}（source={block.source}，模型 {block.model}）；"
        "内容标 [LLM]，与规则写的那几段严格分开"
    )


def task_state_update(block: LlmBlock, report: LlmReport) -> str:
    return (
        f"state.llm.blocks[{block.source}] = {block.status}"
        f"（本次累计调用 {report.calls}/{report.call_limit}）"
    )


# ---- 三块内容的解析与校验 ----------------------------------------------------

_PARSERS: Dict[str, Any] = {}


def parse_draft(block: LlmBlock, reply: str) -> None:
    """建议写法：整段文本，越界（出现结论词）就整块丢弃。"""
    payload = parse_json_reply(reply)
    text = ""
    if isinstance(payload, dict):
        text = _clean(payload.get("draft") or payload.get("text"), MAX_DRAFT_CHARS)
    text = text or _clean(reply, MAX_DRAFT_CHARS)
    if not text:
        block.error = "模型没有返回可用的写法草稿"
        return
    hit = verdict_word(text)
    if hit:
        block.discarded = True
        block.error = f"生成内容含结论性表述「{hit}」，按「LLM 不做结论」已丢弃"
        return
    block.text = text


def parse_interview(block: LlmBlock, reply: str) -> None:
    """追问预演：最多 3 条，每条带 target 与 prepare。"""
    payload = parse_json_reply(reply)
    items: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        for raw in _as_list(payload.get("questions")):
            item = _interview_item(raw)
            if item:
                items.append(item)
    if not items:
        items = _lines_as_questions(reply)
    block.items = items[:INTERVIEW_COUNT]
    if not block.items:
        block.error = "模型没有返回可追问的问题"
        return
    hit = verdict_word(" ".join(item["question"] for item in block.items))
    if hit:
        block.items = []
        block.discarded = True
        block.error = f"生成内容含结论性表述「{hit}」，按「LLM 不做结论」已丢弃"


def parse_polish(block: LlmBlock, reply: str) -> None:
    """润色：一句原文对一句润色；模型只给整段文字时退化成一条。"""
    payload = parse_json_reply(reply)
    items: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        for raw in _as_list(payload.get("items") or payload.get("polished")):
            item = _polish_item(raw)
            if item:
                items.append(item)
    if not items and isinstance(payload, dict):
        text = _clean(payload.get("text"), MAX_POLISH_CHARS)
        if text:
            items.append({"ref": "", "original": "", "polished": text})
    if not items:
        text = _clean(reply, MAX_POLISH_CHARS)
        if text:
            items.append({"ref": "", "original": "", "polished": text})
    if not items:
        block.error = "模型没有返回可用的润色结果"
        return
    joined = " ".join(item["polished"] for item in items)
    hit = verdict_word(joined)
    if hit:
        block.items = []
        block.discarded = True
        block.error = f"润色结果含结论性表述「{hit}」，按「LLM 不做结论」已丢弃"
        return
    block.items = items


_PARSERS.update(
    {
        SOURCE_LLM_SUGGEST: parse_draft,
        SOURCE_LLM_INTERVIEW: parse_interview,
        SOURCE_LLM_POLISH: parse_polish,
    }
)


# ---- 小工具（与主线 generate.py 同一套清洗口径）-----------------------------

def _interview_item(raw: Any) -> Optional[Dict[str, str]]:
    if isinstance(raw, dict):
        question = _clean(_pick(raw, ("question", "q", "ask")), MAX_LINE_CHARS)
        target = _clean(_pick(raw, ("target", "ref", "focus", "requirement")), MAX_LINE_CHARS)
        prepare = _clean(_pick(raw, ("prepare", "how", "hint", "answer")), MAX_LINE_CHARS)
    else:
        question, target, prepare = _clean(raw, MAX_LINE_CHARS), "", ""
    if not question:
        return None
    return {"question": question, "target": target, "prepare": prepare}


def _polish_item(raw: Any) -> Optional[Dict[str, str]]:
    if not isinstance(raw, dict):
        text = _clean(raw, MAX_POLISH_CHARS)
        return {"ref": "", "original": "", "polished": text} if text else None
    polished = _clean(_pick(raw, ("polished", "text", "rewrite")), MAX_POLISH_CHARS)
    if not polished:
        return None
    return {
        "ref": _clean(_pick(raw, ("ref", "source", "from")), MAX_LINE_CHARS),
        "original": _clean(_pick(raw, ("original", "before", "source_text")), MAX_FACT_CHARS),
        "polished": polished,
    }


def _lines_as_questions(reply: str) -> List[Dict[str, str]]:
    """兜底：模型没给 JSON 时，按「问号结尾」的行凑 3 条。"""
    items: List[Dict[str, str]] = []
    for raw in str(reply or "").splitlines():
        line = _clean(raw, MAX_LINE_CHARS)
        if not line or not line.endswith(("？", "?")):
            continue
        items.append({"question": line, "target": "", "prepare": ""})
        if len(items) >= INTERVIEW_COUNT:
            break
    return items


def _pick(raw: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value)
    return ""


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clean(value: Any, limit: int) -> str:
    """压成一行、去掉引号与列表符号（与主线 generate.py 的清洗口径一致）。"""
    text = " ".join(str(value or "").split())
    text = text.strip().strip("\"'“”「」")
    text = re.sub(r"^(?:[-*+•·]|\d{1,2}\s*[.、)])\s*", "", text)
    return _clip(text, limit)


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"
