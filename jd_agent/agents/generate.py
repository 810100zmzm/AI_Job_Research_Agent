"""报告生成之后的 LLM 装饰层（可选，DeepSeek）：只做三件规则做不到的事。

    1) 建议写法草稿生成   source = llm-suggest   —— 用规则已核验的证据拼一版写法草稿
    2) 面试追问预演       source = llm-interview —— 针对风险与缺口，预演最可能被追问的 3 个问题
    3) 报告自然语言润色   source = llm-polish    —— 把规则写出的句子润顺，不增不减信息

三条红线（每条都有对应测试，可以直接逐条检查）：
  * 规则优先：词库能匹配的绝不调 LLM；v1.1 不做语义兜底；LLM 只在「规则报告已生成」之后介入，
    而且只读规则已经产出的结论 JSON，不读 JD / 项目原文；
  * 来源可区分：每块内容都带 source（llm-suggest / llm-interview / llm-polish），报告里标 [LLM]，
    与 [规则] 段落严格分开；LLM 输出的结论性表述会被丢弃；
  * 失败可降级：任何异常都只变成该块的 error +「调用失败」，规则报告与退出码不受影响；
    不带 --llm 时这一层根本不会被创建。

三条调用纪律（常量，可直接检查）：单次超时 30 秒、失败重试 1 次、一次运行最多 10 次调用。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.llm import (
    DEFAULT_CALL_LIMIT,
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_RETRIES,
    LLMError,
    OpenAICompatClient,
    parse_json_reply,
)
from ..core.schema import (
    SOURCE_LLM_INTERVIEW,
    SOURCE_LLM_POLISH,
    SOURCE_LLM_SUGGEST,
    AgentState,
    LlmBlock,
    LlmReport,
)

INTERVIEW_COUNT = 3          # 追问预演固定 3 条
MAX_DRAFT_CHARS = 300        # 草稿长度上限（超了截断，不让模型无限发挥）
MAX_POLISH_CHARS = 420
MAX_ITEM_CHARS = 120
MAX_TARGET_CHARS = 60
MAX_FACT_CHARS = 160
MAX_FACTS = 8                # 送给模型的「已核验事实」最多几条

# 面试追问预演暂时闲置：用户对当前效果不满意，后续会改版。
# 改回 True 即恢复调用（提示词 INTERVIEW_SYSTEM_PROMPT 与解析逻辑都还在）。
INTERVIEW_ENABLED = False

# 「LLM 不做结论」：出现这些词就说明模型越界了，该块内容直接丢弃
VERDICT_WORDS = ("不值得写", "暂不建议写", "不建议写", "建议改写", "值得写", "值得投", "不要写", "别写")

# source -> (Trace 里的 Action 名, 报告里的标题, 一句话职责)
TASK_SPECS = (
    (SOURCE_LLM_SUGGEST, "GenerateDraft", "建议写法草稿", "用规则已核验的证据拼一版可编辑的简历写法草稿"),
    (SOURCE_LLM_INTERVIEW, "RehearseInterview", "面试追问预演", "针对风险与缺口预演最可能被追问的 3 个问题"),
    (SOURCE_LLM_POLISH, "PolishReport", "报告润色", "把规则写出的句子润顺，不增不减信息"),
)
TASK_TITLE = {source: title for source, _, title, _ in TASK_SPECS}
TASK_ACTION = {source: action for source, action, _, _ in TASK_SPECS}
TASK_DUTY = {source: duty for source, _, _, duty in TASK_SPECS}
IDLE_REASON = "按用户要求暂时闲置（待改版），本次不调用模型"
TASK_ENABLED = {
    SOURCE_LLM_SUGGEST: True,
    SOURCE_LLM_INTERVIEW: INTERVIEW_ENABLED,
    SOURCE_LLM_POLISH: True,
}


class LLMBudgetError(LLMError):
    """本次运行的调用预算用完：不是调用失败，报告里单独标「已达调用上限」。"""


# ---- 三条任务的提示词（都把边界写在提示里，再靠下面的校验兜底）----------------

DRAFT_SYSTEM_PROMPT = (
    "你是简历写作教练。用户会给你一份「规则系统已经定稿」的报告 JSON（含结论档位、覆盖率、"
    "理由、风险、缺口与已核验事实）。你只做一件事：写一版可编辑的简历项目写法草稿（3~5 句）。\n"
    "硬性要求：\n"
    "1) 只能使用 JSON 里出现的事实、数字与出处，一个字都不能新增或放大；\n"
    "2) 不要输出「值得写 / 不值得写」这类判断，也不要改动结论档位；\n"
    "3) 只输出严格 JSON：{\"draft\": \"...\"}。"
)

INTERVIEW_SYSTEM_PROMPT = (
    "你是技术面试官教练。用户会给你一份「规则系统已经定稿」的报告 JSON（含结论档位、覆盖率、"
    "理由、风险、缺口与已核验事实）。你只做一件事：预演面试官最可能追问的 3 个问题。\n"
    "硬性要求：\n"
    "1) 每个问题必须针对 JSON 里已有的证据、缺口或风险，越具体越好；\n"
    "2) 每条给出 target（这条追问对着哪条要求或哪句原文，尽量带上 JSON 里的出处）"
    "和 prepare（一句话：候选人该怎么准备）；\n"
    "3) 不要新增 JSON 之外的事实或数字，不要重新判断「值不值得写」；\n"
    "4) 只输出严格 JSON：{\"questions\": [{\"question\": \"...\", \"target\": \"...\", \"prepare\": \"...\"}]}。"
)

POLISH_SYSTEM_PROMPT = (
    "你是技术报告编辑。用户会给你一份「规则系统已经定稿」的报告 JSON，里面的理由、风险与建议写法"
    "是规则拼出来的，读起来很生硬。你只做一件事：把这些句子润成一段通顺、克制的自然语言"
    "（用于自述或复盘），不要写形容词堆砌的营销文案。\n"
    "硬性要求：\n"
    "1) 信息不增不减：不许新增事实、数字、公司、时间，也不许删掉已有的出处；\n"
    "2) 不要给出「值得写 / 不值得写」这类判断，也不要复述结论档位；\n"
    "3) 只输出严格 JSON：{\"text\": \"...\"}。"
)

TASK_PROMPT = {
    SOURCE_LLM_SUGGEST: DRAFT_SYSTEM_PROMPT,
    SOURCE_LLM_INTERVIEW: INTERVIEW_SYSTEM_PROMPT,
    SOURCE_LLM_POLISH: POLISH_SYSTEM_PROMPT,
}


class CallBudget:
    """调用账本：超时、重试、次数上限都在这里记账，三块任务都不能绕过它直接调模型。"""

    def __init__(
        self,
        client: OpenAICompatClient,
        model: str,
        limit: int = DEFAULT_CALL_LIMIT,
        timeout: float = DEFAULT_CALL_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        self.client = client
        self.model = model
        self.limit = max(1, int(limit))
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.calls = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.calls)

    def ask(self, messages: Sequence[Dict[str, Any]], temperature: float = 0.3) -> str:
        """发一次请求；失败重试 retries 次；超预算直接抛 LLMError（不发请求）。"""
        last: Optional[Exception] = None
        for _ in range(self.retries + 1):
            if self.calls >= self.limit:
                raise LLMBudgetError(f"本次运行已达调用上限（{self.limit} 次），跳过剩余生成任务")
            self.calls += 1
            try:
                return self.client.chat(messages, model=self.model, temperature=temperature)
            except LLMError as exc:
                last = exc
            except Exception as exc:            # 可选层不能带崩主流程
                last = LLMError(f"{exc.__class__.__name__}: {exc}")
        detail = f"调用失败（已重试 {self.retries} 次）：{last}" if self.retries else f"调用失败：{last}"
        raise LLMError(detail)


def build_report_payload(state: AgentState) -> Dict[str, Any]:
    """把规则已经生成的内容整理成给模型看的 JSON（不含 JD / 项目全文）。"""
    verdict = state.verdict
    facts = [
        {"ref": fact.ref, "text": _clip(fact.text, MAX_FACT_CHARS), "level": fact.level_label}
        for fact in state.project.actionable_facts[:MAX_FACTS]
    ]
    return {
        "stage": "rule-report-ready",
        "sufficiency": state.sufficiency,
        "jd": {"title": state.jd.name, "source_file": state.jd.source_file},
        "project": {"title": state.project.name, "source_file": state.project.source_file},
        "call": verdict.call if verdict else "",
        "headline": verdict.headline if verdict else "",
        "stop_reason": verdict.stop_reason if verdict else "",
        "coverage": round(state.coverage, 3),
        "core_requirement_count": len(state.core_requirements),
        "solid_match_count": len(state.solid),
        "result_level_count": state.result_match_count,
        "reasons": [
            {
                "text": reason.text,
                "requirement": reason.requirement_name,
                "jd_ref": reason.jd_ref,
                "project_refs": reason.project_refs,
            }
            for reason in (verdict.reasons if verdict else [])
        ],
        "risks": [
            {"kind": risk.kind, "text": risk.text, "refs": risk.refs}
            for risk in (verdict.risks if verdict else [])
        ],
        "rewrite": verdict.rewrite if verdict else "",
        "missing_requirements": [match.requirement_name for match in state.missing],
        "evidence": [
            {
                "requirement": match.requirement_name,
                "requirement_level": match.requirement.level_label,
                "level": match.level_label,
                "note": match.note,
            }
            for match in state.matches
        ],
        "verified_facts": facts,
    }


def unavailable_llm_report(model: str, note: str) -> LlmReport:
    """没有 Key：三块都标「未启用」，一次请求都不发。"""
    report = LlmReport(enabled=True, model=model, note=note)
    report.blocks = [
        LlmBlock(source=source, title=title, model=model, enabled=False, error=note)
        for source, _, title, _ in TASK_SPECS
    ]
    return report


def failed_llm_report(model: str, note: str) -> LlmReport:
    """装饰层整体出问题（不该发生）时的兜底：三块都标「调用失败」，主报告照出。"""
    report = LlmReport(enabled=True, model=model, note=note)
    report.blocks = [
        LlmBlock(source=source, title=title, model=model, enabled=True, error=note, failed=True)
        for source, _, title, _ in TASK_SPECS
    ]
    return report


def generate_llm_report(state: AgentState, budget: CallBudget) -> LlmReport:
    """报告已生成之后：依次跑三块生成任务；单块失败只影响这一块。"""
    user = "规则系统已经定稿的报告 JSON：\n" + json.dumps(
        build_report_payload(state), ensure_ascii=False
    )
    report = LlmReport(
        enabled=True,
        model=budget.model,
        call_limit=budget.limit,
        timeout=budget.timeout,
        retries=budget.retries,
    )
    report.blocks = [_run_task(budget, user, source) for source in TASK_TITLE]
    report.calls = budget.calls
    return report


def _run_task(budget: CallBudget, user: str, source: str) -> LlmBlock:
    block = LlmBlock(source=source, title=TASK_TITLE[source], model=budget.model)
    if not TASK_ENABLED.get(source, True):
        block.idle = True
        block.notes = [IDLE_REASON]
        return block
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
        return block
    except LLMError as exc:
        block.calls = budget.calls - before
        block.error = str(exc)
        block.failed = True
        return block
    block.calls = budget.calls - before
    _PARSERS[source](block, reply)
    return block


# ---- 三块内容的解析与校验 ----------------------------------------------------

def parse_draft(block: LlmBlock, reply: str) -> None:
    payload = parse_json_reply(reply)
    text = ""
    if isinstance(payload, dict):
        text = _clean(payload.get("draft") or payload.get("text"), MAX_DRAFT_CHARS)
    text = text or _clean(reply, MAX_DRAFT_CHARS)
    if not text:
        block.error = "模型没有返回可用的草稿"
        return
    clean, hit = _guard(text)
    if hit:
        block.discarded = True
        block.error = f"生成内容含结论性表述「{hit}」，按「LLM 不做结论」已丢弃"
        return
    block.text = clean


def parse_interview(block: LlmBlock, reply: str) -> None:
    payload = parse_json_reply(reply)
    items: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        for raw in _as_list(payload.get("questions")):
            item = _interview_item(raw)
            if item:
                items.append(item)
    if not items:
        items = _interview_from_lines(reply)
    block.items = items[:INTERVIEW_COUNT]
    if not block.items:
        block.error = "模型没有返回可追问的问题"
        return
    joined = " ".join(item["question"] for item in block.items)
    hit = verdict_word(joined)
    if hit:
        block.items = []
        block.discarded = True
        block.error = f"生成内容含结论性表述「{hit}」，按「LLM 不做结论」已丢弃"


def parse_polish(block: LlmBlock, reply: str) -> None:
    payload = parse_json_reply(reply)
    text = ""
    if isinstance(payload, dict):
        text = _clean(payload.get("text") or payload.get("polished"), MAX_POLISH_CHARS)
    text = text or _clean(reply, MAX_POLISH_CHARS)
    if not text:
        block.error = "模型没有返回可用的润色文本"
        return
    clean, hit = _guard(text)
    if hit:
        block.discarded = True
        block.error = f"润色结果含结论性表述「{hit}」，按「LLM 不做结论」已丢弃"
        return
    block.text = clean


_PARSERS = {
    SOURCE_LLM_SUGGEST: parse_draft,
    SOURCE_LLM_INTERVIEW: parse_interview,
    SOURCE_LLM_POLISH: parse_polish,
}


def _guard(text: str) -> Tuple[str, str]:
    """返回 (清理后的文本, 命中的结论词)；命中时调用方应丢弃这段内容。"""
    clean = _clip(text, MAX_ITEM_CHARS * 4)
    return clean, verdict_word(text)


def verdict_word(text: str) -> str:
    """LLM 输出里是否出现了「LLM 不做结论」禁止的档位词；没有则返回空串。"""
    raw = str(text or "")
    for word in VERDICT_WORDS:
        if word in raw:
            return word
    return ""


def _interview_item(raw: Any) -> Optional[Dict[str, str]]:
    if isinstance(raw, dict):
        question = _clean(_pick(raw, ("question", "q", "ask")), MAX_ITEM_CHARS)
        target = _clean(_pick(raw, ("target", "ref", "focus", "requirement")), MAX_TARGET_CHARS)
        prepare = _clean(_pick(raw, ("prepare", "how", "hint", "answer")), MAX_ITEM_CHARS)
    else:
        question, target, prepare = _clean(raw, MAX_ITEM_CHARS), "", ""
    if not question:
        return None
    return {"question": question, "target": target, "prepare": prepare}


def _interview_from_lines(reply: str) -> List[Dict[str, str]]:
    """兜底：模型没给 JSON 时，按「问号结尾」的行凑 3 条。"""
    items: List[Dict[str, str]] = []
    for raw in str(reply or "").splitlines():
        line = _clean(raw, MAX_ITEM_CHARS)
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
    text = " ".join(str(value or "").split())
    text = text.strip().strip("\"'“”「」")
    text = re.sub(r"^(?:[-*+•·]|\d{1,2}\s*[.、)])\s*", "", text)
    return _clip(text, limit)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
