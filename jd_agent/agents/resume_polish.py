"""Rule-first resume polishing and Markdown / HTML resume generation.

The pipeline deliberately keeps the model on a short leash:

* only facts that passed ``resume_facts`` screening are eligible for polishing;
* known wording is handled by a deterministic dictionary and never sent to an LLM;
* every returned comparison item is labelled ``source=llm-polish`` so the
  frontend can separate model-assisted output from the original resume;
* model failures or unsafe rewrites fall back to the original sentence and
  never block Markdown / HTML generation.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.llm import parse_json_reply
from ..core.schema import (
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_RESULT,
    SOURCE_LLM_POLISH,
)
from ..domain.jd import clean_text
from ..domain.project import ACTION_MARKERS, DELIVERY_MARKERS, METRIC_RE, OWNERSHIP_MARKERS
from ..domain.resume_facts import (
    ROLE_META,
    ResumeFact,
    extract_resume,
)
from ..tools.resume import Node, Resume, parse_resume, render_html, render_markdown
from ..tools.resume_styles import DEFAULT_STYLE, all_styles, get_style

MAX_LLM_FACTS = 10
MAX_FACT_CHARS = 240
MAX_POLISH_CHARS = 280
MAX_POLISH_RATIO = 1.8
MAX_LLM_CALLS = 3


@dataclass(frozen=True)
class PolishLevel:
    key: str
    name: str
    summary: str
    guidance: str


@dataclass(frozen=True)
class TermMapping:
    source: str
    l2: str
    l3: str
    boundary: str = ""

    def replacement(self, level: str) -> str:
        return self.l3 if level == "L3" else self.l2


POLISH_LEVELS: Tuple[PolishLevel, ...] = (
    PolishLevel(
        key="L1",
        name="通顺",
        summary="只改语法、标点和冗余连接，不加词义，不换动词或术语。",
        guidance="保留事实与原始措辞；只做基本语法清理，不换动词、不换术语、不加形容词、不前置结果。",
    ),
    PolishLevel(
        key="L2",
        name="专业",
        summary="术语化、动词升级、结构对仗，不改变事实。",
        guidance="允许换动词、换术语和做轻度结果前置；不新增事实、数字、职责、成果或证据强度。",
    ),
    PolishLevel(
        key="L3",
        name="亮眼",
        summary="结果前置、行业术语、主动语态，强化但不夸大。",
        guidance="允许结果前置、行业术语、主动语态和基于原文数字计算百分比；严禁升级责任或能力等级。",
    ),
)
POLISH_LEVEL_BY_KEY = {item.key: item for item in POLISH_LEVELS}

# 分级术语映射表既驱动规则词典，也原样进入 L2 / L3 的 Prompt。
TERM_MAPPING: Tuple[TermMapping, ...] = (
    TermMapping("用了", "使用", "基于", "只替换一般动作词"),
    TermMapping("用", "使用", "基于", "仅后接技术或工具名时替换"),
    TermMapping("使用", "采用", "基于", "前面有熟练/熟悉/掌握/了解时不换"),
    TermMapping("写了", "编写", "实现", "不改产物类型"),
    TermMapping("做了", "完成", "实现", "不补结果"),
    TermMapping("搭建", "构建", "工程化搭建", "不补设计或主导"),
    TermMapping("开发", "研发", "工程化研发", "不扩大职责范围"),
    TermMapping("调试", "排查", "调试排障", "不补修复结果"),
    TermMapping("测试", "验证", "测试验证", "不补上线或交付"),
    TermMapping("优化", "调优", "深度调优", "不补性能结论"),
    TermMapping("整理", "梳理", "结构化梳理", "不补数据规模"),
    TermMapping("分析", "分析", "深度分析", "不补分析结论"),
    TermMapping("跑", "执行", "运行", "不补运行结果"),
    TermMapping("对接", "联调", "接口联调", "不补上线状态"),
    TermMapping("接入", "集成", "工程化集成", "不补调用量"),
    TermMapping("调参", "参数调优", "系统调参", "不补效果提升"),
    TermMapping("协助", "配合", "协同支持", "不得升级为负责或主导"),
    TermMapping("帮助", "支持", "支撑", "不得升级为负责或主导"),
    TermMapping("负责", "负责", "负责推进", "不扩大职责范围"),
    TermMapping("参与", "参与", "参与", "绝对禁止改成主导"),
    TermMapping("了解", "了解", "了解", "绝对禁止改成精通"),
    TermMapping("熟悉", "熟悉", "熟悉", "不升级能力等级"),
    TermMapping("降到", "降低至", "降低至", "保留原数字与顺序"),
    TermMapping("下降到", "下降至", "下降至", "保留原数字与顺序"),
    TermMapping("减少到", "减少至", "减少至", "保留原数字与顺序"),
    TermMapping("提升到", "提升至", "提升至", "保留原数字与顺序"),
    TermMapping("提高到", "提高至", "提高至", "保留原数字与顺序"),
    TermMapping("增长到", "增长至", "增长至", "保留原数字与顺序"),
    TermMapping("增加到", "增加至", "增加至", "保留原数字与顺序"),
)

TERM_MAPPING_PROMPT = "术语映射表（只在事实、责任和能力等级不变时使用）：\n" + "\n".join(
    f"- {item.source} -> L2: {item.l2}；L3: {item.l3}（{item.boundary}）"
    for item in TERM_MAPPING
)

# L1 只允许语法清理，不换动词。
_RULE_PATTERNS: Dict[str, Tuple[Tuple[re.Pattern[str], str], ...]] = {
    "L1": (
        (re.compile(r"^\s*我(?:们)?\s*"), ""),
        (re.compile(r"能够"), "能"),
        (re.compile(r"主要是"), ""),
        (re.compile(r"\s+([，。；：、])"), r"\1"),
        (re.compile(r"([，。；：])\1+"), r"\1"),
    ),
    "L2": (
        (re.compile(r"^\s*我(?:们)?\s*"), ""),
        (re.compile(r"能够"), "能"),
        (re.compile(r"主要是"), ""),
        (re.compile(r"\s+([，。；：、])"), r"\1"),
        (re.compile(r"([，。；：])\1+"), r"\1"),
    ),
    "L3": (
        (re.compile(r"^\s*我(?:们)?\s*"), ""),
        (re.compile(r"能够"), "能"),
        (re.compile(r"主要是"), ""),
        (re.compile(r"\s+([，。；：、])"), r"\1"),
        (re.compile(r"([，。；：])\1+"), r"\1"),
    ),
}

POLISH_SYSTEM_PROMPT = (
    "你是严格的简历文字编辑。你只会收到规则系统已经核验过的简历事实，绝不能读取或补写"
    "未提供的内容。只做措辞润色，输出可直接放回简历的中文句子。\n"
    "绝对红线（所有等级）：\n"
    "1) 不许编造或改动任何数字、职责、成果、公司、学校、项目名、时间；\n"
    "2) 不许把「参与」写成「主导」，不许把「了解」写成「精通」；\n"
    "3) 不许新增原文没有的「负责 / 独立 / 主导 / 上线 / 交付」；\n"
    "4) 只输出严格 JSON：{\"items\":[{\"ref\":\"...\",\"original\":\"...\",\"polished\":\"...\"}]}，"
    "每条 ref 和 original 必须与输入完全一致，不能漏项、合并或新增；\n"
    "5) 不输出评价、建议、结论或解释。"
)

POLISH_LEVEL_PROMPTS: Dict[str, str] = {
    "L1": (
        POLISH_SYSTEM_PROMPT
        + "\n等级 L1（通顺）：只改语法，不加词义。允许删去「我 / 我们」、把「能够」缩成「能」、"
        + "清理标点与冗词；禁止换动词、换术语、加形容词、前置结果或改变句式重点。"
    ),
    "L2": (
        POLISH_SYSTEM_PROMPT
        + "\n等级 L2（专业）：允许术语化、动词升级、结构对仗和轻度结果前置，适合正式投递。"
        "只改表达，不改事实、数字、责任范围、公司 / 学校 / 项目名与证据强度。\n"
        + TERM_MAPPING_PROMPT
    ),
    "L3": (
        POLISH_SYSTEM_PROMPT
        + "\n等级 L3（亮眼）：结果前置、行业术语、主动语态、强化词优先。"
        "原文已有数字可以前置；百分比只能由原文已有数字数学推导，不得新增任何其他数字。"
        "仍不得把参与写成主导、了解写成精通，也不得增加职责或成果。\n"
        + TERM_MAPPING_PROMPT
    ),
}


@dataclass
class ResumePolishItem:
    ref: str
    original: str
    polished: str
    source: str = SOURCE_LLM_POLISH
    engine: str = "original"
    status: str = "保留原文"
    reason: str = ""

    @property
    def changed(self) -> bool:
        return self.original != self.polished


@dataclass
class ResumePolishStats:
    total: int = 0
    dictionary: int = 0
    llm: int = 0
    original: int = 0
    blocked: int = 0
    calls: int = 0


@dataclass
class ResumePolishResult:
    level: str
    style: str
    items: List[ResumePolishItem] = field(default_factory=list)
    markdown: str = ""
    html: str = ""
    notice: str = ""
    stats: ResumePolishStats = field(default_factory=ResumePolishStats)
    notes: List[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.notice)


def get_polish_level(key: str) -> PolishLevel:
    """Return a level or raise a user-facing ValueError for unknown input."""
    name = str(key or "").strip().upper()
    if name not in POLISH_LEVEL_BY_KEY:
        raise ValueError(f"没有这种表达强度：{key}（可选 L1、L2、L3）")
    return POLISH_LEVEL_BY_KEY[name]


def polish_system_prompt(level: str) -> str:
    """Return the level-specific system prompt used by the LLM path."""
    chosen = get_polish_level(level)
    return POLISH_LEVEL_PROMPTS[chosen.key]


def level_options() -> List[Dict[str, str]]:
    return [
        {"key": item.key, "name": item.name, "summary": item.summary}
        for item in POLISH_LEVELS
    ]


def style_options() -> List[Dict[str, str]]:
    return [
        {
            "key": item.key,
            "name": item.name,
            "summary": item.summary,
            "label": item.label,
        }
        for item in all_styles()
    ]


def polish_resume(
    text: str,
    *,
    level: str = "L2",
    style: str = DEFAULT_STYLE,
    source_file: str = "resume.md",
    text_client: Optional[Any] = None,
    max_llm_calls: int = MAX_LLM_CALLS,
    enable_polish: bool = True,
) -> ResumePolishResult:
    """Polish verified facts and render a styled Markdown / HTML resume.

    ``text_client`` is optional. When absent, or when every eligible fact is
    handled by the dictionary, no LLM call is attempted.

    Set ``enable_polish=False`` to generate from the original resume without
    applying dictionary rules or invoking an LLM.
    """
    chosen_level = get_polish_level(level)
    chosen_style = get_style(style)
    if not enable_polish:
        original_resume = parse_resume(text, source_file=source_file)
        original_resume.style = chosen_style.key
        return ResumePolishResult(
            level=chosen_level.key,
            style=chosen_style.key,
            markdown=render_markdown(original_resume),
            html=render_html(original_resume),
            notice="仅生成简历，未启用润色。原文保持不变。",
            notes=list(original_resume.notes),
        )

    doc = extract_resume(text, source_file=source_file)
    eligible = [
        fact
        for fact in doc.facts
        if fact.usable and fact.role != ROLE_META and fact.text.strip()
    ]
    blocked = [fact for fact in doc.facts if not fact.usable]

    items: List[ResumePolishItem] = []
    pending: List[ResumeFact] = []
    for fact in eligible:
        item = _dictionary_item(fact, chosen_level.key)
        if item is not None:
            items.append(item)
        else:
            pending.append(fact)

    stats = ResumePolishStats(total=len(eligible), blocked=len(blocked))
    calls = 0
    if pending and text_client is not None and max_llm_calls > 0:
        llm_items, calls, error = _llm_items(pending, chosen_level, text_client, max_llm_calls)
        items.extend(llm_items)
        if error:
            _mark_llm_failure(llm_items, error)
    elif pending:
        for fact in pending:
            items.append(_fallback_item(fact, "润色未启用", "没有可用的文本模型；已保留原文"))

    # Keep comparison output in original resume order.
    order = {fact.ref: index for index, fact in enumerate(eligible)}
    items.sort(key=lambda item: order.get(item.ref, len(order)))

    stats.dictionary = sum(1 for item in items if item.engine == "dictionary")
    stats.llm = sum(1 for item in items if item.engine == "llm")
    stats.original = sum(1 for item in items if item.engine == "original")
    stats.calls = calls

    replacements = {
        clean_text(item.original): item.polished
        for item in items
        if item.changed and item.polished.strip()
    }
    polished_resume = _apply_replacements(
        parse_resume(text, source_file=source_file),
        replacements,
        style=chosen_style.key,
    )
    markdown = render_markdown(polished_resume)
    html = render_html(polished_resume)

    notice = ""
    degraded_items = [item for item in items if item.engine == "original"]
    if degraded_items:
        notice = "润色未启用：未处理的条目已保留原文，Markdown 和 HTML 仍可正常生成。"
    elif not eligible:
        notice = "没有可核验的简历事实，已按原文生成简历。"
    elif blocked:
        notice = f"有 {len(blocked)} 条否定或背景内容未进入润色，已在生成简历中按原文保留。"
    elif not stats.llm and stats.dictionary:
        notice = "本次全部由规则词典完成润色，没有调用大模型。"

    notes = list(doc.notes)
    if replacements:
        notes.append(f"已应用 {len(replacements)} 条润色，事实与出处未改动")
    return ResumePolishResult(
        level=chosen_level.key,
        style=chosen_style.key,
        items=items,
        markdown=markdown,
        html=html,
        notice=notice,
        stats=stats,
        notes=notes,
    )


def _dictionary_item(fact: ResumeFact, level: str) -> Optional[ResumePolishItem]:
    candidate, hits = _apply_rules_with_hits(fact.text, level)
    if candidate == fact.text or not _safe_rewrite(fact, candidate, level):
        return None
    hit_text = "、".join(hits) if hits else "语法精简"
    return ResumePolishItem(
        ref=fact.ref,
        original=fact.text,
        polished=candidate,
        engine="dictionary",
        status="规则词典",
        reason=f"{level} 词典命中：{hit_text}；未调用大模型",
    )


def _fallback_item(fact: ResumeFact, status: str, reason: str) -> ResumePolishItem:
    return ResumePolishItem(
        ref=fact.ref,
        original=fact.text,
        polished=fact.text,
        engine="original",
        status=status,
        reason=reason,
    )


def _apply_rules(text: str, level: str) -> str:
    value, _ = _apply_rules_with_hits(text, level)
    return value


def _apply_rules_with_hits(text: str, level: str) -> Tuple[str, List[str]]:
    value = clean_text(str(text or "")).strip()
    hits: List[str] = []
    rules: Iterable[Tuple[re.Pattern[str], str]] = _RULE_PATTERNS.get(level, ())
    for pattern, replacement in rules:
        updated = pattern.sub(replacement, value)
        if updated != value and "语法精简" not in hits:
            hits.append("语法精简")
        value = updated

    if level == "L3":
        updated = _add_derived_percentage(value)
        if updated != value:
            hits.append("百分比推导")
        value = updated

    if level in ("L2", "L3"):
        for mapping in sorted(TERM_MAPPING, key=lambda item: len(item.source), reverse=True):
            replacement = mapping.replacement(level)
            if replacement == mapping.source:
                continue
            updated = _term_pattern(mapping.source).sub(replacement, value)
            if updated != value:
                hits.append(f"{mapping.source}->{replacement}")
            value = updated

    if level == "L3":
        updated = _frontload_metric_clause(value)
        if updated != value:
            hits.append("结果前置")
        value = updated

    value = re.sub(r"\s+", " ", value).strip()
    return value, hits


_TERM_NOMINAL_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "写了": ("编写", "实现"),
    "做了": ("完成", "实现"),
    "开发": ("者", "人员", "工程师", "团队", "部门", "岗位", "职位"),
    "调试": (
        "者", "人员", "工程师", "团队", "部门", "岗位", "职位", "排障",
        "日志", "信息", "工具", "环境",
    ),
    "测试": (
        "者", "人员", "工程师", "团队", "部门", "岗位", "职位", "验证",
        "用例", "案例", "报告", "结果", "数据", "集", "环境", "工具", "框架",
        "平台", "方案", "策略", "方法", "流程", "计划", "能力", "经验", "指标",
    ),
    "优化": (
        "者", "人员", "工程师", "团队", "部门", "岗位", "职位", "调优",
        "方案", "策略", "措施", "手段", "建议", "思路", "方法", "目标",
        "结果", "效果", "空间", "算法", "问题", "器",
    ),
    "整理": ("者", "人员", "工作", "流程", "方法", "工具", "方案", "报告", "结果"),
    "协助": ("者", "人员", "工程师", "团队", "部门", "岗位", "职位", "工作", "事项"),
    "帮助": ("文档", "中心", "信息", "功能", "方法", "页面", "选项", "内容", "人员"),
    "负责": (
        "人", "人员", "工程师", "团队", "部门", "岗位", "职位", "职责",
        "范围", "推进", "负责",
    ),
}


def _term_pattern(source: str) -> re.Pattern[str]:
    suffix = _TERM_NOMINAL_SUFFIXES.get(source, ())
    suffix_guard = (
        ""
        if not suffix
        else r"(?!" + "|".join(re.escape(item) for item in suffix) + r")"
    )
    if source == "分析":
        return re.compile(r"(?<!深度)" + re.escape(source) + suffix_guard)
    if source == "跑":
        return re.compile(
            r"(?<![\u4e00-\u9fff])跑"
            r"(?=了?\s*(?:[A-Za-z0-9]|模型|脚本|实验|任务|测试|数据|程序|流程|基线))"
        )
    if source == "用":
        return re.compile(r"(?<![\u4e00-\u9fff])用(?=\s*[A-Za-z0-9])")
    if source == "用了":
        return re.compile(r"(?<![\u4e00-\u9fff])用了(?=\s*[A-Za-z0-9])")
    if source == "使用":
        return re.compile(r"(?<![\u4e00-\u9fff])使用(?=\s*[A-Za-z0-9])")
    return re.compile(re.escape(source) + suffix_guard)


_METRIC_PAIR_RE = re.compile(
    r"(?P<prefix>从|由)\s*(?P<old>\d+(?:\.\d+)?)\s*"
    r"(?P<old_unit>%|％|ms|毫秒|秒|分钟|小时|万条|条|次|倍)?\s*"
    r"(?P<verb>降到|降低到|下降到|减少到|缩短到|提升到|提高到|增长到|增加到|涨到)\s*"
    r"(?P<new>\d+(?:\.\d+)?)\s*"
    r"(?P<new_unit>%|％|ms|毫秒|秒|分钟|小时|万条|条|次|倍)?",
    re.IGNORECASE,
)


def _add_derived_percentage(text: str) -> str:
    """L3 may add only a percentage mathematically derived from original numbers."""
    def replace(match: re.Match[str]) -> str:
        old_unit = (match.group("old_unit") or "").lower()
        new_unit = (match.group("new_unit") or "").lower()
        if old_unit != new_unit or old_unit in ("%", "％"):
            return match.group(0)
        old = float(match.group("old"))
        new = float(match.group("new"))
        if old == 0:
            return match.group(0)
        if match.group("verb") in ("降到", "降低到", "下降到", "减少到", "缩短到"):
            percent = (old - new) / old * 100
            label = "降低"
        else:
            percent = (new - old) / old * 100
            label = "提升"
        if percent <= 0:
            return match.group(0)
        return f"{match.group(0)}（{label} {_format_percentage(percent)}%）"

    return _METRIC_PAIR_RE.sub(replace, text)


def _format_percentage(value: float) -> str:
    formatted = f"{value:.1f}"
    return formatted[:-2] if formatted.endswith(".0") else formatted


def _frontload_metric_clause(text: str) -> str:
    clauses = re.findall(r"[^，,；;。]+[，,；;。]?", text)
    if len(clauses) < 2:
        return text
    contents = [clause.strip("，,；;。 ") for clause in clauses]
    metric_index = next(
        (index for index, clause in enumerate(contents) if METRIC_RE.search(clause)),
        -1,
    )
    if metric_index <= 0:
        return text
    moved = contents.pop(metric_index)
    rest = [clause for clause in contents if clause]
    ending = text[-1] if text and text[-1] in "。！？" else ""
    return "；".join([moved] + rest) + ending


def _llm_items(
    facts: Sequence[ResumeFact],
    level: PolishLevel,
    client: Any,
    max_calls: int,
) -> Tuple[List[ResumePolishItem], int, str]:
    items: List[ResumePolishItem] = []
    calls = 0
    error = ""
    for start in range(0, len(facts), MAX_LLM_FACTS):
        if calls >= max_calls:
            for fact in facts[start:]:
                items.append(_fallback_item(fact, "已达调用上限", "本轮 LLM 调用预算已用完"))
            break
        chunk = list(facts[start : start + MAX_LLM_FACTS])
        try:
            reply = client.chat(
                [
                    {"role": "system", "content": polish_system_prompt(level.key)},
                    {"role": "user", "content": _llm_payload(chunk, level)},
                ],
                max_tokens=1200,
                temperature=0.2,
            )
            calls += 1
            parsed = _parse_llm_items(reply)
        except Exception as exc:  # noqa: BLE001 - degrade per fact, never fail generation
            calls += 1
            error = f"润色未启用：LLM 调用失败（{exc}），已保留原文"
            items.extend(_fallback_item(fact, "调用失败", str(exc)) for fact in chunk)
            continue

        by_ref = {fact.ref: fact for fact in chunk}
        used: set[str] = set()
        for raw in parsed:
            ref = str(raw.get("ref") or "").strip()
            original = str(raw.get("original") or "").strip()
            polished = str(raw.get("polished") or "").strip()
            fact = by_ref.get(ref)
            if fact is None or ref in used or clean_text(original) != clean_text(fact.text):
                continue
            if not polished or len(polished) > MAX_POLISH_CHARS:
                continue
            if not _safe_rewrite(fact, polished, level.key):
                items.append(
                    ResumePolishItem(
                        ref=fact.ref,
                        original=fact.text,
                        polished=fact.text,
                        engine="original",
                        status="安全校验未通过",
                        reason="润色结果可能新增或夸大事实，已保留原文",
                    )
                )
                used.add(ref)
                continue
            items.append(
                ResumePolishItem(
                    ref=fact.ref,
                    original=fact.text,
                    polished=polished,
                    engine="llm",
                    status="LLM 润色",
                    reason=f"{level.key}：{level.guidance}",
                )
            )
            used.add(ref)
        for fact in chunk:
            if fact.ref not in used:
                items.append(_fallback_item(fact, "模型漏项", "模型没有返回这一条，已保留原文"))
    return items, calls, error


def _llm_payload(facts: Sequence[ResumeFact], level: PolishLevel) -> str:
    payload = {
        "level": level.key,
        "level_name": level.name,
        "guidance": level.guidance,
        "verified_facts": [
            {
                "ref": fact.ref,
                "level": fact.level_label,
                "text": fact.text,
            }
            for fact in facts[:MAX_LLM_FACTS]
        ],
    }
    if level.key in ("L2", "L3"):
        payload["term_mapping"] = [
            {
                "source": item.source,
                "replacement": item.replacement(level.key),
                "boundary": item.boundary,
            }
            for item in TERM_MAPPING
        ]
    return "只润色下面这些已核验事实：\n" + json.dumps(payload, ensure_ascii=False)


def _parse_llm_items(reply: str) -> List[Dict[str, str]]:
    payload = parse_json_reply(reply)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("items")
    if not isinstance(rows, list):
        return []
    result: List[Dict[str, str]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        result.append(
            {
                "ref": str(raw.get("ref") or "").strip(),
                "original": str(raw.get("original") or "").strip(),
                "polished": str(raw.get("polished") or "").strip(),
            }
        )
    return result


def _safe_rewrite(fact: ResumeFact, polished: str, level: str = "L2") -> bool:
    """Reject rewrites that change facts, strength, ownership, delivery, or numbers."""
    value = clean_text(polished).strip()
    original = clean_text(fact.text).strip()
    if not value or value == original:
        return False
    if len(value) > MAX_POLISH_CHARS or len(value) > int(len(original) * MAX_POLISH_RATIO) + 24:
        return False
    if _verdict_hit(value):
        return False
    if not _numbers_safe(original, value, level):
        return False
    if _ownership_signature(original) != _ownership_signature(value):
        return False
    if _competence_signature(original) != _competence_signature(value):
        return False
    if _entity_signature(original) != _entity_signature(value):
        return False

    original_actions = _action_signature(original)
    if fact.level == EVIDENCE_MENTION and _has_any(value, ACTION_MARKERS):
        return False
    if _action_signature(value) - original_actions:
        return False
    if fact.level == EVIDENCE_ACTION and _has_any(value, DELIVERY_MARKERS):
        return False
    if fact.level != EVIDENCE_RESULT and _has_delivery(value):
        return False
    if _delivery_signature(original) != _delivery_signature(value):
        return False
    if fact.level == EVIDENCE_MENTION and not _has_any(value, ("了解", "接触", "用过", "使用", "熟悉", "掌握")):
        return False
    if not original_actions and _has_any(value, ACTION_MARKERS):
        return False

    # A paraphrase may use synonyms, but it must not add a known capability.
    if _new_capabilities(original, value):
        return False
    return True


def _verdict_hit(text: str) -> bool:
    words = ("值得写", "不值得写", "建议写", "不建议写", "暂不建议", "结论")
    return any(word in text for word in words)


def _metric_tokens(text: str) -> List[str]:
    return [match.group(0).replace(" ", "").lower() for match in METRIC_RE.finditer(text)]


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_OWNERSHIP_STRENGTH_MARKERS = ("独立", "主导", "负责", "牵头", "独自")
_COMPETENCE_MARKERS = (
    ("了解", r"了解"),
    ("接触", r"接触"),
    ("用过", r"用过"),
    ("会", r"会(?=\s*[A-Za-z])"),
    ("熟悉", r"熟悉"),
    ("掌握", r"掌握"),
    ("熟练", r"熟练"),
    ("精通", r"精通"),
)
_ACTION_FAMILIES = (
    ("build", ("搭建", "构建", "开发", "研发", "实现", "完成", "做了", "做过", "写了", "编写", "落地")),
    ("debug", ("调试", "排查", "排障")),
    ("optimize", ("优化", "调优")),
    ("test", ("测试", "验证")),
    ("deploy", ("部署", "上线", "发布")),
    ("integrate", ("接入", "集成", "封装")),
    ("design", ("设计", "选型", "调研")),
    ("train", ("训练", "微调", "调参", "评测")),
    ("clean", ("清洗", "标注", "整理", "梳理")),
    ("repair", ("修复",)),
    ("reproduce", ("复现",)),
)
_DELIVERY_FAMILIES = (
    ("deploy", ("部署", "上线", "发布")),
    ("deliver", ("交付", "提测", "验收")),
    ("open-source", ("开源",)),
    ("portfolio", ("作品集", "demo", "readme")),
    ("award", ("获奖",)),
)
_ENTITY_RE = re.compile(
    r"《[^》]+》|“[^”]+”|「[^」]+」|"
    r"[\u4e00-\u9fffA-Za-z0-9·]{2,12}(?:有限公司|公司|集团|科技|实验室|研究院|大学|学院)"
)


def _numbers_safe(original: str, value: str, level: str) -> bool:
    before = Counter(_NUMBER_RE.findall(original))
    after = Counter(_NUMBER_RE.findall(value))
    if level != "L3":
        return before == after
    if any(after[token] < count for token, count in before.items()):
        return False
    extras = after - before
    if not extras:
        return True
    derived = _derived_percentages(original)
    for token, count in extras.items():
        if count > 1:
            return False
        if not re.search(rf"(?<!\d){re.escape(token)}\s*[%％]", value):
            return False
        number = float(token)
        if not any(abs(number - item) <= 0.11 for item in derived):
            return False
    return True


def _derived_percentages(text: str) -> List[float]:
    values = [float(item) for item in _NUMBER_RE.findall(text)]
    result: List[float] = []
    for left in values:
        for right in values:
            if left == 0:
                continue
            result.extend(
                (
                    right / left * 100,
                    abs(left - right) / abs(left) * 100,
                )
            )
            if max(abs(left), abs(right)):
                result.append(abs(left - right) / max(abs(left), abs(right)) * 100)
    return result


def _signature(text: str, families: Sequence[Tuple[str, Sequence[str]]]) -> Tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        name
        for name, markers in families
        if any(str(marker).lower() in lowered for marker in markers)
    )


def _ownership_signature(text: str) -> Tuple[str, ...]:
    return tuple(marker for marker in _OWNERSHIP_STRENGTH_MARKERS if marker in text)


def _competence_signature(text: str) -> Tuple[str, ...]:
    return tuple(name for name, pattern in _COMPETENCE_MARKERS if re.search(pattern, text))


def _action_signature(text: str) -> set[str]:
    return set(_signature(text, _ACTION_FAMILIES))


def _delivery_signature(text: str) -> Tuple[str, ...]:
    return _signature(text, _DELIVERY_FAMILIES)


def _entity_signature(text: str) -> Tuple[str, ...]:
    return tuple(match.group(0) for match in _ENTITY_RE.finditer(text))


def _has_any(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(str(marker).lower() in lowered for marker in markers)


def _has_delivery(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in DELIVERY_MARKERS)


def _new_capabilities(original: str, polished: str) -> bool:
    from ..domain.lexicon import match_capabilities

    before = set(match_capabilities(original))
    after = set(match_capabilities(polished))
    return bool(after - before)


def _apply_replacements(resume: Resume, replacements: Dict[str, str], style: str) -> Resume:
    """Apply polished sentence text while preserving the original structure."""
    resume.style = style
    applied = 0
    for section in resume.sections:
        for node in section.nodes:
            if node.kind == "bullets":
                next_lines = [_replace_line(line, replacements) for line in node.lines]
                applied += sum(1 for before, after in zip(node.lines, next_lines) if before != after)
                node.lines = next_lines
            elif node.kind == "text":
                next_text = _replace_line(node.text, replacements)
                applied += int(next_text != node.text)
                node.text = next_text
            elif node.kind == "table" and node.table:
                for row in node.table.rows:
                    for index, cell in enumerate(row):
                        next_cell = _replace_line(cell, replacements)
                        applied += int(next_cell != cell)
                        row[index] = next_cell
    if applied:
        resume.notes.append(f"润色已应用：规则优先，原文未改事实（{applied} 个文本块）")
    return resume


def _replace_line(line: str, replacements: Dict[str, str]) -> str:
    key = clean_text(str(line or "")).strip()
    return replacements.get(key, line)




def _mark_llm_failure(items: Sequence[ResumePolishItem], error: str) -> None:
    for item in items:
        if item.engine == "original" and item.status == "调用失败":
            item.reason = error
