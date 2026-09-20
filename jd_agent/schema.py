"""数据结构：岗位需求 -> 项目事实 -> 证据匹配 -> 判断与 Agent Trace。

一条硬规则：任何出现在「理由 / 风险」里的说法，都必须同时带 JD 行号和项目行号，
保证运行 Trace 可以逐条回查，而不是靠模型自由发挥。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---- 岗位需求层级 -----------------------------------------------------------
LEVEL_MUST = "must"   # 任职要求
LEVEL_DUTY = "duty"   # 岗位职责
LEVEL_PLUS = "plus"   # 加分项

LEVEL_LABEL = {LEVEL_MUST: "任职要求", LEVEL_DUTY: "岗位职责", LEVEL_PLUS: "加分项"}
LEVEL_WEIGHT = {LEVEL_MUST: 1.0, LEVEL_DUTY: 0.8, LEVEL_PLUS: 0.5}

# 项目无法提供证据的类别：
#   准入门槛（学历 / 届别 / 实习时长）不是项目能证明的；通用素质也不该由单个项目背书
NON_PROVABLE_CATEGORIES = ("准入门槛", "通用素质")

# ---- 项目证据等级 -----------------------------------------------------------
EVIDENCE_RESULT = "result"    # 有结果：量化指标 / 上线 / 开源 / 部署 / 获奖
EVIDENCE_ACTION = "action"    # 有动作：搭建 / 实现 / 调试 / 调优，但没写结果
EVIDENCE_MENTION = "mention"  # 仅提及：技术栈罗列 / 使用过 / 了解
EVIDENCE_NONE = "none"        # 项目里找不到

EVIDENCE_LABEL = {
    EVIDENCE_RESULT: "有结果",
    EVIDENCE_ACTION: "有动作",
    EVIDENCE_MENTION: "仅提及",
    EVIDENCE_NONE: "无证据",
}
EVIDENCE_RANK = {EVIDENCE_RESULT: 3, EVIDENCE_ACTION: 2, EVIDENCE_MENTION: 1, EVIDENCE_NONE: 0}

# ---- 每一步只允许这四种 Decision --------------------------------------------
DECISION_CONTINUE = "Continue"
DECISION_ADJUST = "Adjust"
DECISION_ASK = "Ask"
DECISION_STOP = "Stop"
DECISIONS = (DECISION_CONTINUE, DECISION_ADJUST, DECISION_ASK, DECISION_STOP)

# ---- 最终判断 ---------------------------------------------------------------
VERDICT_YES = "值得写"
VERDICT_EDIT = "值得写，但必须先改写"
VERDICT_NO = "暂不建议写"

# ---- 风险类型 ---------------------------------------------------------------
RISK_GAP = "证据缺口"        # JD 要，项目里完全找不到
RISK_CONFLICT = "事实冲突"   # 项目里明确写了没做过
RISK_PROBE = "追问风险"      # 只有动作、没有结果
RISK_WORDING = "表述风险"    # 只有技术栈级证据，别用过强的词


@dataclass(frozen=True)
class SubItem:
    """能力下的可拆分维度，例如「深度学习框架」拆成 PyTorch / TensorFlow。"""

    name: str
    keywords: Tuple[str, ...]


@dataclass(frozen=True)
class Capability:
    """一个可对比的能力项（能力词典里的一条）。"""

    key: str
    name: str
    category: str
    keywords: Tuple[str, ...]
    subs: Tuple[SubItem, ...] = ()
    actions: Tuple[str, ...] = ()

    @property
    def all_keywords(self) -> Tuple[str, ...]:
        extra = tuple(kw for sub in self.subs for kw in sub.keywords)
        return tuple(dict.fromkeys(self.keywords + extra))


@dataclass
class Requirement:
    """岗位真正需要的一项能力（带 JD 原文出处）。"""

    capability_key: str
    capability_name: str
    category: str
    level: str
    section: str
    line_no: int
    quote: str
    source_file: str

    @property
    def name(self) -> str:
        return self.capability_name

    @property
    def weight(self) -> float:
        return LEVEL_WEIGHT.get(self.level, 0.5)

    @property
    def level_label(self) -> str:
        return LEVEL_LABEL.get(self.level, self.level)

    @property
    def ref(self) -> str:
        return f"{self.source_file}::L{self.line_no}"

    @property
    def provable(self) -> bool:
        """这个要求能不能靠一个项目来证明。"""
        return self.category not in NON_PROVABLE_CATEGORIES

    @property
    def quote_short(self) -> str:
        return _short(self.quote)


@dataclass
class JobPosting:
    """一份岗位描述（单岗位）。"""

    jd_id: str
    title: str
    source_file: str
    company: str = ""
    job_type: str = ""
    line_count: int = 0
    bullet_count: int = 0
    position_count: int = 1
    position_titles: List[str] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.title or self.jd_id

    def by_key(self, capability_key: str) -> List[Requirement]:
        return [item for item in self.requirements if item.capability_key == capability_key]


@dataclass
class ProjectFact:
    """项目描述里的一条事实（一个要点或一个段落）。"""

    text: str
    section: str
    line_no: int
    source_file: str
    capabilities: List[str] = field(default_factory=list)
    negated: List[str] = field(default_factory=list)
    level: str = EVIDENCE_MENTION
    has_metric: bool = False
    has_ownership: bool = False
    metrics: List[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        if self.line_no > 0:
            return f"{self.source_file}::L{self.line_no}"
        return f"{self.source_file}::补充说明"

    @property
    def level_label(self) -> str:
        return EVIDENCE_LABEL.get(self.level, self.level)

    @property
    def quote_short(self) -> str:
        return _short(self.text)


@dataclass
class Project:
    """一份项目描述（解析结果）。"""

    title: str
    source_file: str
    line_count: int = 0
    facts: List[ProjectFact] = field(default_factory=list)
    text: str = ""

    @property
    def name(self) -> str:
        return self.title or self.source_file.rsplit("/", 1)[-1]

    @property
    def actionable_facts(self) -> List[ProjectFact]:
        """有动作或结果的事实 —— 只有这类才能真正支撑简历。"""
        return [f for f in self.facts if f.level in (EVIDENCE_RESULT, EVIDENCE_ACTION)]

    @property
    def result_facts(self) -> List[ProjectFact]:
        return [f for f in self.facts if f.level == EVIDENCE_RESULT]

    @property
    def ownership_facts(self) -> List[ProjectFact]:
        return [f for f in self.facts if f.has_ownership]

    def facts_for(self, capability_key: str) -> List[ProjectFact]:
        return [f for f in self.facts if capability_key in f.capabilities]

    def negated_keys(self) -> set:
        return {key for fact in self.facts for key in fact.negated}


@dataclass
class EvidenceMatch:
    """一条「岗位要求 -> 项目证据」的比对结果。"""

    requirement: Requirement
    facts: List[ProjectFact] = field(default_factory=list)
    level: str = EVIDENCE_NONE
    matched_subs: List[str] = field(default_factory=list)
    missing_subs: List[str] = field(default_factory=list)
    relaxed: bool = False   # 是否只有放宽规则（允许「仅提及」级）才匹配上
    note: str = ""

    @property
    def requirement_name(self) -> str:
        return self.requirement.name

    @property
    def level_label(self) -> str:
        return EVIDENCE_LABEL.get(self.level, self.level)

    @property
    def has_evidence(self) -> bool:
        return self.level != EVIDENCE_NONE

    @property
    def best_fact(self) -> Optional[ProjectFact]:
        if not self.facts:
            return None
        return max(self.facts, key=lambda f: EVIDENCE_RANK.get(f.level, 0))

    @property
    def score(self) -> float:
        return self.requirement.weight * EVIDENCE_RANK.get(self.level, 0)


@dataclass
class TraceStep:
    """Agent Loop 的一步：Action -> Observation -> State Update -> Decision。"""

    index: int
    action: str
    observation: str
    state_update: str
    decision: str
    decision_reason: str

    def as_row(self) -> List[str]:
        return [
            str(self.index),
            self.action,
            self.observation,
            self.state_update,
            f"{self.decision}（{self.decision_reason}）",
        ]


@dataclass
class Reason:
    """一条理由，必须同时挂 JD 出处与项目出处。"""

    text: str
    requirement_name: str
    jd_ref: str
    project_refs: List[str] = field(default_factory=list)


@dataclass
class Risk:
    """一条风险：面试官会怎么追问，或哪里会被戳穿。"""

    text: str
    kind: str
    refs: List[str] = field(default_factory=list)


@dataclass
class Verdict:
    """最终判断。"""

    call: str
    headline: str
    reasons: List[Reason] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    rewrite: str = ""
    stop_reason: str = ""


@dataclass
class AgentState:
    """Agent 的完整状态，也是唯一的数据出口（报告和 JSON 都由它渲染）。"""

    jd: JobPosting
    project: Project
    core_requirements: List[Requirement] = field(default_factory=list)
    skipped_requirements: List[Requirement] = field(default_factory=list)
    matches: List[EvidenceMatch] = field(default_factory=list)
    trace: List[TraceStep] = field(default_factory=list)
    rounds: int = 1
    question: str = ""
    question_reason: str = ""
    answer: str = ""
    sufficiency: str = ""
    verdict: Optional[Verdict] = None

    @property
    def decision(self) -> str:
        return self.trace[-1].decision if self.trace else ""

    @property
    def finished(self) -> bool:
        return self.decision == DECISION_STOP

    @property
    def needs_answer(self) -> bool:
        return self.decision == DECISION_ASK

    @property
    def matched(self) -> List[EvidenceMatch]:
        return [m for m in self.matches if m.has_evidence]

    @property
    def missing(self) -> List[EvidenceMatch]:
        return [m for m in self.matches if not m.has_evidence]

    @property
    def solid(self) -> List[EvidenceMatch]:
        """有动作或结果、能写进简历的匹配。"""
        return [m for m in self.matches if m.level in (EVIDENCE_RESULT, EVIDENCE_ACTION)]

    @property
    def coverage(self) -> float:
        if not self.core_requirements:
            return 0.0
        return len(self.solid) / len(self.core_requirements)

    @property
    def result_match_count(self) -> int:
        return len([m for m in self.matches if m.level == EVIDENCE_RESULT])

    @property
    def stop_reason(self) -> str:
        return self.verdict.stop_reason if self.verdict else ""

    def level_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {level: 0 for level in EVIDENCE_LABEL}
        for match in self.matches:
            counts[match.level] = counts.get(match.level, 0) + 1
        return counts


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
