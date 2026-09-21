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

# ---- 来源标记：规则算出来的 vs 大模型生成的 ---------------------------------
# 报告里每一段都要能回答「这段话是谁写的」，因此每条内容都带一个 source。
SOURCE_RULE = "rule"
SOURCE_LLM_SUGGEST = "llm-suggest"       # 建议写法草稿生成
SOURCE_LLM_INTERVIEW = "llm-interview"   # 面试追问预演
SOURCE_LLM_POLISH = "llm-polish"         # 报告自然语言润色
SOURCE_LLM = (SOURCE_LLM_SUGGEST, SOURCE_LLM_INTERVIEW, SOURCE_LLM_POLISH)
SOURCES = (SOURCE_RULE,) + SOURCE_LLM
SOURCE_LABEL = {SOURCE_RULE: "[规则]"}
SOURCE_LABEL.update({source: "[LLM]" for source in SOURCE_LLM})

# 一块 LLM 内容的状态（红线 3：调用失败必须看得见，主报告照跑）
STATUS_OFF = "未启用"
STATUS_OK = "正常"
STATUS_FAILED = "调用失败"
STATUS_EMPTY = "未返回可用内容"
STATUS_DISCARDED = "已丢弃"
STATUS_IDLE = "已闲置"          # 按用户要求暂时停用的块（例如待改版的面试追问预演）
STATUS_SKIPPED = "已达调用上限"   # 还有活没干，但本次运行的调用预算用完了

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
        return self.source_file

    @property
    def level_label(self) -> str:
        return EVIDENCE_LABEL.get(self.level, self.level)

    @property
    def quote_short(self) -> str:
        return _short(self.text)


@dataclass(frozen=True)
class ImageRef:
    """项目描述里引用到的一张本地图片（可选视觉层的输入）。"""

    raw: str          # 原文里怎么写的就是什么，例如 "assets/chart.png"
    path: str         # 解析后的本地路径（已确认文件存在）
    ref: str          # 报告里的出处写法：「图片:<相对路径>」
    line_no: int = 0  # 引用出现在第几行；--image 显式传入时为 0


@dataclass
class VisionReport:
    """视觉层（可选）的执行结果：只记录看到了什么、做了什么，不参与判定。"""

    enabled: bool = False
    model: str = ""
    images: List[ImageRef] = field(default_factory=list)
    parsed: int = 0            # 真正解析出事实的图片数
    facts_added: int = 0
    error: str = ""
    notes: List[str] = field(default_factory=list)


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
class LlmBlock:
    """报告生成之后，由大模型追加的一块内容（三类任务各一块）。

    硬边界：只挂在 Verdict.llm_report 上，绝不替换规则写出的结论、理由、风险与建议写法；
    每块都带 source（llm-suggest / llm-interview / llm-polish），报告里一律标 [LLM]。
    """

    source: str
    title: str
    model: str = ""
    enabled: bool = True          # 是否真的发出了调用
    calls: int = 0                # 这一块用掉的调用次数（含重试）
    text: str = ""                # 草稿 / 润色这类整段文本
    items: List[Dict[str, str]] = field(default_factory=list)   # 追问预演这类条目
    error: str = ""
    failed: bool = False          # 调用本身失败（网络 / 额度 / 超时 / 超上限）
    discarded: bool = False       # 触碰「LLM 不做结论」红线，内容已丢弃
    idle: bool = False            # 按用户要求闲置（待改版），本次不调用模型
    skipped: bool = False         # 预算用完，没轮到它
    notes: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return SOURCE_LABEL.get(self.source, "[LLM]")

    @property
    def is_empty(self) -> bool:
        return not (self.text or self.items)

    @property
    def status(self) -> str:
        """报告里显示的状态：未启用 / 调用失败 / 正常 ……"""
        if not self.enabled:
            return STATUS_OFF
        if self.idle:
            return STATUS_IDLE
        if self.discarded:
            return STATUS_DISCARDED
        if self.skipped:
            return STATUS_SKIPPED
        if self.failed:
            return STATUS_FAILED
        if self.is_empty:
            return STATUS_EMPTY
        return STATUS_OK

    @property
    def ok(self) -> bool:
        return not (self.failed or self.discarded or self.skipped or self.idle) and not self.is_empty

    def summary(self) -> str:
        if self.items:
            return f"{len(self.items)} 条"
        if self.text:
            return f"{len(self.text)} 字"
        return "无内容"


@dataclass
class LlmReport:
    """一次运行里 LLM 装饰层的整体情况：三块内容 + 调用账本（可限额、可降级）。"""

    enabled: bool = False
    model: str = ""
    calls: int = 0                # 本次运行实际发出的调用数（含重试）
    call_limit: int = 10          # 调用上限，默认 10
    timeout: float = 30.0         # 单次调用超时（秒）
    retries: int = 1              # 失败重试次数
    blocks: List[LlmBlock] = field(default_factory=list)
    note: str = ""                # 整体说明，例如「未配置 DEEPSEEK_API_KEY」

    def block(self, source: str) -> Optional[LlmBlock]:
        for item in self.blocks:
            if item.source == source:
                return item
        return None

    @property
    def sources(self) -> List[str]:
        return [item.source for item in self.blocks]

    @property
    def ok_count(self) -> int:
        return len([item for item in self.blocks if item.ok])

    @property
    def degraded(self) -> List[LlmBlock]:
        """没能正常产出内容的块（未启用 / 调用失败 / 已丢弃 / 空内容）。"""
        return [item for item in self.blocks if not item.ok]

    @property
    def is_empty(self) -> bool:
        return not self.blocks and not self.note


@dataclass
class Verdict:
    """最终判断（全部由规则产生）。"""

    call: str
    headline: str
    reasons: List[Reason] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    rewrite: str = ""
    stop_reason: str = ""
    llm_report: Optional[LlmReport] = None


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
    vision: VisionReport = field(default_factory=VisionReport)

    @property
    def decision(self) -> str:
        return self.trace[-1].decision if self.trace else ""

    @property
    def llm_report(self) -> Optional[LlmReport]:
        """报告生成后追加的 LLM 内容（可选）：三块生成结果，不参与判定。"""
        return self.verdict.llm_report if self.verdict else None

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
