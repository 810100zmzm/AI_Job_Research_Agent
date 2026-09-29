"""简历事实抽取：把简历拆成「可核验的事实条目」，逐条判证据等级。

这一层只做规则：不联网、不读 .env、不调用大模型。Resume Agent、三个工具与前端共用同一份口径。

**判定顺序是有讲究的：先挡伪装，再判等级。**

    1. 否定（negated）—— 「未接触过向量数据库」「没做过 Docker 部署」「尚未有实习经历」。
       哪怕同一行里出现了「部署」这种动作词，也不算做过：它是反向证据。
    2. 背景（background）—— 写的是课题背景 / 项目简介 / 目标计划
       （「本项目旨在解决…」「计划学习 Go」），不是「我做过什么」。
    3. 课程（course）—— 修读 / 选修 / 培训 / 自学，且这一行没有写出任何指标：
       等级封顶到「仅提及」（上过课 != 做过事）。
    4. 判等级 —— 有结果 > 有动作 > 仅提及。

等级标记词与主线 project.py 逐字相同（量化指标表、交付词、动作词、归属词全部复用），
只多一条简历口径：**荣誉与名次**（获奖 / 奖学金 / 省级一等奖）是已经拿到的外部认定，
直接算「有结果」。

放宽口径（Agent Loop 里的 Adjust）只放宽「证据强弱」：仅提及也能算弱证据。
**否定与背景永远不放宽** —— 「没做过」写多少遍都不会变成「做过」，这正是这一层存在的理由。

宁可多挡一条，也不让「没做过」混进证据：被挡掉的条目会连同原文与行号一起列出来，
一眼就能核对它挡得对不对。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .jd import clean_text, display_path, read_text
from .lexicon import (
    CAPABILITIES,
    CAPABILITY_BY_KEY,
    CLAUSE_BOUNDARIES,
    NEGATION_PREFIXES_SORTED,
    find_keyword,
    match_capabilities,
    normalize_text,
)
from .project import (
    ACTION_MARKERS,
    DELIVERY_MARKERS,
    DELIVERY_NOISE,
    METRIC_RE,
    OWNERSHIP_MARKERS,
)
from ..core.schema import (
    EVIDENCE_ACTION,
    EVIDENCE_LABEL,
    EVIDENCE_MENTION,
    EVIDENCE_RANK,
    EVIDENCE_RESULT,
)

# ---- 挡住「写了但没做过」的两种判定 -----------------------------------------
FACT_BLOCK_NEGATED = "negated"          # 原文写了否定：反向证据
FACT_BLOCK_BACKGROUND = "background"    # 只是背景 / 目标 / 计划
FACT_BLOCK_LABEL = {
    FACT_BLOCK_NEGATED: "否定",
    FACT_BLOCK_BACKGROUND: "背景",
}
FACT_BLOCK_REASON_HINT = {
    FACT_BLOCK_NEGATED: "不能当「已具备」，放宽口径也不放",
    FACT_BLOCK_BACKGROUND: "只能算背景，不算做过的事",
}

SOLID_LEVELS = (EVIDENCE_RESULT, EVIDENCE_ACTION)

# ---- 章节角色：决定这一节是抬头信息、背景，还是正常判级 ---------------------
ROLE_META = "meta"        # 基本信息 / 求职意向：抬头信息，不进事实清单
ROLE_EDU = "edu"          # 教育背景
ROLE_EXP = "exp"          # 项目 / 实习 / 工作 / 科研 / 校园经历
ROLE_SKILL = "skill"      # 专业技能 / 技能清单
ROLE_AWARD = "award"      # 奖项荣誉 / 竞赛与获奖
ROLE_CLAIM = "claim"      # 自我评价 / 兴趣爱好
ROLE_OTHER = "other"      # 认不出的章节：原样保留，正常判级

ROLE_LABEL = {
    ROLE_META: "抬头信息",
    ROLE_EDU: "教育背景",
    ROLE_EXP: "经历",
    ROLE_SKILL: "技能",
    ROLE_AWARD: "荣誉",
    ROLE_CLAIM: "自我评价",
    ROLE_OTHER: "其他",
}

SECTION_ROLE = {
    "基本信息": ROLE_META, "个人信息": ROLE_META, "个人资料": ROLE_META, "个人档案": ROLE_META,
    "联系方式": ROLE_META, "求职意向": ROLE_META, "个人信息与求职意向": ROLE_META,
    "教育背景": ROLE_EDU, "教育经历": ROLE_EDU, "学习经历": ROLE_EDU, "学历信息": ROLE_EDU,
    "项目经历": ROLE_EXP, "项目经验": ROLE_EXP, "实习经历": ROLE_EXP, "实习经验": ROLE_EXP,
    "工作经历": ROLE_EXP, "工作经验": ROLE_EXP, "校园经历": ROLE_EXP, "实践经历": ROLE_EXP,
    "科研经历": ROLE_EXP, "研究经历": ROLE_EXP, "实习与校园经历": ROLE_EXP, "经历": ROLE_EXP,
    "专业技能": ROLE_SKILL, "技能清单": ROLE_SKILL, "技能特长": ROLE_SKILL, "专业能力": ROLE_SKILL,
    "技能": ROLE_SKILL, "技术栈": ROLE_SKILL, "掌握技能": ROLE_SKILL, "计算机技能": ROLE_SKILL,
    "奖项荣誉": ROLE_AWARD, "荣誉奖项": ROLE_AWARD, "竞赛与获奖": ROLE_AWARD, "获奖情况": ROLE_AWARD,
    "荣誉": ROLE_AWARD, "获奖经历": ROLE_AWARD, "奖项": ROLE_AWARD, "竞赛经历": ROLE_AWARD,
    "自我评价": ROLE_CLAIM, "个人评价": ROLE_CLAIM, "自我描述": ROLE_CLAIM, "个人总结": ROLE_CLAIM,
    "自我介绍": ROLE_CLAIM, "个人简介": ROLE_CLAIM, "兴趣爱好": ROLE_CLAIM,
}

# 认不出的章节名按关键词兜底；顺序有意义（先荣誉再经历：「竞赛与获奖」算荣誉）
ROLE_PATTERNS = (
    (re.compile(r"自我|评价|总结|描述|兴趣|爱好"), ROLE_CLAIM),
    (re.compile(r"奖|荣誉|竞赛|获奖|名次"), ROLE_AWARD),
    (re.compile(r"技能|技术栈|能力|证书"), ROLE_SKILL),
    (re.compile(r"教育|学历|学校|课程"), ROLE_EDU),
    (re.compile(r"项目|实习|工作|经历|实践|科研"), ROLE_EXP),
    (re.compile(r"基本|联系|个人|信息|资料"), ROLE_META),
)

# 整节都是背景的章节名：写的是课题 / 项目本身，不是「我做过什么」
BACKGROUND_SECTION_RE = re.compile(r"背景|简介|概述|痛点|需求分析|项目介绍|课题介绍|立项|研究意义")

# ---- 行级标记 ---------------------------------------------------------------
HEADING_RE = re.compile(r"^\s{0,3}(?P<hashes>#{1,6})\s*(?P<title>.+?)\s*#*\s*$")
BOLD_HEAD_RE = re.compile(r"^\s*\*{2}(?P<title>[^*]{1,24})\*{2}\s*[:：]?\s*$")
BULLET_RE = re.compile(r"^\s*(?:[-*+•·▪]|\d{1,2}\s*[.、)])\s+(?P<text>.*\S)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|(?P<cells>.*)\|\s*$")
DIVIDER_CELL_RE = re.compile(r"^:?-{2,}:?$")
QUOTE_RE = re.compile(r"^\s*>\s?(?P<text>.*)$")
KV_RE = re.compile(r"^(?P<key>[^：:]{1,10})\s*[:：]\s*(?P<value>\S.*)$")
RULE_RE = re.compile(r"^\s*(?:[-=*_—]{3,}|<[^>]+>)\s*$")
SECTION_INDEX_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]+|\d+)\s*[、.．)）]\s*")
TITLE_SUFFIX_RE = re.compile(r"\s*[·・|｜]\s*(?:个人)?(?:简历|经历|resume)\s*$", re.IGNORECASE)

# 表头行（「| 项目 | 内容 |」这种）不进事实清单
TABLE_HEADER_WORDS = frozenset({"项目", "内容", "字段", "名称", "说明", "信息", "项", "值", "详情", "备注"})

# ---- 否定：写了「没做过」的地方，一个都不许算成「做过」 ---------------------
# 强否定写法（不依赖能力词典也能认出来）
STRONG_NEGATION_RE = re.compile(
    r"未接触|没接触|没做过|未做过|没经验|无经验|零经验|没实践|未实践|没落地|未落地"
    r"|尚未|不曾|不具备|没有任何|没系统(?:学习|学过)|没实际|还不会|尚未掌握|没掌握"
    r"|待加强|待提升|待补"
)
# 「没有（过）…经历 / 经验 / 实践 / 实习」这类整体性否认，两个方向都认
ABSENCE_RE = re.compile(
    r"(?:尚未|不曾|未|没有|暂无|无|缺乏|欠缺)[^，。；,;]{0,12}?(?:经历|经验|实践|实习)"
    r"|(?:经历|经验|实践|实习)\s*[：:]?\s*(?:暂无|无|没有|零)"
)
NEGATION_TAIL_WINDOW = 20      # 关键词之后再往后看多少字，用来认「了解 X 但没做过完整项目」

# ---- 背景 / 目标 / 计划 -----------------------------------------------------
BACKGROUND_SENTENCE_RE = re.compile(
    r"旨在|为了解决|用于解决|目的是|研究现状|行业现状|业务现状|需求方|招标|立项"
    r"|^(?:\S{0,10}?)(?:背景|简介|概述|痛点|需求分析|项目介绍)\s*[：:是为]"
)
# 「按计划完成」不是「计划做」：三个常见的前缀先排除掉
PLAN_RE = re.compile(
    r"(?<!按)(?<!照)(?<!原)计划(?:学习|自学|掌握|参加|做|深入|研究|报名)"
    r"|打算(?:学习|自学|掌握|参加|做)"
    r"|想要(?:学习|从事)|希望(?:从事|成为|进入)"
    r"|目标是|职业目标|即将(?:学习|入职|开始)"
    r"|正在(?:学习|自学)|准备(?:学习|考)|待(?:学习|补充|提升|加强)"
)

# ---- 课程 / 培训 / 自学：上过课 != 做过事（有指标时才不当成课程）------------
COURSE_RE = re.compile(r"修读|选修|必修|主修|课程|公开课|慕课|mooc|训练营|网课|培训|讲座|lecture")

# ---- 荣誉：已经拿到的外部认定，直接算「有结果」------------------------------
AWARD_MARKERS = (
    "获奖", "奖学金", "一等奖", "二等奖", "三等奖", "特等奖", "优秀奖", "金奖", "银奖", "铜奖",
    "冠军", "亚军", "季军", "第一名", "第二名", "第三名", "名次",
    "三好学生", "优秀学生", "优秀班干部", "优秀干部", "标兵", "先进个人",
)


# ---- 判定用小工具 -----------------------------------------------------------

def _short(text: str, limit: int = 60) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _metrics(text: str) -> List[str]:
    return [" ".join(match.group(0).split()) for match in METRIC_RE.finditer(text)]


def _award_hit(text: str) -> str:
    return next((marker for marker in AWARD_MARKERS if marker in text), "")


def _first_marker(context: str) -> str:
    """上下文里出现的第一个否定写法（长词优先，避免「未」抢在「未接触」前面）。"""
    return next((prefix for prefix in NEGATION_PREFIXES_SORTED if prefix in context), "")


def _clause_tail(text_lower: str, position: int, window: int = NEGATION_TAIL_WINDOW) -> str:
    """关键词之后的一小段（遇到标点就截断）：用来认「了解 TensorFlow 但没做过完整项目」。"""
    tail = text_lower[position: position + window]
    cuts = [index for index in (tail.find(char) for char in CLAUSE_BOUNDARIES) if index >= 0]
    return tail[: min(cuts)] if cuts else tail


def negated_capability_keys(text: str) -> Tuple[List[str], str]:
    """返回 (被否定的能力 key, 命中的否定写法)。

    主线只看关键词**之前**的否定（「未接触过向量数据库」）；简历里还有
    「了解 TensorFlow 但没做过完整项目」这种把否定写在后面的说法，所以前后都要看。
    标点即边界：「熟悉 Python，没写过 C++」里的 Python 不会被后面的「没写过」连坐。
    """
    text_lower = normalize_text(text).lower()
    negated: List[str] = []
    marker = ""
    for capability in CAPABILITIES:
        for keyword in capability.all_keywords:
            position = find_keyword(text_lower, keyword)
            if position < 0:
                continue
            before = _first_marker(text_lower[max(0, position - 24): position])
            after = _first_marker(_clause_tail(text_lower, position + len(keyword)))
            if before or after:
                marker = marker or before or after
                negated.append(capability.key)
            break
    return negated, marker


def negation_hit(text: str) -> str:
    """整行的否定写法：命中的话，这一行不能当「已具备」的证据。"""
    strong = STRONG_NEGATION_RE.search(text)
    if strong:
        return strong.group(0)
    absence = ABSENCE_RE.search(text)
    return absence.group(0) if absence else ""


def background_hit(text: str) -> str:
    """整行的背景 / 目标写法（「本项目旨在…」）。"""
    match = BACKGROUND_SENTENCE_RE.search(text)
    return match.group(0).strip() if match else ""


def plan_hit(text: str) -> str:
    """整行的「打算做」写法（「计划学习 Go」）。"""
    match = PLAN_RE.search(text)
    return match.group(0) if match else ""


def is_course(text: str) -> bool:
    return bool(COURSE_RE.search(text))


def judge_level(text: str) -> Tuple[str, str, List[str], bool]:
    """判一条事实的证据等级：返回 (等级, 判定依据, 量化指标, 是否写了个人贡献)。

    标记词与主线 project.py 完全一致：量化指标 / 交付动作（+ 动作或归属）/ 动手动作 / 归属词。
    """
    lowered = text.lower()
    for noise in DELIVERY_NOISE:
        lowered = lowered.replace(noise, "")

    metrics = _metrics(text)
    has_ownership = any(marker in text for marker in OWNERSHIP_MARKERS)
    has_action = any(marker in text for marker in ACTION_MARKERS)
    has_delivery = any(marker in lowered for marker in DELIVERY_MARKERS)
    award = _award_hit(text)

    if metrics:
        return EVIDENCE_RESULT, f"量化指标：{'、'.join(metrics[:3])}", metrics, has_ownership
    if award:
        return EVIDENCE_RESULT, f"荣誉 / 名次：{award}（已经拿到的外部认定）", metrics, has_ownership
    if has_delivery and (has_action or has_ownership):
        return EVIDENCE_RESULT, "交付动作：开源 / 上线 / 发布这类词", metrics, has_ownership
    if has_action:
        return EVIDENCE_ACTION, "动手动作：搭建 / 实现 / 调试这类词", metrics, has_ownership
    if has_ownership:
        return EVIDENCE_ACTION, "写了「我 / 独立 / 负责」这类归属词", metrics, has_ownership
    if has_delivery:
        return EVIDENCE_ACTION, "交付动作（但没写是谁做的）", metrics, has_ownership
    return EVIDENCE_MENTION, "只写了「了解 / 熟悉 / 使用过」这类词汇，没有动手动作", metrics, has_ownership


# ---- 数据结构 ---------------------------------------------------------------

@dataclass
class ResumeFact:
    """简历里的一条事实（一个要点 / 一个段落 / 一行表格）。"""

    index: int
    text: str
    section: str
    role: str
    line_no: int
    source_file: str
    subsection: str = ""                  # 所属的 ### 小标题（项目名 / 岗位名）
    capabilities: List[str] = field(default_factory=list)   # 提到且没被否定的能力
    negated_keys: List[str] = field(default_factory=list)   # 被否定掉的能力
    level: str = EVIDENCE_MENTION         # 判出来的等级（挡住也照样记下来）
    basis: str = ""                       # 等级的判定依据
    blocked: str = ""                     # "" / negated / background
    block_reason: str = ""                # 挡住的原因（原文写法 + 为什么不算）
    has_metric: bool = False
    metrics: List[str] = field(default_factory=list)
    has_ownership: bool = False

    @property
    def ref(self) -> str:
        if self.line_no > 0:
            return f"{self.source_file}::L{self.line_no}"
        return self.source_file

    @property
    def usable(self) -> bool:
        """能不能当证据：否定与背景永不算数（放宽口径也不放）。"""
        return not self.blocked

    @property
    def solid(self) -> bool:
        return self.usable and self.level in SOLID_LEVELS

    @property
    def level_label(self) -> str:
        return FACT_BLOCK_LABEL[self.blocked] if self.blocked else EVIDENCE_LABEL.get(self.level, self.level)

    @property
    def detail(self) -> str:
        """表格里的「依据 / 为什么不算」列。"""
        return self.block_reason if self.blocked else self.basis

    @property
    def capability_names(self) -> List[str]:
        names = [CAPABILITY_BY_KEY[key].name for key in self.capabilities if key in CAPABILITY_BY_KEY]
        return list(dict.fromkeys(names))

    @property
    def negated_names(self) -> List[str]:
        names = [CAPABILITY_BY_KEY[key].name for key in self.negated_keys if key in CAPABILITY_BY_KEY]
        return list(dict.fromkeys(names))

    @property
    def quote_short(self) -> str:
        return _short(self.text)


@dataclass
class ResumeSection:
    """简历里的一节（## 标题下的内容）。"""

    title: str
    role: str
    line_no: int
    background: bool = False              # 整节都是背景（项目简介 / 课题背景）
    facts: List[ResumeFact] = field(default_factory=list)

    @property
    def bullet_count(self) -> int:
        return len(self.facts)


@dataclass
class ResumeDoc:
    """一份解析完的简历：抬头信息 + 章节 + 全部事实条目。"""

    title: str
    source_file: str
    line_count: int = 0
    meta: List[Tuple[str, str]] = field(default_factory=list)
    sections: List[ResumeSection] = field(default_factory=list)
    facts: List[ResumeFact] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    text: str = ""

    @property
    def name(self) -> str:
        return self.title or self.source_file.rsplit("/", 1)[-1]

    @property
    def bullet_count(self) -> int:
        return len(self.facts)

    @property
    def capability_keys(self) -> List[str]:
        keys = {key for fact in self.facts if fact.usable for key in fact.capabilities}
        return sorted(keys)

    def meta_value(self, key: str) -> str:
        for name, value in self.meta:
            if name == key:
                return value
        return ""

    def add_meta(self, key: str, value: str) -> None:
        """记一条抬头信息：同一个字段写了两遍时，后写的覆盖先写的。

        「开头引用一行求职意向、基本信息里再写一遍」在简历里很常见，所以同一个字段名只留一条；
        认不出字段名的整行（key 是「基本信息」）走 append，不参与覆盖，免得把不同内容压成一条。
        """
        for index, (name, _) in enumerate(self.meta):
            if name == key:
                self.meta[index] = (key, value)
                return
        self.meta.append((key, value))


@dataclass
class EvidenceSummary:
    """一次证据判定的结果：能当证据的、被挡掉的，各是多少。"""

    facts: List[ResumeFact]
    relaxed: bool = False

    @property
    def usable(self) -> List[ResumeFact]:
        """没被否掉也没被判成背景的条目。"""
        return [fact for fact in self.facts if fact.usable]

    @property
    def blocked(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.blocked]

    @property
    def negated(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.blocked == FACT_BLOCK_NEGATED]

    @property
    def background(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if fact.blocked == FACT_BLOCK_BACKGROUND]

    @property
    def solid(self) -> List[ResumeFact]:
        """硬证据：有动作或有结果，且没被挡住。"""
        return [fact for fact in self.usable if fact.level in SOLID_LEVELS]

    @property
    def weak(self) -> List[ResumeFact]:
        """弱证据：仅提及（只写了「了解 / 熟悉」，没有动手动作）。"""
        return [fact for fact in self.usable if fact.level == EVIDENCE_MENTION]

    @property
    def counts(self) -> Dict[str, int]:
        counts = {level: 0 for level in SOLID_LEVELS + (EVIDENCE_MENTION,)}
        for fact in self.usable:
            counts[fact.level] = counts.get(fact.level, 0) + 1
        return counts

    def counts_as_evidence(self, fact: ResumeFact) -> bool:
        """严格口径只认硬证据；放宽口径才把「仅提及」当弱证据 —— 但否定与背景永远不算。"""
        if not fact.usable:
            return False
        return fact.level in SOLID_LEVELS or self.relaxed

    @property
    def evidence(self) -> List[ResumeFact]:
        return [fact for fact in self.facts if self.counts_as_evidence(fact)]

    def evidence_count(self, relaxed: Optional[bool] = None) -> int:
        if relaxed is None or relaxed == self.relaxed:
            return len(self.evidence)
        return len([f for f in self.facts if f.usable and (f.level in SOLID_LEVELS or relaxed)])

    @property
    def conclusion(self) -> str:
        """一句话结论：能拿来当证据的有几条、本次按当前口径计入几条、挡掉了几条。"""
        counts = self.counts
        parts = [
            f"{len(self.usable)} 条事实能当证据"
            f"（硬证据 {len(self.solid)}：有结果 {counts.get(EVIDENCE_RESULT, 0)} / "
            f"有动作 {counts.get(EVIDENCE_ACTION, 0)}；弱证据 {len(self.weak)}：仅提及 "
            f"{counts.get(EVIDENCE_MENTION, 0)}）",
            f"本次按{'放宽' if self.relaxed else '严格'}口径计入 {len(self.evidence)} 条",
        ]
        if self.blocked:
            # 否定与背景是两个不放宽的口径：写多少遍「没做过」都不会变成「做过」
            parts.append(
                f"{len(self.blocked)} 条被挡掉（否定 {len(self.negated)} / 背景 {len(self.background)}），"
                "它们都不能算「已具备的证据」"
            )
        return "；".join(parts) + "。"


# ---- 抽取 -------------------------------------------------------------------

def section_role(title: str) -> str:
    """章节名 -> 角色（认不出的按关键词兜底，再认不出就是 other）。"""
    name = clean_text(SECTION_INDEX_RE.sub("", str(title or ""))).strip()
    if not name:
        return ROLE_OTHER
    if name in SECTION_ROLE:
        return SECTION_ROLE[name]
    key = name.replace(" ", "").lower()
    if key in SECTION_ROLE:
        return SECTION_ROLE[key]
    for pattern, role in ROLE_PATTERNS:
        if pattern.search(name):
            return role
    return ROLE_OTHER


def is_background_section(title: str) -> bool:
    """整节都是背景（「项目简介」「课题背景」）—— 里面的条目一律不算做过的事。

    「教育背景」「个人简介」不算：前者是学历信息，后者是自我介绍，各有各的角色，
    不能因为名字里带「背景 / 简介」就整节作废。
    """
    if section_role(title) in (ROLE_META, ROLE_EDU, ROLE_CLAIM, ROLE_AWARD, ROLE_SKILL):
        return False
    return bool(BACKGROUND_SECTION_RE.search(str(title or "")))


def make_fact(
    index: int,
    text: str,
    section: str,
    role: str,
    line_no: int,
    source_file: str,
    subsection: str = "",
    background_section: bool = False,
) -> ResumeFact:
    """一条事实的完整判定：否定 -> 背景 / 计划 -> 课程封顶 -> 等级。"""
    level, basis, metrics, has_ownership = judge_level(text)
    mentioned = match_capabilities(text)
    negated_keys, marker = negated_capability_keys(text)
    capabilities = [key for key in mentioned if key not in negated_keys]

    blocked = ""
    reason = ""
    whole_line_negation = negation_hit(text)
    if whole_line_negation or negated_keys:
        blocked = FACT_BLOCK_NEGATED
        reason = f"原文写了「{whole_line_negation or marker}」：这是反向证据，不能当「已具备」"
    if not blocked:
        background = background_hit(text)
        plan = plan_hit(text)
        if background_section:
            blocked = FACT_BLOCK_BACKGROUND
            reason = f"整节都是「{section}」：写的是背景 / 简介，不是「我做过什么」"
        elif background:
            blocked = FACT_BLOCK_BACKGROUND
            reason = f"写的是背景 / 目标（{background}），不是个人动作"
        elif plan:
            blocked = FACT_BLOCK_BACKGROUND
            reason = f"写的是打算（{plan}）：只说明想学，不等于已经具备"

    if not blocked and is_course(text) and not metrics:
        level = EVIDENCE_MENTION
        basis = "课程 / 培训 / 自学：上过课不等于做过事，等级封顶「仅提及」"
    if negated_keys:
        names = _names(negated_keys)
        if names:
            reason += f"；被否定的能力：{'、'.join(names)}"
    positive_names = _names(capabilities)
    if blocked == FACT_BLOCK_NEGATED and positive_names:
        reason += f"；同一行还提到了：{'、'.join(positive_names)}（整行都不计入证据）"

    fact = ResumeFact(
        index=index,
        text=text,
        section=section,
        role=role,
        line_no=line_no,
        source_file=source_file,
        subsection=subsection,
        capabilities=capabilities,
        negated_keys=negated_keys,
        level=level,
        basis=basis,
        blocked=blocked,
        block_reason=reason,
        has_metric=bool(metrics),
        metrics=metrics,
        has_ownership=has_ownership,
    )
    return fact


def _names(keys) -> List[str]:
    return [CAPABILITY_BY_KEY[key].name for key in keys if key in CAPABILITY_BY_KEY]


def _file_title(lines) -> Tuple[str, int]:
    """文件最上面那行一级标题（`# 姓名 · 个人简历`）；第一个标题就是章节名的话，说明没有文档标题。"""
    for line_no, raw in enumerate(lines, start=1):
        match = HEADING_RE.match(raw)
        if not match:
            continue
        if len(match.group("hashes")) != 1:
            return "", 0
        return clean_text(match.group("title")), line_no
    return "", 0


def _clean_title(name: str) -> str:
    return TITLE_SUFFIX_RE.sub("", clean_text(name)).strip()


def _table_cells(line: str) -> Optional[List[str]]:
    match = TABLE_ROW_RE.match(line)
    if not match:
        return None
    cells = [clean_text(cell) for cell in match.group("cells").split("|")]
    return [cell for cell in cells if cell]


def _is_table_header(cells) -> bool:
    return bool(cells) and all(cell.strip() in TABLE_HEADER_WORDS for cell in cells)


def _is_table_divider(cells) -> bool:
    return bool(cells) and all(DIVIDER_CELL_RE.match(cell.strip() or "-") for cell in cells)


def extract_resume(text: str, source_file: str = "", title: str = "") -> ResumeDoc:
    """把简历原文拆成事实条目：章节切分 + 抬头信息 + 逐条判定。"""
    lines = str(text or "").splitlines()
    file_title, title_line = _file_title(lines)
    doc_title = title.strip() or _clean_title(file_title) or Path(source_file).stem or "简历"

    doc = ResumeDoc(title=doc_title, source_file=source_file, line_count=len(lines), text=text)
    current: Optional[ResumeSection] = None
    subsection = ""
    index = 0

    def section() -> ResumeSection:
        nonlocal current
        if current is None:
            current = ResumeSection(title="（开头）", role=ROLE_OTHER, line_no=0)
            doc.sections.append(current)
        return current

    def add_fact(clean: str, line_no: int) -> None:
        nonlocal index
        target = section()
        index += 1
        target.facts.append(
            make_fact(
                index,
                clean,
                target.title,
                target.role,
                line_no,
                doc.source_file,
                subsection=subsection,
                background_section=target.background,
            )
        )

    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if line_no == title_line or not stripped or RULE_RE.match(line):
            continue

        quote = QUOTE_RE.match(line)
        if quote:
            # 引用行是文件级说明：能认出「key：value」就当抬头信息，其余（示例提示之类）跳过
            head = KV_RE.match(clean_text(quote.group("text")))
            if head:
                doc.add_meta(head.group("key").strip(), head.group("value").strip())
            continue

        heading = HEADING_RE.match(line)
        if heading:
            name = clean_text(heading.group("title"))
            level = len(heading.group("hashes"))
            if level <= 1:
                continue                       # 一级标题已经在上面当过文档标题了
            if level == 2:
                title_text = clean_text(SECTION_INDEX_RE.sub("", name))
                current = ResumeSection(
                    title=title_text or name,
                    role=section_role(name),
                    line_no=line_no,
                    background=is_background_section(name),
                )
                doc.sections.append(current)
                subsection = ""
                if current.background:
                    doc.notes.append(f"「{current.title}」整节按背景处理（第 {line_no} 行）")
            else:
                subsection = name            # ### 是章节里的小标题（项目名 / 岗位名）
                section()
            continue

        bold = BOLD_HEAD_RE.match(line)
        if bold:
            current = ResumeSection(
                title=clean_text(bold.group("title")),
                role=section_role(bold.group("title")),
                line_no=line_no,
                background=is_background_section(bold.group("title")),
            )
            doc.sections.append(current)
            subsection = ""
            continue

        cells = _table_cells(line)
        if cells is not None:
            if _is_table_divider(cells) or _is_table_header(cells):
                continue
            if section().role == ROLE_META:
                if len(cells) >= 2:
                    doc.add_meta(cells[0], "：".join(cells[1:]))
                continue
            add_fact("：".join(cells[:2]) if len(cells) == 2 else " | ".join(cells), line_no)
            continue

        bullet = BULLET_RE.match(line)
        content = clean_text(bullet.group("text") if bullet else line)
        if not content:
            continue

        if section().role == ROLE_META:
            head = KV_RE.match(content)
            if head:
                doc.add_meta(head.group("key").strip(), head.group("value").strip())
            else:
                doc.meta.append(("基本信息", content))
            continue
        add_fact(content, line_no)

    doc.facts = [fact for item in doc.sections for fact in item.facts]
    # 抬头信息节（基本信息 / 求职意向）的内容已经全进了 doc.meta：空壳章节不留，免得白占章节
    doc.sections = [section for section in doc.sections if section.facts or section.role != ROLE_META]
    if not doc.sections:
        doc.notes.append("没有识别到任何章节：整份简历当成一段处理")
    return doc


def load_resume(path, title: str = "") -> ResumeDoc:
    """读一份简历文件（md / txt），解析成事实条目。"""
    candidate = Path(path)
    return extract_resume(read_text(candidate), source_file=display_path(candidate), title=title)


def summarize(facts, relaxed: bool = False) -> EvidenceSummary:
    """把事实条目汇总成证据判定结果。"""
    return EvidenceSummary(facts=list(facts), relaxed=relaxed)


def table_lines(header: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    """渲染一张 Markdown 表格（单元格里的 | 换成全角，免得把表撑坏）。"""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [str(cell).replace("|", "｜").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines
