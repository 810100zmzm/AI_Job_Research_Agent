"""Polish Agent 的规则层：把「JD 要求」与「已核验的简历事实」对齐成一份写作简报。

Polish Agent 最终要产出三样东西（建议写法 / 追问预演 / 润色），这三样都由大模型写。
但在调模型之前，先把**模型能看到什么**用规则钉死 —— 这就是这一层：

    1. JD 要求    —— 能力词典从 JD 原文里匹配出来的要求（带 JD 行号与原文引用）；
    2. 已核验事实 —— Resume Agent 那一套判定（先挡否定 / 背景，再判有结果 / 有动作 / 仅提及）
                     之后**能当证据**的条目（带简历行号与原文）；
    3. 对照与风险 —— 要求 ↔ 事实 的能力词典匹配：哪条要求有证据（强 / 弱）、哪条完全没提、
                     哪条正好是「简历里写着没做过」的反向证据、哪条只有动作没有结果。

三条纪律：
  * 规则先跑、模型后介入：模型只读这份简报（不是 JD / 简历全文，只有规则挑出来的句子与出处）；
  * 事实不增不减：简报里每一句都带出处（`文件::L行号`），模型只能在已核验事实的范围内组织语言；
  * 反向证据必须露出来：JD 要的能力如果恰好是简历里「没做过」的那条，先当成风险标出来，
    而不是让模型照着 JD 去编一段出来。

这一层只做规则：不联网、不读 `.env`、不调用大模型。三个工具（简历结构化 / 否定背景识别 /
证据等级判定）与它共用同一份口径，命令行与前端也共用同一份。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..domain.jd import clean_text
from ..domain.lexicon import (
    CAPABILITY_BY_KEY,
    find_keyword,
    match_capabilities,
    match_sub_items,
    normalize_text,
)
from ..domain.resume_facts import EvidenceSummary, ResumeDoc, ResumeFact
from ..core.schema import (
    EVIDENCE_ACTION,
    EVIDENCE_LABEL,
    EVIDENCE_MENTION,
    EVIDENCE_NONE,
    EVIDENCE_RANK,
    RISK_CONFLICT,
    RISK_GAP,
    RISK_PROBE,
    RISK_WORDING,
    JobPosting,
    Requirement,
    Risk,
)

# 口径（写死在这里，方便直接检查，也方便日后调）
MIN_FACTS = 3            # 可当证据的事实少于这个数：撑不住一份写法建议，先 Ask
MAX_RISKS = 6            # 风险最多列几条：简报要短，模型不该被淹
MAX_REFS = 3             # 每条要求最多带几条事实出处


# ---- 能力词典匹配 -----------------------------------------------------------

@dataclass(frozen=True)
class CapabilityHit:
    """一处能力词典命中：某个能力项出现在第几行、那一行说了什么、命中了哪些关键词。"""

    key: str
    name: str
    category: str
    line_no: int
    line: str
    keywords: Tuple[str, ...] = ()

    @property
    def position(self) -> str:
        return f"L{self.line_no}" if self.line_no else "整段"


def _keywords_hit(line_lower: str, keywords: Sequence[str]) -> Tuple[str, ...]:
    return tuple(keyword for keyword in keywords if find_keyword(line_lower, keyword) >= 0)


def capability_hits(text: str) -> List[CapabilityHit]:
    """逐行做能力词典匹配：命中的能力项 + 行号 + 命中的关键词（顺序与原文一致）。"""
    hits: List[CapabilityHit] = []
    for line_no, raw in enumerate(str(text or "").splitlines(), start=1):
        line = clean_text(raw)
        if not line:
            continue
        lowered = normalize_text(line).lower()
        for key in match_capabilities(line):
            capability = CAPABILITY_BY_KEY.get(key)
            if capability is None:
                continue
            hits.append(
                CapabilityHit(
                    key=capability.key,
                    name=capability.name,
                    category=capability.category,
                    line_no=line_no,
                    line=line,
                    keywords=_keywords_hit(lowered, capability.all_keywords),
                )
            )
    return hits


def capability_keys(text: str) -> List[str]:
    """材料里命中的能力项 key（按首次出现的顺序去重）。"""
    return list(dict.fromkeys(hit.key for hit in capability_hits(text)))


def capability_names(text: str) -> List[str]:
    """材料里命中的能力项名字（给人看的写法）。"""
    return [CAPABILITY_BY_KEY[key].name for key in capability_keys(text) if key in CAPABILITY_BY_KEY]


def group_hits(hits: Sequence[CapabilityHit]) -> List[Tuple[str, str, str, List[int], List[str]]]:
    """按能力项聚合：返回 (key, 名字, 类别, 命中行号, 命中的关键词)。

    能力词典匹配 Tool 与简报共用这一份聚合，保证「工具看到的」和「模型看到的」是同一份。
    """
    order: List[str] = []
    rows = {}
    for hit in hits:
        if hit.key not in rows:
            order.append(hit.key)
            rows[hit.key] = {
                "name": hit.name,
                "category": hit.category,
                "lines": [],
                "keywords": [],
            }
        entry = rows[hit.key]
        if hit.line_no and hit.line_no not in entry["lines"]:
            entry["lines"].append(hit.line_no)
        for keyword in hit.keywords:
            if keyword not in entry["keywords"]:
                entry["keywords"].append(keyword)
    return [
        (key, rows[key]["name"], rows[key]["category"], rows[key]["lines"], rows[key]["keywords"])
        for key in order
    ]


# ---- 要求 ↔ 事实 的对照 -----------------------------------------------------

@dataclass
class RequirementLink:
    """一条 JD 要求与简历事实的对照结果。"""

    requirement: Requirement
    facts: List[ResumeFact] = field(default_factory=list)          # 支持它的可当证据事实
    reverse_facts: List[ResumeFact] = field(default_factory=list)  # 提到但被挡掉的（反向证据）
    subs_hit: List[str] = field(default_factory=list)              # 已覆盖的子维度
    subs_missing: List[str] = field(default_factory=list)          # 还缺的子维度
    level: str = EVIDENCE_NONE                                     # 最高的证据等级

    @property
    def name(self) -> str:
        return self.requirement.name

    @property
    def level_label(self) -> str:
        return EVIDENCE_LABEL.get(self.level, self.level)

    @property
    def has_evidence(self) -> bool:
        return self.level != EVIDENCE_NONE

    @property
    def best_fact(self) -> Optional[ResumeFact]:
        return self.facts[0] if self.facts else None

    @property
    def weak(self) -> bool:
        """只有「仅提及」级证据：能用，但用词要降级。"""
        return self.level == EVIDENCE_MENTION

    @property
    def action_only(self) -> bool:
        """有动作没结果：容易被追问「做到什么程度」。"""
        return self.level == EVIDENCE_ACTION

    @property
    def refs(self) -> List[str]:
        return [fact.ref for fact in self.facts[:MAX_REFS]]

    @property
    def reverse_refs(self) -> List[str]:
        return [fact.ref for fact in self.reverse_facts[:MAX_REFS]]




def link_requirements(posting: Optional[JobPosting], facts: Sequence[ResumeFact]) -> List[RequirementLink]:
    """把每一条 JD 要求对上简历事实：支持的、反向的、以及缺掉的子维度。

    只认**能当证据**的事实（被否定 / 背景挡掉的另算反向证据）；等级取命中事实里最高的一档。
    """
    if posting is None:
        return []
    links: List[RequirementLink] = []
    for requirement in posting.requirements:
        key = requirement.capability_key
        support = [fact for fact in facts if fact.usable and key in fact.capabilities]
        reverse = [
            fact
            for fact in facts
            if key in fact.negated_keys or (fact.blocked and key in fact.capabilities)
        ]
        capability = CAPABILITY_BY_KEY.get(key)
        subs = capability.subs if capability else ()
        subs_hit = sorted({name for fact in support for name in match_sub_items(fact.text, subs)})
        subs_missing = [sub.name for sub in subs if sub.name not in subs_hit]
        level = EVIDENCE_NONE
        if support:
            level = max((fact.level for fact in support), key=lambda item: EVIDENCE_RANK.get(item, 0))
        links.append(
            RequirementLink(
                requirement=requirement,
                facts=sorted(support, key=lambda fact: -EVIDENCE_RANK.get(fact.level, 0)),
                reverse_facts=reverse,
                subs_hit=subs_hit,
                subs_missing=subs_missing,
                level=level,
            )
        )
    return links


# ---- 写作简报 ---------------------------------------------------------------

@dataclass
class PolishBrief:
    """一次 Polish Agent 运行的规则部分：要求、已核验事实、对照、缺口与风险。

    这是**唯一**送进模型的东西（`polish_llm.build_brief_payload` 就是它的 JSON 版），
    也是最终 Markdown 里「写作简报」那一段的数据来源。
    """

    doc: ResumeDoc
    summary: EvidenceSummary
    posting: Optional[JobPosting] = None
    links: List[RequirementLink] = field(default_factory=list)
    screen: Optional[object] = None            # ScreenReport（只用来数被挡掉的条目）
    jd_markdown: str = ""                      # JD 结构化 Tool 的输出（统一模板）
    source_line: str = ""                      # JD 来源写法
    notes: List[str] = field(default_factory=list)

    # -- 素材 --
    @property
    def requirements(self) -> List[Requirement]:
        return list(self.posting.requirements) if self.posting else []

    @property
    def usable_facts(self) -> List[ResumeFact]:
        return self.summary.usable

    @property
    def blocked_facts(self) -> List[ResumeFact]:
        return self.summary.blocked

    @property
    def jd_name(self) -> str:
        return self.posting.name if self.posting else "（JD 没解析出要求）"

    # -- 对照结论 --
    @property
    def core_links(self) -> List[RequirementLink]:
        """能靠简历证明的要求：学历 / 届别这类准入门槛不算。"""
        return [link for link in self.links if link.requirement.provable]

    @property
    def matched(self) -> List[RequirementLink]:
        return [link for link in self.core_links if link.has_evidence]

    @property
    def gaps(self) -> List[RequirementLink]:
        return [link for link in self.core_links if not link.has_evidence]

    @property
    def reverse(self) -> List[RequirementLink]:
        """JD 要、简历里却写着「没做过」的要求 —— 最该先处理的一类。"""
        return [link for link in self.core_links if link.reverse_facts]

    @property
    def weak(self) -> List[RequirementLink]:
        return [link for link in self.core_links if link.weak]

    @property
    def action_only(self) -> List[RequirementLink]:
        return [link for link in self.core_links if link.action_only]

    @property
    def coverage(self) -> float:
        if not self.core_links:
            return 0.0
        return len(self.matched) / len(self.core_links)

    @property
    def insufficient(self) -> bool:
        """规则部分撑不撑得住：JD 认不出要求，或者能当证据的事实太少。"""
        return not self.requirements or len(self.summary.usable) < MIN_FACTS

    # -- 给人看的几行 --
    def headline(self) -> str:
        if not self.requirements:
            return "JD 里没有用能力词典匹配出任何要求：这份简报没有目标，先换一份 JD。"
        return (
            f"{len(self.requirements)} 条要求里 {len(self.matched)} 条对上了可当证据的事实"
            f"（{len(self.core_links)} 条能靠简历证明的要求中覆盖 {self.coverage:.0%}）："
            f"{len(self.gaps)} 条没对上证据，{len(self.reverse)} 条正好写着「没做过」；"
            f"已核验事实 {len(self.summary.usable)} 条"
            f"（硬证据 {len(self.summary.solid)} / 弱证据 {len(self.summary.weak)}）。"
        )

    def summary_rows(self) -> List[Tuple[str, str]]:
        """给终端与前端看的摘要行（标签, 值）。"""
        return [
            ("JD", f"{self.jd_name}（{len(self.requirements)} 条要求）"),
            ("简历", f"{self.doc.name}（{len(self.summary.facts)} 条事实）"),
            (
                "可当证据",
                f"{len(self.summary.usable)} 条（硬证据 {len(self.summary.solid)} / "
                f"弱证据 {len(self.summary.weak)}）",
            ),
            ("要求覆盖", f"{len(self.matched)}/{len(self.core_links)}（{self.coverage:.0%}）"),
            ("缺口 / 反向证据", f"{len(self.gaps)} 条没对上 / {len(self.reverse)} 条写着没做过"),
            ("风险", f"{len(self.risks())} 条"),
        ]

    def risks(self) -> List[Risk]:
        """规则算出来的风险，顺序就是处理顺序：反向证据 > 缺口 > 弱证据 > 只有动作没结果。"""
        risks: List[Risk] = []
        for link in self.reverse:
            fact = link.reverse_facts[0]
            risks.append(
                Risk(
                    kind=RISK_CONFLICT,
                    text=(
                        f"JD 要「{link.name}」，简历里写的却是「{fact.quote_short}」"
                        "：这条别写，先补一段真实做过的经历"
                    ),
                    refs=[link.requirement.ref] + link.reverse_refs,
                )
            )
        for link in self.gaps:
            if link.reverse_facts:      # 反向证据上面已经报过一次：它不是「没提」，别再报成缺口
                continue
            risks.append(
                Risk(
                    kind=RISK_GAP,
                    text=(
                        f"「{link.name}」简历里完全没提"
                        f"（JD：{link.requirement.quote_short}）"
                    ),
                    refs=[link.requirement.ref],
                )
            )
        for link in self.weak:
            fact = link.best_fact
            risks.append(
                Risk(
                    kind=RISK_WORDING,
                    text=(
                        f"「{link.name}」只有「仅提及」级证据：写法用「了解 / 接触过」，"
                        "别用「熟练 / 精通」"
                    ),
                    refs=[fact.ref] if fact else [],
                )
            )
        for link in self.action_only:
            fact = link.best_fact
            risks.append(
                Risk(
                    kind=RISK_PROBE,
                    text=(
                        f"「{link.name}」有动作没结果：面试官会追问「做到什么程度 / 有没有数字」，"
                        "先想好答案"
                    ),
                    refs=[fact.ref] if fact else [],
                )
            )
        return risks[:MAX_RISKS]


def build_brief(
    doc: ResumeDoc,
    summary: EvidenceSummary,
    posting: Optional[JobPosting] = None,
    screen: Optional[object] = None,
    jd_markdown: str = "",
    source_line: str = "",
    notes: Sequence[str] = (),
) -> PolishBrief:
    """把「要求 + 事实 + 对照结论」组装成一份写作简报（对照在这里现算）。"""
    return PolishBrief(
        doc=doc,
        summary=summary,
        posting=posting,
        links=link_requirements(posting, doc.facts),
        screen=screen,
        jd_markdown=jd_markdown,
        source_line=source_line,
        notes=list(notes),
    )


def link_rows(brief: PolishBrief) -> List[List[object]]:
    """要求 ↔ 事实 的表格行（渲染交给上层，这里只给数据）。"""
    rows: List[List[object]] = []
    for index, link in enumerate(brief.core_links, start=1):
        rows.append(
            [
                index,
                link.name,
                link.requirement.level_label,
                link.level_label,
                len(link.facts),
                "、".join(link.subs_missing) or "—",
                link.requirement.ref,
            ]
        )
    return rows
