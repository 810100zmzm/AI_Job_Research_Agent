"""证据匹配：把「岗位真正需要的能力」逐条拿到项目里去找支撑。

两种口径，对应 Agent Loop 里的一次 Adjust：
    严格  strict  —— 只认「有动作 / 有结果」的事实，技术栈罗列不算证据
    放宽  relaxed —— 允许把「仅提及」级事实计为弱证据，避免把空白误判成「没做过」
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .lexicon import CAPABILITY_BY_KEY, match_sub_items
from .schema import (
    EVIDENCE_ACTION,
    EVIDENCE_LABEL,
    EVIDENCE_MENTION,
    EVIDENCE_NONE,
    EVIDENCE_RANK,
    EVIDENCE_RESULT,
    EvidenceMatch,
    Project,
    ProjectFact,
    Requirement,
)

SOLID_LEVELS = (EVIDENCE_RESULT, EVIDENCE_ACTION)


def _sub_coverage(requirement: Requirement, facts: Sequence[ProjectFact]) -> Tuple[List[str], List[str]]:
    """岗位强调的子维度（比如 PyTorch / TensorFlow）在项目里对上了哪些。"""
    capability = CAPABILITY_BY_KEY.get(requirement.capability_key)
    if capability is None or not capability.subs:
        return [], []
    text = " ".join(fact.text for fact in facts)
    matched = match_sub_items(text, capability.subs)
    all_subs = [sub.name for sub in capability.subs]
    return matched, [name for name in all_subs if name not in matched]


def match_requirement(requirement: Requirement, project: Project, allow_mention: bool = False) -> EvidenceMatch:
    """对一条岗位要求做检索：命中的事实、证据等级、还缺哪个子维度。"""
    facts = project.facts_for(requirement.capability_key)
    usable = [f for f in facts if f.level in SOLID_LEVELS or allow_mention]
    usable.sort(key=lambda fact: -EVIDENCE_RANK.get(fact.level, 0))

    level = EVIDENCE_NONE
    for fact in usable:
        if EVIDENCE_RANK.get(fact.level, 0) > EVIDENCE_RANK[level]:
            level = fact.level

    negated = [f for f in facts if requirement.capability_key in f.negated]
    matched_subs, missing_subs = _sub_coverage(requirement, usable or facts)

    note = ""
    if level == EVIDENCE_NONE and negated:
        note = "项目描述里明确写了「没做过」，不能当成证据"
    elif level == EVIDENCE_MENTION:
        note = "只有技术栈级的弱证据（放宽规则后才匹配到）"
    elif level != EVIDENCE_NONE and matched_subs:
        note = f"岗位强调：{'、'.join(matched_subs)}"

    return EvidenceMatch(
        requirement=requirement,
        facts=usable,
        level=level,
        matched_subs=matched_subs,
        missing_subs=missing_subs,
        relaxed=level == EVIDENCE_MENTION,
        note=note,
    )


def match_requirements(
    requirements: Sequence[Requirement], project: Project, allow_mention: bool = False
) -> List[EvidenceMatch]:
    return [match_requirement(requirement, project, allow_mention) for requirement in requirements]


def level_text(level: str) -> str:
    return EVIDENCE_LABEL.get(level, level)
