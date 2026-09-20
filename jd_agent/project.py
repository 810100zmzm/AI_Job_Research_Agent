"""项目描述解析：把项目文本拆成「可核验的事实条目」，并给每条打证据等级。

证据等级（判断的基石，规则完全公开，不依赖大模型）：
    有结果 result —— 有量化指标（"准确率 85%"），或有交付动作（"已开源 / 已上线"）且能看出是你做的
    有动作 action —— 有「搭建 / 实现 / 调试 / 调优」这类动作词，但没写结果
    仅提及 mention —— 只出现在技术栈罗列里，或只有「用过 / 了解」

另外单独记录「否定句」：写了「没做过 X」的地方，绝不能被当成 X 的证据。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

from .jd import clean_text, display_path, first_heading, read_text
from .lexicon import match_capabilities
from .schema import (
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_RESULT,
    Project,
    ProjectFact,
)

# 「这是我做的」的信号
OWNERSHIP_MARKERS = (
    "我", "本人", "自己", "独立", "主导", "负责", "牵头", "独自", "个人",
)

# 「我真的动过手」的信号
ACTION_MARKERS = (
    "搭建", "实现", "开发", "接入", "封装", "调试", "优化", "重构", "跑通", "训练", "微调",
    "评测", "对比", "修复", "排查", "设计", "编写", "写了", "完成", "部署", "上线", "迁移",
    "调参", "测试", "验证", "整理", "落地", "复现", "集成", "选型", "调研", "清洗", "标注",
    "做了", "做过", "搭建",
)

# 「有交付物」的信号
DELIVERY_MARKERS = (
    "开源", "上线", "发布", "部署", "交付", "提测", "验收", "获奖", "作品集", "demo", "readme",
)

# 这些词组里的动词不算交付（「发布时间」不是「发布了」）
DELIVERY_NOISE = ("发布时间", "发布日期", "更新日期")

# 只认真正代表结果的量化写法，避免把「分了 3 个模块」当成指标
METRIC_RE = re.compile(
    r"(?:从|由)\s*\d+(?:\.\d+)?\s*(?:%|％)?\s*(?:提升|提高|增长|涨到|下降到|降到|降低|减少)"
    r"|\d+(?:\.\d+)?\s*(?:%|％)"
    r"|\d+(?:\.\d+)?\s*(?:倍|毫秒|ms|秒|分钟|小时|万条|条数据|人日)"
    r"|(?:准确率|召回率|精确率|命中率|f1|响应时间|耗时|吞吐|qps|覆盖率|通过率|满意度)\D{0,6}\d+(?:\.\d+)?",
    re.IGNORECASE,
)

BULLET_RE = re.compile(r"^\s*(?:[-*+•·▪]|\d{1,2}\s*[.、)])\s+(?P<text>.*\S)\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?P<title>.+?)\s*#*\s*$")
BOLD_RE = re.compile(r"^\s*\*{2}(?P<title>[^*]{1,24})\*{2}\s*[:：]?\s*$")
RULE_RE = re.compile(r"^\s*(?:[-=*_—]{3,}|<[^>]+>)\s*$")


# 背景段落只算背景，不作为证据
BACKGROUND_MARKERS = ("背景", "目标", "痛点", "需求", "动机", "简介", "介绍", "why")


def _metrics(text: str) -> List[str]:
    return [" ".join(match.group(0).split()) for match in METRIC_RE.finditer(text)]


def classify(text: str, section: str = "") -> Tuple[str, List[str], bool]:
    """返回 (证据等级, 量化指标, 是否有个人贡献表述)。"""
    lowered = text.lower()
    for noise in DELIVERY_NOISE:
        lowered = lowered.replace(noise, "")
    metrics = _metrics(text)
    has_ownership = any(marker in text for marker in OWNERSHIP_MARKERS)
    has_action = any(marker in text for marker in ACTION_MARKERS)
    has_delivery = any(marker in lowered for marker in DELIVERY_MARKERS)

    # 背景 / 目标段落里的数字（「人工整理要 3 小时」）不是交付证据
    if any(marker in section.lower() for marker in BACKGROUND_MARKERS):
        return EVIDENCE_MENTION, metrics, has_ownership

    if metrics or (has_delivery and (has_action or has_ownership)):
        level = EVIDENCE_RESULT
    elif has_action or has_ownership or has_delivery:
        level = EVIDENCE_ACTION
    else:
        level = EVIDENCE_MENTION
    return level, metrics, has_ownership


def make_fact(text: str, section: str, line_no: int, source_file: str) -> ProjectFact:
    positive = match_capabilities(text, negation=True)
    negated = [key for key in match_capabilities(text) if key not in positive]
    level, metrics, has_ownership = classify(text, section)
    return ProjectFact(
        text=text,
        section=section,
        line_no=line_no,
        source_file=source_file,
        capabilities=positive,
        negated=negated,
        level=level,
        has_metric=bool(metrics),
        metrics=metrics,
        has_ownership=has_ownership,
    )


def parse_project_text(text: str, source_file: str, title: str = "") -> Project:
    lines = text.splitlines()
    facts: List[ProjectFact] = []
    section = "项目描述"
    file_title = first_heading(lines)

    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or RULE_RE.match(line) or stripped.startswith(">"):
            continue

        heading = HEADING_RE.match(line) or BOLD_RE.match(line)
        if heading:
            name = clean_text(heading.group("title"))
            if name:
                section = name
            continue

        bullet = BULLET_RE.match(line)
        content = clean_text(bullet.group("text") if bullet else line)
        if not content:
            continue
        facts.append(make_fact(content, section, line_no, source_file))

    resolved_title = title or file_title or Path(source_file).stem
    return Project(
        title=resolved_title,
        source_file=source_file,
        line_count=len(lines),
        facts=facts,
        text=text,
    )


def load_project(path, title: str = "") -> Project:
    path = Path(path)
    return parse_project_text(read_text(path), source_file=display_path(path), title=title)


def apply_answer(project: Project, answer: str, source: str = "用户补充说明") -> int:
    """把用户对 Ask 的回答并入证据池，返回新增的事实条数。"""
    chunks: Sequence[str] = [
        chunk.strip(" -•·*")
        for chunk in re.split(r"[\n;；。]+", answer)
        if chunk.strip(" -•·*")
    ]
    if not chunks:
        return 0
    for chunk in chunks:
        project.facts.append(make_fact(chunk, source, 0, source))
    return len(chunks)
