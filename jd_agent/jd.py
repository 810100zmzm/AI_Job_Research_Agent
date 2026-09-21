"""JD 解析：把一份岗位描述拆成「岗位真正需要的能力」，每条都带原文行号。

格式上尽量宽容：
  * `## 任职要求` / `**任职要求**` / `任职要求：` 都能认出章节
  * 一个文件里放了多个岗位时：load_jd 默认只取第一个（可用 --jd-title 指定），
    load_jd_positions 一次返回全部岗位（不指定 --jd 的批量模式走这条）
  * 同一项能力被多行提到时，只保留层级最高（必备 > 职责 > 加分）的那一条
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .lexicon import CAPABILITY_BY_KEY, match_capabilities
from .schema import LEVEL_DUTY, LEVEL_MUST, LEVEL_PLUS, JobPosting, Requirement

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?P<title>.+?)\s*#*\s*$")
JOB_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?P<title>岗位\s*[一二三四五六七八九十百千\d]+\s*[:：、.\-—]?.*)$"
)
SECTION_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?\*{0,2}\s*(?P<name>[^：:*#]{1,16}?)\s*\*{0,2}\s*[:：]?\s*$"
)
BULLET_RE = re.compile(r"^\s*(?:[-*+•·▪]|\d{1,2}\s*[.、)])\s+(?P<text>.*\S)\s*$")
META_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?\*{0,2}(?P<key>[^：:*#]{2,12}?)\*{0,2}\s*[:：]\s*(?P<value>\S.*?)\s*$"
)
RULE_RE = re.compile(r"^\s*(?:[-=*_—]{3,}|<[^>]+>)\s*$")

SECTION_LEVEL = {
    "岗位职责": LEVEL_DUTY,
    "工作职责": LEVEL_DUTY,
    "职责": LEVEL_DUTY,
    "工作内容": LEVEL_DUTY,
    "主要职责": LEVEL_DUTY,
    "岗位内容": LEVEL_DUTY,
    "任职要求": LEVEL_MUST,
    "岗位要求": LEVEL_MUST,
    "任职资格": LEVEL_MUST,
    "任职条件": LEVEL_MUST,
    "职位要求": LEVEL_MUST,
    "要求": LEVEL_MUST,
    "我们希望你": LEVEL_MUST,
    "加分项": LEVEL_PLUS,
    "加分": LEVEL_PLUS,
    "优先条件": LEVEL_PLUS,
    "优先": LEVEL_PLUS,
    "nicetohave": LEVEL_PLUS,
}

SECTION_NAME = {LEVEL_DUTY: "岗位职责", LEVEL_MUST: "任职要求", LEVEL_PLUS: "加分项"}

META_KEYS = {
    "公司", "公司名称", "企业", "公司简介",
    "工作地点", "地点", "城市", "办公地点",
    "岗位类型", "招聘类型", "职位类型", "招聘对象", "届别",
    "薪资", "薪资范围", "发布时间", "备注",
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def display_path(path) -> str:
    """报告里显示的路径：能相对项目根目录就相对，否则用绝对路径。"""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def read_text(path) -> str:
    """按常见编码读取文本，避免 Windows 下的编码坑。"""
    data = Path(path).read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gbk", "utf-16"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.strip()
    text = (
        text.replace("\\-", "-")
        .replace("\\_", "_")
        .replace("\\*", "*")
        .replace("\\", "")
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*[-*+•·▪]\s+", "", text)
    return text.strip()


def norm_section(name: str) -> str:
    name = name.strip().strip("*").strip()
    name = re.sub(r"^[一二三四五六七八九十\d]+[、.．)）]\s*", "", name)
    return name.replace(" ", "").rstrip("：:").lower()


def first_heading(lines: Sequence[str]) -> str:
    for raw in lines:
        match = HEADING_RE.match(raw)
        if match:
            return clean_text(match.group("title"))
    return ""


def _split_positions(lines: Sequence[str]) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """按「岗位一 / 岗位二」这类标题切分；没有则整体作为一个岗位。"""
    blocks: List[Tuple[str, List[Tuple[int, str]]]] = []
    title = ""
    body: List[Tuple[int, str]] = []
    file_title = first_heading(lines)
    for line_no, raw in enumerate(lines, start=1):
        match = JOB_HEADING_RE.match(raw)
        if match:
            # 第一个岗位标题之前的内容是文件级前言（标题、说明），不算一个岗位
            if title:
                blocks.append((title, body))
            title = clean_text(match.group("title"))
            body = []
            continue
        body.append((line_no, raw))
    if title or body:
        blocks.append((title or file_title, body))
    return blocks


def _dedupe(requirements: Sequence[Requirement]) -> List[Requirement]:
    """同一项能力保留层级最高的一条（必备 > 职责 > 加分），顺序按原文行号。"""
    best: Dict[str, Requirement] = {}
    for item in requirements:
        current = best.get(item.capability_key)
        if current is None or item.weight > current.weight:
            best[item.capability_key] = item
    return sorted(best.values(), key=lambda item: item.line_no)


def parse_jd_text(text: str, source_file: str, jd_id: str = "JD01", select: str = "") -> JobPosting:
    lines = text.splitlines()
    blocks = _split_positions(lines)
    titles = [title for title, _ in blocks]

    if select:
        chosen: Optional[Tuple[str, List[Tuple[int, str]]]] = next(
            (block for block in blocks if select in block[0]), None
        )
        if chosen is None:
            raise ValueError(
                f"JD 里没有匹配「{select}」的岗位；这份文件里可选的是：{'、'.join(t for t in titles if t) or '（没有识别到岗位标题）'}"
            )
    else:
        chosen = blocks[0]

    return _build_posting(chosen, lines, source_file, jd_id, len(blocks), titles)


def parse_jd_positions(text: str, source_file: str) -> List[JobPosting]:
    """把一个文件里的**所有**岗位都解析出来（顺序与文件一致）。

    没有指定 --jd 时走这条路径：input/jd 下每个文件、每个岗位各出一份报告。
    """
    lines = text.splitlines()
    blocks = _split_positions(lines)
    titles = [title for title, _ in blocks]
    stem = Path(source_file).stem
    return [
        _build_posting(block, lines, source_file, f"{stem}#{index}", len(blocks), titles)
        for index, block in enumerate(blocks, start=1)
    ]


def _build_posting(
    chosen: Tuple[str, List[Tuple[int, str]]],
    lines: Sequence[str],
    source_file: str,
    jd_id: str,
    position_count: int,
    titles: Sequence[str],
) -> JobPosting:
    title, body = chosen
    requirements: List[Requirement] = []
    company = ""
    job_type = ""
    current_level = LEVEL_DUTY
    matched_lines = 0
    bullet_lines = 0

    for line_no, raw in body:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if RULE_RE.match(line) or stripped.startswith(">") or stripped.startswith("|"):
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            name = norm_section(section_match.group("name"))
            if name in SECTION_LEVEL:
                current_level = SECTION_LEVEL[name]
                continue

        if stripped.startswith("#") and not BULLET_RE.match(line):
            continue

        bullet = BULLET_RE.match(line)
        content = clean_text(bullet.group("text") if bullet else line)
        if not content:
            continue
        if bullet:
            bullet_lines += 1

        meta = META_RE.match(content)
        if meta and norm_section(meta.group("key")) in META_KEYS:
            key = norm_section(meta.group("key"))
            value = clean_text(meta.group("value"))
            if key in ("公司", "公司名称", "企业") and not company:
                company = value
            elif key in ("岗位类型", "招聘类型", "职位类型") and not job_type:
                job_type = value
            continue

        keys = match_capabilities(content)
        if not keys:
            continue
        matched_lines += 1
        section = SECTION_NAME.get(current_level, "")
        for key in keys:
            capability = CAPABILITY_BY_KEY[key]
            requirements.append(
                Requirement(
                    capability_key=key,
                    capability_name=capability.name,
                    category=capability.category,
                    level=current_level,
                    section=section,
                    line_no=line_no,
                    quote=content,
                    source_file=source_file,
                )
            )

    return JobPosting(
        jd_id=jd_id,
        title=title or Path(source_file).name,
        source_file=source_file,
        company=company,
        job_type=job_type,
        line_count=len(lines),
        bullet_count=matched_lines,
        position_count=position_count,
        position_titles=[t for t in titles if t],
        requirements=_dedupe(requirements),
    )


def load_jd(path, select: str = "") -> JobPosting:
    path = Path(path)
    return parse_jd_text(read_text(path), source_file=display_path(path), select=select)


def load_jd_positions(path) -> List[JobPosting]:
    """读一个 JD 文件，返回里面所有岗位。"""
    path = Path(path)
    return parse_jd_positions(read_text(path), source_file=display_path(path))
