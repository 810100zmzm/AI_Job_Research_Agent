"""Polish Agent（LangGraph）：基于「已核验事实 + JD 要求」生成建议写法 / 追问预演 / 润色。

图长这样（节点名就是 Trace 里的 Action；两路来源分别排队）：

    detect ─┬─ JD 路：文本 / 文件 / 图片 / 网址 ─┐
            └─ 简历路：文本 / 文件 / 图片 / 网址 ─┘┄┄►（两路都取完）
                     structure_jd ─► structure_cv ─► screen ─► evidence ─┬─► cap_match ─┬─► suggest ─► interview ─► polish ─► finalize
                                                                        │              └─（要求认不出 / 事实太少）─► ask
                                                严格口径一条硬证据都没有 ─► relax ─► evidence

分工是这一层的重点：**工具负责「模型能看到什么」，模型只负责「怎么说」**。

    structure_jd   JD 结构化 Tool（jd-struct）      JD 原文 → 统一模板（岗位标题 + 元信息 + 章节）
    structure_cv   简历结构化 Tool（resume-struct）  简历原文 → 章节 + 事实条目（带原文行号）
    screen         否定 / 背景识别 Tool              先把「写了但没做过 / 只是背景」挡掉
    evidence       证据等级判定 Tool                 剩下的条目才判 有结果 / 有动作 / 仅提及
    cap_match      能力词典匹配 Tool                 要求 ↔ 事实 的对照、缺口、反向证据

判定顺序仍然是「先挡伪装、再判等级、最后才让模型开口」：`relax`（Adjust）只放宽证据强弱，
否定与背景永远不放宽；JD 认不出要求、或者一条能当证据的事实都没有，就直接 Ask —— 不烧调用次数。

每走一步都往 `state.trace` 里落一条，四要素缺一不可：
    Action（做了什么）→ Observation（真实看到了什么）→ State Update（状态改成了什么）
    → Decision（只能是 Continue / Adjust / Ask / Stop）

调用纪律复用主线的账本（`generate.CallBudget`）：单次 30 秒、失败重试 1 次、
一次运行最多 `--llm-max-calls`（默认 10）次；每块内容失败只降级成「未启用 / 调用失败 / 已丢弃」，
**不改 StopReason、不改退出码** —— 简报（规则部分）照出。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from langgraph.graph import END, StateGraph

from .agent import build_text_budget
from .generate import CallBudget
from ..domain.jd import display_path, parse_jd_text
from ..domain.jd_html import clean_lines, count_lines, html_to_markdown
from ..services.jd_source import (
    JD_IMAGE_SYSTEM_PROMPT,
    JD_IMAGE_USER_PROMPT,
    SOURCE_FILE,
    SOURCE_IMAGE,
    SOURCE_TEXT,
    SOURCE_URL,
    JDSource,
    SourceError,
    describe_sources,
    detect_sources,
    fetch_html,
    image_to_markdown,
    unusable_notes,
)
from ..core.llm import DEFAULT_CALL_LIMIT, DEFAULT_QWEN_VL_MODEL, OpenAICompatClient
from .polish_brief import MIN_FACTS, PolishBrief, build_brief, link_rows
from .polish_llm import (
    TASK_ACTION,
    TASK_DUTY,
    build_user_message,
    new_report,
    run_task,
    task_observation,
    task_state_update,
    unavailable_report,
)
from ..domain.resume_facts import EVIDENCE_LABEL, extract_resume, summarize, table_lines
from .resume_graph import RESUME_IMAGE_SYSTEM_PROMPT, RESUME_IMAGE_USER_PROMPT
from ..core.schema import (
    DECISION_ADJUST,
    DECISION_ASK,
    DECISION_CONTINUE,
    DECISION_STOP,
    DECISIONS,
    SOURCE_LLM_INTERVIEW,
    SOURCE_LLM_POLISH,
    SOURCE_LLM_SUGGEST,
    JobPosting,
    LlmBlock,
    LlmReport,
    TraceStep,
)
from ..core.settings import LLMSettings
from ..tools.cap_match import CAP_MATCH_TOOL
from ..tools.jd_files import JD_FILE_TOOL, KIND_LABEL, parse_file_markdown
from ..tools.jd_struct import JD_STRUCT_TOOL, render_jd_markdown, structure_jd_text
from ..tools.resume_evidence import LEVEL_ORDER, RESUME_EVIDENCE_TOOL, render_evidence_table
from ..tools.resume_negation import (
    RESUME_NEGATION_TOOL,
    ScreenReport,
    render_blocked_table,
    screen_facts,
)
from ..tools.resume_struct import RESUME_STRUCT_TOOL

# 判定口径（写死在这里，方便直接检查，也方便日后调）
MIN_SOLID_FACTS = 1     # 严格口径下硬证据少于这个数（一条都没有）：换放宽口径重判一次
DEFAULT_STEM = "polish"

SIDE_JD = "jd"
SIDE_CV = "cv"
SIDE_LABEL = {SIDE_JD: "JD", SIDE_CV: "简历"}

NODE_NAMES = (
    "detect",       # 识别 JD / 简历两路来源
    "read_text",    # 直接读取两路文本
    "parse_file",   # 解析 JD / 简历文件
    "ocr_image",    # 转录 JD / 简历图片
    "fetch_url",    # 抓取 JD / 简历网页
    "structure_jd", # 结构化 JD 要求
    "structure_cv", # 结构化简历事实
    "screen",       # 拦截否定 / 背景事实
    "evidence",     # 判定证据等级
    "cap_match",    # 匹配能力要求与事实
    "relax",        # 放宽证据口径后重判
    "suggest",      # 生成建议写法
    "interview",    # 生成追问预演
    "polish",       # 润色简历表述
    "finalize",     # 生成并写出最终报告
    "ask",          # 资料不足时提问
)

# 条件边的路由表：来源类型 -> 该走哪个节点（"structure_jd" 表示两路队列都空了）
ROUTES = {
    SOURCE_TEXT: "read_text",
    SOURCE_FILE: "parse_file",
    SOURCE_IMAGE: "ocr_image",
    SOURCE_URL: "fetch_url",
    "structure_jd": "structure_jd",
}

# 三块 LLM 任务在 Trace 里的顺序（建议写法 -> 追问预演 -> 润色）
LLM_STEPS = (SOURCE_LLM_SUGGEST, SOURCE_LLM_INTERVIEW, SOURCE_LLM_POLISH)
LLM_REASON = (
    "三块内容彼此独立：这一块失败只影响这一块，简报（规则部分）与退出码都不变；"
    "接下来还有别的块要跑"
)


def _clip(text: object, limit: int = 120) -> str:
    """压成一行并截断：Trace 里的信息不该把一行撑爆。"""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ---- 状态 -------------------------------------------------------------------

@dataclass
class PolishRequest:
    """用户给的输入：两路来源（JD 一条链、简历一条链），各自四种取法 + 可选标题。"""

    jd_text: str = ""
    jd_files: Tuple[str, ...] = ()
    jd_images: Tuple[str, ...] = ()
    jd_urls: Tuple[str, ...] = ()
    jd_title: str = ""
    cv_text: str = ""
    cv_files: Tuple[str, ...] = ()
    cv_images: Tuple[str, ...] = ()
    cv_urls: Tuple[str, ...] = ()
    cv_title: str = ""

    def empty_side(self, side: str) -> bool:
        if side == SIDE_JD:
            return not (self.jd_text.strip() or self.jd_files or self.jd_images or self.jd_urls)
        return not (self.cv_text.strip() or self.cv_files or self.cv_images or self.cv_urls)

    @property
    def empty(self) -> bool:
        return self.empty_side(SIDE_JD) and self.empty_side(SIDE_CV)

    @property
    def missing_sides(self) -> List[str]:
        return [SIDE_LABEL[side] for side in (SIDE_JD, SIDE_CV) if self.empty_side(side)]


@dataclass
class PolishChunk:
    """一个来源取到的 Markdown 原文（note 非空表示这个来源没取到）。"""

    kind: str
    ref: str
    label: str
    markdown: str = ""
    key: str = ""              # 原始 key（路径 / 网址），出错时用来回查
    note: str = ""
    raw_lines: int = 0
    dropped: int = 0

    @property
    def line_count(self) -> int:
        return count_lines(self.markdown)

    @property
    def empty(self) -> bool:
        return not self.markdown.strip()


@dataclass
class PolishAgentState:
    """Polish Agent 的完整状态，也是唯一的数据出口（Markdown / 终端 / Trace 都由它渲染）。"""

    request: PolishRequest
    jd_sources: List[JDSource] = field(default_factory=list)
    cv_sources: List[JDSource] = field(default_factory=list)
    jd_pending: List[JDSource] = field(default_factory=list)
    cv_pending: List[JDSource] = field(default_factory=list)
    jd_chunks: List[PolishChunk] = field(default_factory=list)
    cv_chunks: List[PolishChunk] = field(default_factory=list)
    pages: Dict[str, str] = field(default_factory=dict)      # 网址 -> 已抓到的 Markdown
    trace: List[TraceStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    posting: Optional[JobPosting] = None                     # JD 要求（能力词典匹配出来的）
    jd_markdown: str = ""                                    # JD 结构化 Tool 的输出（统一模板）
    doc: Optional[object] = None                             # ResumeDoc
    screen: Optional[ScreenReport] = None
    summary: Optional[object] = None                         # EvidenceSummary
    brief: Optional[PolishBrief] = None
    llm: Optional[LlmReport] = None
    budget: Optional[CallBudget] = None
    markdown: str = ""
    question: str = ""
    question_reason: str = ""
    stop_reason: str = ""
    relaxed: bool = False
    needs_relax: bool = False
    output_dir: str = ""
    stem: str = DEFAULT_STEM
    generated_at: str = ""
    written: List[str] = field(default_factory=list)
    settings: Optional[LLMSettings] = None
    text_client: Optional[OpenAICompatClient] = None
    vision_client: Optional[OpenAICompatClient] = None
    fetcher: Optional[Callable] = None
    llm_max_calls: int = DEFAULT_CALL_LIMIT

    # -- 给外部读的几个判断 --
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
    def facts(self) -> List[object]:
        return list(self.doc.facts) if self.doc else []

    @property
    def usable_count(self) -> int:
        return len(self.summary.usable) if self.summary else 0

    @property
    def solid_count(self) -> int:
        return len(self.summary.solid) if self.summary else 0

    @property
    def blocked_count(self) -> int:
        return len(self.summary.blocked) if self.summary else 0

    @property
    def requirement_count(self) -> int:
        return len(self.posting.requirements) if self.posting else 0

    @property
    def thin(self) -> bool:
        """简报撑不撑得住：JD 认不出要求，或者能当证据的事实太少。"""
        return self.brief is None or self.brief.insufficient

    def chunks(self, side: str) -> List[PolishChunk]:
        return self.jd_chunks if side == SIDE_JD else self.cv_chunks

    def usable_chunks(self, side: str) -> List[PolishChunk]:
        return [chunk for chunk in self.chunks(side) if not chunk.empty]

    def failed_chunks(self, side: str) -> List[PolishChunk]:
        return [chunk for chunk in self.chunks(side) if chunk.note]

    def line_count(self, side: str) -> int:
        return sum(chunk.line_count for chunk in self.usable_chunks(side))

    def source_line(self, side: str) -> str:
        """这一路的来源怎么写：只有一个文件来源时用文件路径（行号就是文件里的行号）。"""
        chunks = self.usable_chunks(side)
        if len(chunks) == 1 and chunks[0].kind == SOURCE_FILE:
            return display_path(chunks[0].key) or chunks[0].label
        labels = [source.label for source in self.sources(side) if source.usable]
        return "、".join(dict.fromkeys(labels)) or f"（没有可用的{SIDE_LABEL[side]}来源）"

    def sources(self, side: str) -> List[JDSource]:
        return self.jd_sources if side == SIDE_JD else self.cv_sources


def _record(
    state: PolishAgentState, action: str, observation: str, state_update: str, decision: str, reason: str
) -> None:
    """落一条 Trace：四要素缺一不可，Decision 只能是那四种。"""
    if decision not in DECISIONS:
        raise ValueError(f"Decision 只能是 {DECISIONS}，收到：{decision}")
    state.trace.append(
        TraceStep(
            index=len(state.trace) + 1,
            action=action,
            observation=observation,
            state_update=state_update,
            decision=decision,
            decision_reason=reason,
        )
    )


# ---- 节点：识别与取证（两路来源各走一遍同一套节点）--------------------------

def _take(state: PolishAgentState, kind: str) -> Tuple[Optional[JDSource], str]:
    """取出队头来源：JD 路优先（route_next 已经保证队头就是这一类）。"""
    for side in (SIDE_JD, SIDE_CV):
        queue = state.jd_pending if side == SIDE_JD else state.cv_pending
        if queue and queue[0].kind == kind:
            return queue.pop(0), side
    return None, ""


def _put(state: PolishAgentState, side: str, chunk: PolishChunk) -> None:
    state.chunks(side).append(chunk)


def node_detect(state: PolishAgentState) -> PolishAgentState:
    """识别两路输入类型：哪些来源可用、每个走哪条分支。"""
    parts: List[str] = []
    for side in (SIDE_JD, SIDE_CV):
        sources = state.sources(side)
        usable = [source for source in sources if source.usable]
        skipped = [source for source in sources if not source.usable]
        if side == SIDE_JD:
            state.jd_pending = list(usable)
        else:
            state.cv_pending = list(usable)
        state.notes += unusable_notes(skipped)
        text = (
            f"{SIDE_LABEL[side]} {len(sources)} 个来源（{describe_sources(sources) or '无'}）："
            f"可用 {len(usable)} 个"
        )
        if skipped:
            text += f"，跳过 {len(skipped)} 个（{'；'.join(_clip(source.note, 60) for source in skipped)}）"
        parts.append(text)
    _record(
        state,
        "DetectSource",
        "；".join(parts),
        f"state.jd_pending = {len(state.jd_pending)} 个；state.cv_pending = {len(state.cv_pending)} 个",
        DECISION_CONTINUE,
        "两路来源分别排队：JD 那路定要求、简历那路出事实；文本与文件不联网、不读 .env",
    )
    return state


def node_read_text(state: PolishAgentState) -> PolishAgentState:
    """文本分支：直接读取（不联网、不调模型）。"""
    source, side = _take(state, SOURCE_TEXT)
    if source is None:
        return state
    markdown, dropped = clean_lines(source.raw, strict=False)
    chunk = PolishChunk(
        kind=SOURCE_TEXT,
        ref=source.ref,
        label=source.label,
        markdown=markdown,
        raw_lines=len(source.raw.splitlines()),
        dropped=dropped,
    )
    _put(state, side, chunk)
    _record(
        state,
        "ReadText",
        f"直接读取{source.label}（{SIDE_LABEL[side]}路）：原文 {chunk.raw_lines} 行 -> 有效 "
        f"{chunk.line_count} 行（合掉 {dropped} 行空行 / 重复行），没调模型、没联网",
        f"state.{side}_chunks += 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "文本是最可靠的一手来源：原样读取就够，不需要模型",
    )
    return state


def node_parse_file(state: PolishAgentState) -> PolishAgentState:
    """文件分支：交给工具区的文件解析 Tool（纯规则、不联网）。"""
    source, side = _take(state, SOURCE_FILE)
    if source is None:
        return state
    try:
        parsed = parse_file_markdown(source.raw)
    except ValueError as exc:
        _put(state, side, PolishChunk(SOURCE_FILE, source.ref, source.label, note=str(exc)))
        state.notes.append(f"{source.ref}：{exc}")
        _record(
            state,
            "ParseFile",
            f"文件解析失败（{SIDE_LABEL[side]}路）：{_clip(exc)}",
            f"state.{side}_chunks += 1 段（空）；state.notes += 1",
            DECISION_CONTINUE,
            "这个文件读不出来，但队列里还有别的来源：先跑完，最后如实写进 StopReason",
        )
        return state
    chunk = PolishChunk(
        kind=SOURCE_FILE,
        ref=source.ref,
        label=source.label,
        markdown=parsed.markdown,
        key=source.raw,
        dropped=parsed.dropped,
    )
    _put(state, side, chunk)
    _record(
        state,
        "ParseFile",
        f"文件解析 Tool（{JD_FILE_TOOL.slug}）读 {source.ref}（{SIDE_LABEL[side]}路）："
        f"{KIND_LABEL.get(parsed.kind, parsed.kind)} -> 有效 {chunk.line_count} 行（丢掉 {parsed.dropped} 行）",
        f"state.{side}_chunks += 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "文件解析复用工具区的纯规则实现：不联网、不调模型，命令行与前端同一份口径",
    )
    return state


def node_ocr_image(state: PolishAgentState) -> PolishAgentState:
    """图片分支：Qwen-VL 逐字转成 Markdown（联网）；JD 与简历用各自的转录提示词。"""
    source, side = _take(state, SOURCE_IMAGE)
    if source is None:
        return state
    index = 1 + len([chunk for chunk in state.chunks(side) if chunk.kind == SOURCE_IMAGE])
    model = _vision_model(state)
    system_prompt, user_prompt = _image_prompts(side)
    try:
        markdown = image_to_markdown(
            source.raw,
            state.vision_client,
            model,
            ref=source.ref,
            index=index,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except SourceError as exc:
        _put(state, side, PolishChunk(SOURCE_IMAGE, source.ref, source.label, note=str(exc)))
        state.notes.append(f"{source.ref}：{exc}")
        _record(
            state,
            "OcrImage",
            f"图片转录失败（{SIDE_LABEL[side]}路）：{_clip(exc)}",
            f"state.{side}_chunks += 1 段（空）；state.notes += 1",
            DECISION_CONTINUE,
            "一张图读不出来不影响别的来源：继续跑，最后在 StopReason 里说明这张图没算进去",
        )
        return state
    markdown, dropped = clean_lines(markdown, strict=False)
    chunk = PolishChunk(
        kind=SOURCE_IMAGE,
        ref=source.ref,
        label=source.label,
        markdown=markdown,
        key=source.raw,
        dropped=dropped,
    )
    _put(state, side, chunk)
    _record(
        state,
        "OcrImage",
        f"Qwen-VL（{model}）把{source.label}逐字转成 Markdown（{SIDE_LABEL[side]}路）："
        f"{chunk.line_count} 行；数字与指标请对照原图复核",
        f"state.{side}_chunks += 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "图片内容只有模型能读，但转写只做逐字转录：不翻译、不总结、不补内容",
    )
    return state


def node_fetch_url(state: PolishAgentState) -> PolishAgentState:
    """网址分支：抓一次 HTML -> Markdown。"""
    source, side = _take(state, SOURCE_URL)
    if source is None:
        return state
    cached = state.pages.get(source.raw)
    try:
        if cached is None:
            html, final_url = fetch_html(source.raw, opener=state.fetcher)
            cached = html_to_markdown(html)
            state.pages[source.raw] = cached
            fetched = f"抓取 1 次 HTTP（{final_url}）"
        else:
            fetched = "复用已抓到的内容（不再发请求）"
        markdown, dropped = clean_lines(cached, strict=False)
    except SourceError as exc:
        _put(state, side, PolishChunk(SOURCE_URL, source.ref, source.label, key=source.raw, note=str(exc)))
        state.notes.append(f"{source.ref}：{exc}")
        _record(
            state,
            "FetchUrl",
            f"抓取失败（{SIDE_LABEL[side]}路）：{_clip(exc)}",
            f"state.{side}_chunks += 1 段（空）；state.notes += 1",
            DECISION_CONTINUE,
            "页面抓不到不是输入错：先跑完别的来源，最后如实写进 StopReason，并提示改给文本或截图",
        )
        return state

    chunk = PolishChunk(
        kind=SOURCE_URL,
        ref=source.ref,
        label=source.label,
        markdown=markdown,
        key=source.raw,
        raw_lines=count_lines(cached),
        dropped=dropped,
    )
    _put(state, side, chunk)
    _record(
        state,
        "FetchUrl",
        f"{fetched}（{SIDE_LABEL[side]}路）：HTML 转 Markdown 后有效 {chunk.line_count} 行（丢掉 {dropped} 行）",
        f"state.{side}_chunks += 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "页面上的文字尽量留下：宁可多留也不漏，抓不到就让你补文本或截图",
    )
    return state


# ---- 节点：JD 侧（要求）------------------------------------------------------

def node_structure_jd(state: PolishAgentState) -> PolishAgentState:
    """JD 侧：结构化 Tool 规整成统一模板，要求仍从原文抽（出处能回查原文件）。"""
    merged = "\n\n".join(chunk.markdown for chunk in state.usable_chunks(SIDE_JD))
    source_line = state.source_line(SIDE_JD)
    structured = structure_jd_text(merged, source_label=source_line, title=state.request.jd_title)
    state.jd_markdown = render_jd_markdown(structured, source_line=source_line, generated_at=state.generated_at)

    error = ""
    posting = None
    if merged.strip():                      # 这一路一行都没取到（来源都不可用 / 抓取失败）：没有原文可解析
        try:
            posting = parse_jd_text(merged, source_file=source_line, select=state.request.jd_title)
        except ValueError as exc:           # --jd-name 没匹配到任何岗位
            error = str(exc)
            state.notes.append(error)
    else:
        state.notes.append("JD 侧没有取到任何内容：这一轮只拆了简历，写法建议会先停下来问你")
    state.posting = posting
    if structured.position_count > 1 and not state.request.jd_title:
        state.notes.append(
            f"JD 里识别到 {structured.position_count} 个岗位（{ '、'.join(structured.titles) }）："
            "这一轮用第一个，想换岗位用 --jd-name 指定"
        )

    requirements = len(posting.requirements) if posting else 0
    observation = (
        f"JD 结构化 Tool（{JD_STRUCT_TOOL.slug}）把 {len(state.usable_chunks(SIDE_JD))} 段原文"
        f"（有效 {count_lines(merged)} 行）规整成统一模板：岗位 {structured.position_count} 个 / 条目 "
        f"{structured.bullet_count} 条；要求由能力词典从原文匹配出 {requirements} 条（出处回查原文件）"
    )
    if error:
        observation += f"；岗位没匹配上：{_clip(error)}"
    _record(
        state,
        "StructureJD",
        observation,
        f"state.posting = {requirements} 条要求；state.jd_markdown = "
        f"{count_lines(state.jd_markdown)} 行（统一模板）",
        DECISION_CONTINUE,
        "模板让章节标准、要求仍带原文行号：模型看到的每一条要求都能回查到 JD 原文",
    )
    return state


# ---- 节点：简历侧（事实）------------------------------------------------------

def node_structure_cv(state: PolishAgentState) -> PolishAgentState:
    """简历侧：简历结构化 Tool 把原文拆成章节 + 事实条目（带原文行号）。"""
    merged = "\n\n".join(chunk.markdown for chunk in state.usable_chunks(SIDE_CV))
    source_line = state.source_line(SIDE_CV)
    doc = extract_resume(merged, source_file=source_line, title=state.request.cv_title)
    state.doc = doc
    if len(state.usable_chunks(SIDE_CV)) > 1:
        state.notes.append(
            f"{len(state.usable_chunks(SIDE_CV))} 个简历来源合并后一起拆：条目行号按合并后的文本计"
        )
    roles = "、".join(dict.fromkeys(section.role for section in doc.sections))
    _record(
        state,
        "StructureResume",
        f"简历结构化 Tool（{RESUME_STRUCT_TOOL.slug}）把 {len(state.usable_chunks(SIDE_CV))} 段原文"
        f"（有效 {count_lines(merged)} 行）拆成 {len(doc.sections)} 个章节 / {doc.bullet_count} 条事实，"
        f"另外读到 {len(doc.meta)} 项抬头信息；章节角色：{roles or '—'}",
        f"state.doc = {len(doc.sections)} 个章节 / {doc.bullet_count} 条事实",
        DECISION_CONTINUE,
        "章节怎么切、抬头信息怎么认，只认规则能认出来的：认不出的章节原样保留，一条不丢",
    )
    return state


def node_screen(state: PolishAgentState) -> PolishAgentState:
    """先挡伪装：否定与背景识别 Tool 把「写了但没做过 / 只是背景」的条目挑出来。"""
    report = screen_facts(state.facts)
    state.screen = report
    examples = "；".join(f"「{fact.quote_short}」" for fact in report.blocked[:2])
    observation = (
        f"否定 / 背景识别 Tool（{RESUME_NEGATION_TOOL.slug}）过了一遍 {len(report.facts)} 条事实，"
        f"挡掉 {len(report.blocked)} 条（否定 {len(report.negated)} / 背景 {len(report.background)}）"
    )
    if examples:
        observation += f"：{examples}"
    _record(
        state,
        "ScreenFacts",
        observation,
        f"state.screen = 挡住 {len(report.blocked)} 条；剩下 {len(report.kept)} 条可以判等级",
        DECISION_CONTINUE,
        "顺序不能反：先把「没做过」挑出来，再判等级 —— 否则「没做过 Docker 部署」会被动词骗成「有动作」，"
        "再被模型照抄进写法建议里",
    )
    return state


def node_evidence(state: PolishAgentState) -> PolishAgentState:
    """再判等级：证据等级判定 Tool 给剩下的条目定级。"""
    summary = summarize(state.facts, relaxed=state.relaxed)
    state.summary = summary
    counts = summary.counts
    state.needs_relax = (not state.relaxed) and len(summary.solid) < MIN_SOLID_FACTS
    observation = (
        f"证据等级判定 Tool（{RESUME_EVIDENCE_TOOL.slug}）"
        + ("换放宽口径重判" if state.relaxed else "按严格口径判")
        + f" {len(summary.facts)} 条事实：可当证据 {len(summary.usable)} 条（"
        + " / ".join(f"{EVIDENCE_LABEL[level]} {counts.get(level, 0)}" for level in LEVEL_ORDER)
        + f"），硬证据 {len(summary.solid)} 条"
    )
    if state.needs_relax:
        observation += f"；一条硬证据都没有（门槛 {MIN_SOLID_FACTS} 条），准备换放宽口径"
    _record(
        state,
        "JudgeEvidence",
        observation,
        f"state.summary = 可当证据 {len(summary.usable)} 条 / 硬证据 {len(summary.solid)} 条；"
        f"state.needs_relax = {state.needs_relax}",
        DECISION_CONTINUE,
        "等级决定用词：有结果能写数字、有动作只能写做了什么、仅提及只能写「了解」，模型不许越级",
    )
    return state


def node_relax(state: PolishAgentState) -> PolishAgentState:
    """Adjust：严格口径下一条硬证据都没有，换放宽口径把「仅提及」也算成弱证据，重判一次。"""
    before = len(state.summary.solid) if state.summary else 0
    state.relaxed = True
    state.needs_relax = False
    summary = summarize(state.facts, relaxed=True)
    state.summary = summary
    _record(
        state,
        "RelaxScope",
        f"严格口径下只有 {before} 条硬证据（门槛 {MIN_SOLID_FACTS} 条）：换放宽口径重判，"
        f"现在可当证据 {len(summary.usable)} 条（弱证据 {len(summary.weak)} 条）",
        f"state.relaxed = True；state.summary 换成放宽口径（可当证据 {len(summary.usable)} 条）",
        DECISION_ADJUST,
        "放宽只放宽「证据强弱」：仅提及也能算弱证据，写法上会标注用词要弱；"
        "否定与背景永远不放 —— 「没做过」写多少遍都不会变成「做过」",
    )
    return state


def node_cap_match(state: PolishAgentState) -> PolishAgentState:
    """能力词典匹配 Tool：把每条要求对上已核验事实，并算出缺口与反向证据。"""
    brief = build_brief(
        doc=state.doc,
        summary=state.summary,
        posting=state.posting,
        screen=state.screen,
        jd_markdown=state.jd_markdown,
        source_line=state.source_line(SIDE_JD),
        notes=state.notes,
    )
    state.brief = brief
    risks = brief.risks()
    observation = (
        f"能力词典匹配 Tool（{CAP_MATCH_TOOL.slug}）：{len(brief.requirements)} 条要求里 "
        f"{len(brief.matched)} 条对上了可当证据的事实（覆盖 {brief.coverage:.0%}），"
        f"{len(brief.gaps)} 条没对上证据，{len(brief.reverse)} 条是「简历里写着没做过」的反向证据；"
        f"风险 {len(risks)} 条"
    )
    if risks:
        observation += f"，最该先处理的一条：{_clip(risks[0].text, 90)}"
    _record(
        state,
        "MatchCapabilities",
        observation,
        f"state.brief = 要求 {len(brief.requirements)} 条 / 已核验事实 {len(brief.usable_facts)} 条"
        f"（硬 {len(brief.summary.solid)} / 弱 {len(brief.summary.weak)}）/ 风险 {len(risks)} 条",
        DECISION_CONTINUE,
        "把「模型能看到什么」在这一步钉死：只给规则核验过的事实与出处，缺口与反向证据也一并摆出来，"
        "模型没有机会照着 JD 编经历",
    )
    return state


# ---- 节点：模型（三块内容，各自一步）----------------------------------------

def node_suggest(state: PolishAgentState) -> PolishAgentState:
    """建议写法（llm-suggest）。"""
    return _llm_step(state, SOURCE_LLM_SUGGEST)


def node_interview(state: PolishAgentState) -> PolishAgentState:
    """追问预演（llm-interview）。"""
    return _llm_step(state, SOURCE_LLM_INTERVIEW)


def node_polish(state: PolishAgentState) -> PolishAgentState:
    """润色（llm-polish）。"""
    return _llm_step(state, SOURCE_LLM_POLISH)


def _llm_step(state: PolishAgentState, source: str) -> PolishAgentState:
    """跑一块 LLM 任务：没有 Key 就三块统一标「未启用」，一次请求都不发。"""
    if state.llm is None:
        budget = build_text_budget(state.settings, state.text_client, limit=state.llm_max_calls)
        if budget is None:
            state.llm = unavailable_report(_model(state), "未配置 DEEPSEEK_API_KEY（没有联网）")
        else:
            state.budget = budget
            state.llm = new_report(budget)
    report = state.llm
    if report.block is not None and report.block(source) is not None:
        block = report.block(source)
        _record(
            state,
            _action_of(source),
            task_observation(block, report),
            task_state_update(block, report),
            DECISION_CONTINUE,
            "这一块已经在这次运行里处理过（未启用 / 已达上限），不重复发请求",
        )
        return state
    if state.budget is None:                       # 未启用：没有账本，也就不发请求
        block = report.block(source) or LlmBlock(source=source, title=source, enabled=False)
        _record(
            state,
            _action_of(source),
            task_observation(block, report),
            task_state_update(block, report),
            DECISION_CONTINUE,
            f"{TASK_DUTY.get(source, '')}：没有可用的模型配置，这一块标「未启用」；"
            "简报与其它块照常",
        )
        return state
    block = run_task(report, state.budget, build_user_message(state.brief), source)
    _record(
        state,
        _action_of(source),
        task_observation(block, report),
        task_state_update(block, report),
        DECISION_CONTINUE,
        LLM_REASON,
    )
    return state


# ---- 节点：收尾 -------------------------------------------------------------

def node_finalize(state: PolishAgentState) -> PolishAgentState:
    """写出 Markdown 并收尾（Stop）。"""
    state.stop_reason = _stop_reason(state)     # 先算结论：Markdown 的「跑批说明」要把原因一起写进去
    state.markdown = render_polish_markdown(state)
    state.written = _write_outputs(state)
    target = "、".join(state.written) if state.written else "state.markdown（没有要求落盘）"
    _record(
        state,
        "WriteSuggestions",
        f"写出《写作简报 + 建议写法 + 追问预演 + 润色》：{count_lines(state.markdown)} 行 / "
        f"要求 {state.requirement_count} 条 / 可当证据 {state.usable_count} 条"
        + (f" -> {target}" if state.written else ""),
        f"state.written = {len(state.written)} 个文件；state.stop_reason 已写入",
        DECISION_STOP,
        state.stop_reason,
    )
    return state


def node_ask(state: PolishAgentState) -> PolishAgentState:
    """Ask：简报撑不住，停下来问一个最有价值的问题（一次模型都不调）。"""
    state.question, reason = _ask_question(state)
    state.question_reason = reason
    _record(
        state,
        "AskSource",
        f"JD 侧认到 {state.requirement_count} 条要求，简历侧可当证据 {state.usable_count} 条"
        f"（门槛 {MIN_FACTS} 条）、被挡掉 {state.blocked_count} 条，撑不起一份写法建议",
        "state.question = 一个待回答的问题；没有调模型、没有写出 Markdown",
        DECISION_ASK,
        reason,
    )
    return state


def _model(state: PolishAgentState) -> str:
    return getattr(state.settings, "text_model", "") or "deepseek-chat"


def _vision_model(state: PolishAgentState) -> str:
    model = getattr(state.settings, "vision_model", "") or ""
    return model or DEFAULT_QWEN_VL_MODEL


def _image_prompts(side: str) -> Tuple[str, str]:
    """JD 侧与简历侧用各自的转录提示词（内容不同，纪律一样：逐字转录）。"""
    if side == SIDE_JD:
        return JD_IMAGE_SYSTEM_PROMPT, JD_IMAGE_USER_PROMPT
    return RESUME_IMAGE_SYSTEM_PROMPT, RESUME_IMAGE_USER_PROMPT


def _action_of(source: str) -> str:
    """三块任务在 Trace 里的 Action 名（与 polish_llm.TASK_SPECS 同一份）。"""
    return TASK_ACTION.get(source, "LlmBlock")


def _write_outputs(state: PolishAgentState) -> List[str]:
    if not state.output_dir or not state.markdown.strip():
        return []
    directory = Path(state.output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{state.stem or DEFAULT_STEM}.md"
    target.write_text(state.markdown, encoding="utf-8")
    return [str(target)]


def _ask_question(state: PolishAgentState) -> Tuple[str, str]:
    """只问一个最关键的问题，并写清为什么要问。"""
    missing = [side for side in (SIDE_JD, SIDE_CV) if not state.usable_chunks(side)]
    if missing:
        labels = "、".join(SIDE_LABEL[side] for side in missing)
        return (
            f"{labels} 那边没有取到任何内容。补上它再跑一次 —— "
            "JD 用 --jd-text / --jd-file，简历用 --cv-text / --cv-file，截图用 --jd-image / --cv-image。",
            "两路来源缺一路就没有对照：光有 JD 只能得到要求清单，光有简历只能得到事实清单；"
            "网址可能是前端渲染 / 需要登录，图片可能没配 key，这些只有你能补。",
        )
    if not state.requirement_count:
        return (
            "这份 JD 里没有用能力词典匹配到任何要求：换一份写清「岗位职责 / 任职要求」的 JD，"
            "或者用 --jd-name 指定文件里的某个岗位。",
            "写法建议是冲着「岗位要什么」写的：一条要求都认不出来，就没有可以对齐的目标，"
            "硬写只能变成泛泛而谈。",
        )
    if state.summary is not None and not state.summary.usable:
        blocked = "；".join(f"「{fact.quote_short}」" for fact in state.summary.blocked[:2])
        return (
            f"简历拆出 {len(state.facts)} 条事实，但一条能当证据的都没有 —— 全被否定或背景挡住了"
            f"（比如 {blocked}）。有没有能写成「我做了什么 + 做出什么结果」的经历？",
            "被挡住的条目写的是「没做过」或「只是背景」，按定义不能算「已具备」；"
            "拿它们去写建议，等于教你把没做过的事写进简历。",
        )
    return (
        f"简历里能当证据的事实只有 {state.usable_count} 条（门槛 {MIN_FACTS} 条），"
        "撑不起一版针对这份 JD 的写法建议：还有别的项目 / 实习经历吗？",
        f"能当证据的事实少于 {MIN_FACTS} 条，写法建议只能围着同一两句话打转；"
        "缺的是只有你知道的那部分，再读一遍现有材料不会变多。",
    )


def _stop_reason(state: PolishAgentState) -> str:
    parts = [
        f"JD 侧 {len(state.usable_chunks(SIDE_JD))} 个来源取到 {state.line_count(SIDE_JD)} 行、"
        f"认到 {state.requirement_count} 条要求",
        f"简历侧 {len(state.usable_chunks(SIDE_CV))} 个来源取到 {state.line_count(SIDE_CV)} 行、"
        f"拆出 {len(state.facts)} 条事实（可当证据 {state.usable_count} 条 / 硬证据 {state.solid_count} 条）",
    ]
    if state.brief is not None:
        parts.append(
            f"要求覆盖 {len(state.brief.matched)}/{len(state.brief.core_links)}"
            f"（{state.brief.coverage:.0%}），{len(state.brief.gaps)} 条没对上证据"
        )
    if state.relaxed:
        parts.append("证据改用过放宽口径（仅提及也算弱证据）；否定与背景没有被放宽")
    if state.blocked_count:
        parts.append(f"{state.blocked_count} 条被否定 / 背景挡住，没有送进模型")
    report = state.llm
    if report is not None:
        parts.append(f"模型出了 {report.ok_count}/{len(report.blocks)} 块（累计调用 {report.calls}/{report.call_limit}）")
        degraded = "、".join(block.title for block in report.degraded)
        if degraded:
            parts.append(f"没出的块：{degraded}")
    failed = state.failed_chunks(SIDE_JD) + state.failed_chunks(SIDE_CV)
    if failed:
        parts.append(f"{len(failed)} 个来源没取到（{'、'.join(chunk.ref for chunk in failed)}）")
    parts.append("再读一遍不会产生新内容，停在这里")
    return "；".join(parts) + "。"


# ---- 渲染 -------------------------------------------------------------------

def _render_block(block: Optional[LlmBlock], title: str) -> List[str]:
    """渲染一块 LLM 内容：状态一行 + 正文（表格或整段）。"""
    if block is None:
        return [f"### {title}", "", "（这次运行没有这一块）", ""]
    lines = [f"### {title} {block.label}｜{block.status}", ""]
    if block.error:
        lines += [f"> {block.error}", ""]
    if block.text:
        lines += [block.text, ""]
    if block.items:
        if block.source == SOURCE_LLM_INTERVIEW:
            rows = [
                [index, item.get("question", ""), item.get("target", "") or "—", item.get("prepare", "") or "—"]
                for index, item in enumerate(block.items, start=1)
            ]
            lines += table_lines(("#", "可能被追问", "对着哪条", "怎么准备"), rows)
        else:
            rows = [
                [index, item.get("original", "") or "—", item.get("polished", ""), item.get("ref", "") or "—"]
                for index, item in enumerate(block.items, start=1)
            ]
            lines += table_lines(("#", "原文（已核验）", "润色后", "出处"), rows)
        lines.append("")
    if block.calls:
        lines.append(f"（这一块调用 {block.calls} 次，模型 {block.model}）")
        lines.append("")
    return lines


def render_polish_markdown(state: PolishAgentState) -> str:
    """最终产出：写作简报（规则）+ 建议写法 / 追问预演 / 润色（LLM）。"""
    brief = state.brief
    doc = state.doc
    if brief is None or doc is None:
        return ""
    report = state.llm
    lines = [f"# {doc.name} × {brief.jd_name} · 写作简报与写法建议", ""]
    note = ["由 Polish Agent（LangGraph）生成"]
    note.append(f"JD：{state.source_line(SIDE_JD)}")
    note.append(f"简历：{state.source_line(SIDE_CV)}")
    if state.generated_at:
        note.append(f"生成时间：{state.generated_at}")
    if state.relaxed:
        note.append("证据口径：放宽（仅提及也算弱证据）")
    note.append("规则先跑，模型只做表达：事实、等级与出处一字未改")
    lines += ["> " + "｜".join(note), ""]

    lines += [f"**一句话结论**：{brief.headline()}", ""]
    lines += [
        f"**已核验事实**：{len(brief.usable_facts)} 条"
        f"（硬证据 {len(brief.summary.solid)} / 弱证据 {len(brief.summary.weak)}）"
        f"｜**要求覆盖**：{len(brief.matched)}/{len(brief.core_links)}（{brief.coverage:.0%}）"
        f"｜**被挡掉**：{len(brief.blocked_facts)} 条",
        "",
    ]
    if report is not None:
        lines += [
            f"**模型**：{report.model}｜出了 {report.ok_count}/{len(report.blocks)} 块"
            f"｜累计调用 {report.calls}/{report.call_limit}（超时 {report.timeout:g}s，失败重试 {report.retries} 次）",
            "",
        ]

    lines += ["## 写作简报（规则）", "", "### 要求 ↔ 已核验事实（能力词典匹配）", ""]
    rows = link_rows(brief)
    if rows:
        lines += table_lines(
            ("#", "JD 要求", "层级", "最高证据", "对上几条", "缺口维度", "JD 出处"), rows
        )
    else:
        lines += ["（这份 JD 没有匹配出可证明的要求）"]
    lines.append("")

    lines += ["### 已核验事实（能当证据的）", ""]
    lines += render_evidence_table(brief.usable_facts)
    lines.append("")

    lines += ["### 不能当证据的内容（不许写成「做过」）", ""]
    lines.append(
        "> 否定与背景是不放宽的口径：「没做过」写多少遍都不会变成「做过」，"
        f"它们也不会被送进模型（本次 {len(brief.blocked_facts)} 条）。"
    )
    lines.append("")
    lines += render_blocked_table(brief.blocked_facts)
    lines.append("")

    risks = brief.risks()
    lines += ["### 风险（模型看到的也是这一份）", ""]
    if risks:
        lines += [f"- [{risk.kind}] {risk.text}" + (f"（{ '、'.join(risk.refs) }）" if risk.refs else "") for risk in risks]
    else:
        lines += ["（没有算出来的风险）"]
    lines.append("")

    lines += ["## 建议写法 [LLM]", ""]
    lines += _render_block(report.block(SOURCE_LLM_SUGGEST) if report else None, TASK_DUTY[SOURCE_LLM_SUGGEST])
    lines += ["## 追问预演 [LLM]", ""]
    lines += _render_block(report.block(SOURCE_LLM_INTERVIEW) if report else None, TASK_DUTY[SOURCE_LLM_INTERVIEW])
    lines += ["## 润色 [LLM]", ""]
    lines += _render_block(report.block(SOURCE_LLM_POLISH) if report else None, TASK_DUTY[SOURCE_LLM_POLISH])

    lines += [
        "## 跑批说明",
        "",
        f"- 停在这里的原因：{state.stop_reason}",
        "- 每一段都标了来源：规则写的在「写作简报」里，模型写的都带 [LLM]；",
        "- 模型只读规则给的简报（要求 + 已核验事实 + 缺口与风险），不读 JD / 简历全文。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_polish_console(state: PolishAgentState) -> List[str]:
    """终端输出：先把 Trace 铺满，再给结论（写得出去就说文件，写不出去就提问）。"""
    lines: List[str] = ["", "=" * 72, "运行 Trace（Polish Agent · LangGraph）", "=" * 72]
    for step in state.trace:
        lines += [
            f"[{step.index}] Action      : {step.action}",
            f"    Observation : {step.observation}",
            f"    State Update: {step.state_update}",
            f"    Decision    : {step.decision} —— {step.decision_reason}",
            "",
        ]
    lines.append("=" * 72)
    if state.needs_answer:
        lines += [
            "需要你回答 1 个问题（资料不够，不替你编经历，也没调模型）",
            "=" * 72,
            f"  {state.question}",
            f"  为什么问：{state.question_reason}",
            "",
            '  继续方式：python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md',
        ]
    else:
        lines += ["写完了", "=" * 72]
        if state.brief is not None:
            lines.append(f"  {state.brief.headline()}")
        report = state.llm
        if report is not None:
            lines.append(
                f"  模型：{report.model}｜出了 {report.ok_count}/{len(report.blocks)} 块"
                f"｜累计调用 {report.calls}/{report.call_limit}"
            )
            for block in report.blocks:
                lines.append(f"    - {block.title}：{block.status}｜{block.summary()}")
        lines.append(f"  停在这里的原因：{state.stop_reason}")
    for note in state.notes:
        lines.append(f"  备注：{note}")
    if state.written:
        lines.append("")
        lines.append("已生成：")
        lines += [f"  - {path}" for path in state.written]
    lines.append("")
    return lines


def render_polish_trace(state: PolishAgentState) -> str:
    """把完整 Trace 落成一份 Markdown（--polish-trace）。"""
    lines = [
        "# Polish Agent 运行 Trace",
        "",
        f"- JD 来源：{state.source_line(SIDE_JD)}",
        f"- 简历来源：{state.source_line(SIDE_CV)}",
        f"- 生成时间：{state.generated_at}",
        f"- 最终 Decision：**{state.decision}**",
        "",
        "| # | Action | Observation | State Update | Decision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in state.trace:
        cells = [
            step.index,
            step.action,
            step.observation,
            step.state_update,
            f"{step.decision}（{step.decision_reason}）",
        ]
        lines.append("| " + " | ".join(str(cell).replace("|", "｜") for cell in cells) + " |")
    if state.stop_reason:
        lines += ["", f"**StopReason**：{state.stop_reason}"]
    if state.question:
        lines += ["", f"**待回答**：{state.question}", f"- 为什么问：{state.question_reason}"]
    if state.llm is not None:
        lines += ["", "## 模型调用账本", ""]
        lines += table_lines(
            ("#", "块", "source", "状态", "调用", "内容"),
            [
                [index, block.title, block.source, block.status, block.calls, block.summary()]
                for index, block in enumerate(state.llm.blocks, start=1)
            ],
        )
    return "\n".join(lines) + "\n"


# ---- 路由 -------------------------------------------------------------------

def route_next(state: PolishAgentState) -> str:
    """下一个该走哪个节点：JD 路队头优先，其次简历路，两路都空了 -> structure_jd。"""
    if state.jd_pending:
        return state.jd_pending[0].kind
    if state.cv_pending:
        return state.cv_pending[0].kind
    return "structure_jd"


def route_after_evidence(state: PolishAgentState) -> str:
    """硬证据一条都没有就先 Adjust 一次，否则去做能力词典匹配。"""
    return "relax" if state.needs_relax else "cap_match"


def route_after_brief(state: PolishAgentState) -> str:
    """简报够不够：够就调模型写三块，不够就问一句（一次模型都不调）。"""
    return "ask" if state.thin else "suggest"


@lru_cache(maxsize=1)
def compiled_graph():
    """编译好的 Polish Agent 图（无状态：所有依赖都在 state 里，所以只编译一次）。"""
    graph = StateGraph(PolishAgentState)
    graph.add_node("detect", node_detect)
    graph.add_node("read_text", node_read_text)
    graph.add_node("parse_file", node_parse_file)
    graph.add_node("ocr_image", node_ocr_image)
    graph.add_node("fetch_url", node_fetch_url)
    graph.add_node("structure_jd", node_structure_jd)
    graph.add_node("structure_cv", node_structure_cv)
    graph.add_node("screen", node_screen)
    graph.add_node("evidence", node_evidence)
    graph.add_node("cap_match", node_cap_match)
    graph.add_node("relax", node_relax)
    graph.add_node("suggest", node_suggest)
    graph.add_node("interview", node_interview)
    graph.add_node("polish", node_polish)
    graph.add_node("finalize", node_finalize)
    graph.add_node("ask", node_ask)
    graph.set_entry_point("detect")
    for name in ("detect", "read_text", "parse_file", "ocr_image", "fetch_url"):
        graph.add_conditional_edges(name, route_next, ROUTES)
    graph.add_edge("structure_jd", "structure_cv")
    graph.add_edge("structure_cv", "screen")
    graph.add_edge("screen", "evidence")
    graph.add_conditional_edges("evidence", route_after_evidence, {"relax": "relax", "cap_match": "cap_match"})
    graph.add_edge("relax", "evidence")
    graph.add_conditional_edges("cap_match", route_after_brief, {"suggest": "suggest", "ask": "ask"})
    graph.add_edge("suggest", "interview")
    graph.add_edge("interview", "polish")
    graph.add_edge("polish", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("ask", END)
    return graph.compile()


# ---- 对外入口 ---------------------------------------------------------------

def _rebuild(original: PolishAgentState, result: Dict[str, object]) -> PolishAgentState:
    """把 LangGraph 返回的字段字典还原成状态对象（缺的字段沿用原对象）。"""
    values = {field.name: getattr(original, field.name) for field in dataclasses.fields(PolishAgentState)}
    values.update({key: value for key, value in result.items() if key in values})
    return PolishAgentState(**values)


def run_polish_agent(
    request: PolishRequest,
    settings: Optional[LLMSettings] = None,
    text_client: Optional[OpenAICompatClient] = None,
    vision_client: Optional[OpenAICompatClient] = None,
    fetcher: Optional[Callable] = None,
    jd_sources: Optional[Sequence[JDSource]] = None,
    cv_sources: Optional[Sequence[JDSource]] = None,
    out_dir=None,
    stem: str = DEFAULT_STEM,
    llm_max_calls: int = DEFAULT_CALL_LIMIT,
    generated_at: str = "",
) -> PolishAgentState:
    """跑一次完整的 Polish Agent Loop，返回带着 Trace 的状态。

    out_dir 给了就落一份 Markdown；text_client / vision_client / fetcher 可以注入替身（测试或二次开发）；
    jd_sources / cv_sources 给了就用它（调用方已经识别过一次输入类型，避免重复识别）。
    不给网址就一个 HTTP 都不发，不给图片就一次多模态调用都不调；没有文本层 Key 就三块 LLM 全部「未启用」。
    """
    vision_ready = vision_client is not None or bool(getattr(settings, "vision_ready", False))
    if jd_sources is None:
        jd_sources = detect_sources(
            text=request.jd_text,
            files=request.jd_files,
            images=request.jd_images,
            urls=request.jd_urls,
            vision_ready=vision_ready,
        )
    if cv_sources is None:
        cv_sources = detect_sources(
            text=request.cv_text,
            files=request.cv_files,
            images=request.cv_images,
            urls=request.cv_urls,
            vision_ready=vision_ready,
        )
    state = PolishAgentState(
        request=request,
        jd_sources=list(jd_sources),
        cv_sources=list(cv_sources),
        output_dir=str(out_dir) if out_dir else "",
        stem=stem or DEFAULT_STEM,
        generated_at=generated_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
        settings=settings,
        text_client=text_client,
        vision_client=vision_client,
        fetcher=fetcher,
        llm_max_calls=max(1, int(llm_max_calls or DEFAULT_CALL_LIMIT)),
    )
    return _rebuild(state, compiled_graph().invoke(state))
