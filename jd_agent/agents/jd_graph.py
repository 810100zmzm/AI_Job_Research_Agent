"""JD Agent（LangGraph）：把 JD（文本 / 图片 / 网址）转成 Markdown。

图长这样（节点名就是 Trace 里的 Action，箭头上写着走法）：

    detect ─┬─ 文本 ─► read_text  ─┐
            ├─ 文件 ─► parse_file ┤
            ├─ 图片 ─► ocr_image  ┤──►（队列里还有来源就继续，没有了）──► structure ─┬─► finalize
            └─ 网址 ─► fetch_url  ─┘                                             └─► ask
                          │
                          └─ 严格口径留下的正文太少 ──► relax（换口径重抽，不再发请求）

每走一步都往 state.trace 里落一条，四要素缺一不可：
    Action（这一步做了什么）-> Observation（真实看到了什么，带来源与行数）
      -> State Update（状态被改成了什么）-> Decision（下一步只能 Continue / Adjust / Ask / Stop）

三条纪律：
  * 三种输入类型各走各的分支：文本直读、文件走工具区的文件解析 Tool、图片走 Qwen-VL、网址抓一次 HTML；
  * 「来源取不到」只是降级，不是崩溃：记一条 note 继续跑剩下的来源，最后如实写进 StopReason；
  * 内容太少就 Ask（只问一个最有价值的问题），不替用户编一份 JD 出来。

依赖（settings / vision_client / fetcher）都放在 state 里，所以编译好的图本身是无状态的：
同一份输入进来，出去的是同一份 Markdown 与同一条 Trace。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from langgraph.graph import END, StateGraph

from ..domain.jd_html import clean_lines, count_lines, html_to_markdown
from ..services.jd_source import (
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
from ..core.llm import DEFAULT_QWEN_VL_MODEL, OpenAICompatClient
from ..core.schema import (
    DECISION_ADJUST,
    DECISION_ASK,
    DECISION_CONTINUE,
    DECISION_STOP,
    DECISIONS,
    TraceStep,
)
from ..core.settings import LLMSettings
from ..tools.jd_files import JD_FILE_TOOL, KIND_LABEL, parse_file_markdown
from ..tools.jd_struct import (
    JD_STRUCT_TOOL,
    StructuredJD,
    render_jd_markdown,
    structure_jd_text,
)

# 判定口径（写死在这里，方便直接检查，也方便日后调）
MIN_JD_BULLETS = 3      # 转出来的要点少于这个数：内容太少，Ask 一次
MIN_URL_LINES = 5       # 严格口径下有效行少于这个数：换放宽口径重抽一次
DEFAULT_STEM = "jd"

NODE_NAMES = (
    "detect",       # 识别文本 / 文件 / 图片 / 网址
    "read_text",    # 直接读取文本
    "parse_file",   # 解析文件内容
    "ocr_image",    # 用视觉模型转录图片
    "fetch_url",    # 抓取网址正文
    "relax",        # 放宽口径重抽网址
    "structure",    # 结构化 JD
    "finalize",     # 生成并写出 Markdown
    "ask",          # 内容不足时提问
)

# 条件边的路由表：来源类型 -> 该走哪个节点（"structure" 表示来源都取完了）
ROUTES = {
    SOURCE_TEXT: "read_text",
    SOURCE_FILE: "parse_file",
    SOURCE_IMAGE: "ocr_image",
    SOURCE_URL: "fetch_url",
    "structure": "structure",
}


def _clip(text: object, limit: int = 120) -> str:
    """压成一行并截断：Trace 里的报错信息不该把一行撑爆。"""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ---- 状态 -------------------------------------------------------------------

@dataclass
class JDRequest:
    """用户给的输入：三种来源（文本 / 图片 / 网址）+ 两个可选说明。"""

    text: str = ""
    files: Tuple[str, ...] = ()
    images: Tuple[str, ...] = ()
    urls: Tuple[str, ...] = ()
    title: str = ""

    @property
    def empty(self) -> bool:
        return not (self.text.strip() or self.files or self.images or self.urls)


@dataclass
class JDChunk:
    """一个来源取到的 Markdown 原文（note 非空表示这个来源没取到）。"""

    kind: str
    ref: str
    label: str
    markdown: str = ""
    key: str = ""              # 原始 key（网址 / 路径），放宽口径重抽时用来回查
    note: str = ""
    raw_lines: int = 0         # 清理前的行数
    dropped: int = 0           # 清理时丢掉的行数
    weak: bool = False         # 严格口径下留下的正文太少，等放宽口径重抽

    @property
    def line_count(self) -> int:
        return count_lines(self.markdown)

    @property
    def empty(self) -> bool:
        return not self.markdown.strip()


@dataclass
class JDAgentState:
    """JD Agent 的完整状态，也是唯一的数据出口（Markdown / 终端 / Trace 都由它渲染）。"""

    request: JDRequest
    sources: List[JDSource] = field(default_factory=list)
    pending: List[JDSource] = field(default_factory=list)
    chunks: List[JDChunk] = field(default_factory=list)
    pages: Dict[str, str] = field(default_factory=dict)      # 网址 -> 已抓到的 Markdown（未清理）
    trace: List[TraceStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    structured: Optional[StructuredJD] = None
    markdown: str = ""
    question: str = ""
    question_reason: str = ""
    stop_reason: str = ""
    relaxed: bool = False          # 网址是否已经用过放宽口径
    needs_relax: bool = False      # 刚抓完的页面需要放宽口径重抽
    output_dir: str = ""           # 空串 = 不落盘（只把 Markdown 留在 state 里）
    stem: str = DEFAULT_STEM
    generated_at: str = ""
    written: List[str] = field(default_factory=list)
    settings: Optional[LLMSettings] = None
    vision_client: Optional[OpenAICompatClient] = None
    fetcher: Optional[Callable] = None     # 注入的抓取替身（测试用）

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
    def thin(self) -> bool:
        """转出来的要点太少：撑不住下游的岗位判断。"""
        return self.structured is None or self.structured.bullet_count < MIN_JD_BULLETS

    @property
    def usable_chunks(self) -> List[JDChunk]:
        return [chunk for chunk in self.chunks if not chunk.empty]

    @property
    def failed_chunks(self) -> List[JDChunk]:
        return [chunk for chunk in self.chunks if chunk.note]

    @property
    def line_count(self) -> int:
        return sum(chunk.line_count for chunk in self.usable_chunks)


def _record(
    state: JDAgentState, action: str, observation: str, state_update: str, decision: str, reason: str
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


# ---- 节点：识别与取证 -------------------------------------------------------

def _take(state: JDAgentState, kind: str) -> Optional[JDSource]:
    """取出下一个待处理的来源（route_next 已经保证队头就是这一类）。"""
    for index, source in enumerate(state.pending):
        if source.kind == kind:
            return state.pending.pop(index)
    return None


def node_detect(state: JDAgentState) -> JDAgentState:
    """识别输入类型：哪些来源可用、每个走哪条分支。"""
    usable = [source for source in state.sources if source.usable]
    skipped = [source for source in state.sources if not source.usable]
    state.pending = list(usable)
    state.notes += unusable_notes(skipped)
    observation = (
        f"收到 {len(state.sources)} 个来源（{describe_sources(state.sources) or '无'}）："
        f"可用 {len(usable)} 个"
    )
    if skipped:
        observation += f"，跳过 {len(skipped)} 个（{'；'.join(_clip(source.note, 60) for source in skipped)}）"
    _record(
        state,
        "DetectSource",
        observation,
        f"state.pending = {len(usable)} 个（{'、'.join(source.kind_label for source in usable) or '空'}）",
        DECISION_CONTINUE,
        "来源类型已识别：文本直读、文件走解析 Tool、图片走 Qwen-VL、网址抓一次 HTML",
    )
    return state


def node_read_text(state: JDAgentState) -> JDAgentState:
    """文本分支：直接读取（不联网、不调模型）。"""
    source = _take(state, SOURCE_TEXT)
    if source is None:
        return state
    markdown, dropped = clean_lines(source.raw, strict=False)
    chunk = JDChunk(
        kind=SOURCE_TEXT,
        ref=source.ref,
        label=source.label,
        markdown=markdown,
        raw_lines=len(source.raw.splitlines()),
        dropped=dropped,
    )
    state.chunks.append(chunk)
    _record(
        state,
        "ReadText",
        f"直接读取{source.label}：原文 {chunk.raw_lines} 行 -> 有效 {chunk.line_count} 行"
        f"（合掉 {dropped} 行空行 / 重复行），没调模型、没联网",
        f"state.chunks += 文本 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "文本是最可靠的一手来源：原样读取就够，不需要模型",
    )
    return state


def node_parse_file(state: JDAgentState) -> JDAgentState:
    """文件分支：交给工具区的文件解析 Tool（纯规则、不联网）。"""
    source = _take(state, SOURCE_FILE)
    if source is None:
        return state
    try:
        parsed = parse_file_markdown(source.raw)
    except ValueError as exc:
        state.chunks.append(JDChunk(SOURCE_FILE, source.ref, source.label, note=str(exc)))
        state.notes.append(f"{source.ref}：{exc}")
        _record(
            state,
            "ParseFile",
            f"文件解析失败：{_clip(exc)}",
            "state.chunks += 文件 1 段（空）；state.notes += 1",
            DECISION_CONTINUE,
            "这个文件读不出来，但队列里还有别的来源：先跑完，最后如实写进 StopReason",
        )
        return state
    chunk = JDChunk(
        kind=SOURCE_FILE,
        ref=source.ref,
        label=source.label,
        markdown=parsed.markdown,
        key=source.raw,
        dropped=parsed.dropped,
    )
    state.chunks.append(chunk)
    _record(
        state,
        "ParseFile",
        f"文件解析 Tool（{JD_FILE_TOOL.slug}）读 {source.ref}：{KIND_LABEL.get(parsed.kind, parsed.kind)}"
        f" -> 有效 {chunk.line_count} 行（丢掉 {parsed.dropped} 行）",
        f"state.chunks += 文件 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "文件解析复用工具区的纯规则实现：不联网、不调模型，命令行与前端同一份口径",
    )
    return state


def node_ocr_image(state: JDAgentState) -> JDAgentState:
    """图片分支：Qwen-VL 逐字转成 Markdown（联网）。"""
    source = _take(state, SOURCE_IMAGE)
    if source is None:
        return state
    index = 1 + len([chunk for chunk in state.chunks if chunk.kind == SOURCE_IMAGE])
    model = _vision_model(state)
    try:
        markdown = image_to_markdown(source.raw, state.vision_client, model, ref=source.ref, index=index)
    except SourceError as exc:
        state.chunks.append(JDChunk(SOURCE_IMAGE, source.ref, source.label, note=str(exc)))
        state.notes.append(f"{source.ref}：{exc}")
        _record(
            state,
            "OcrImage",
            f"图片转录失败：{_clip(exc)}",
            "state.chunks += 图片 1 段（空）；state.notes += 1",
            DECISION_CONTINUE,
            "一张图读不出来不影响其它来源：继续跑，最后在 StopReason 里说明这张图没算进去",
        )
        return state
    markdown, dropped = clean_lines(markdown, strict=False)
    chunk = JDChunk(
        kind=SOURCE_IMAGE,
        ref=source.ref,
        label=source.label,
        markdown=markdown,
        key=source.raw,
        dropped=dropped,
    )
    state.chunks.append(chunk)
    _record(
        state,
        "OcrImage",
        f"Qwen-VL（{model}）把 {source.label} 逐字转成 Markdown：{chunk.line_count} 行；"
        f"数字与指标请对照原图复核",
        f"state.chunks += 图片 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "图片内容只有模型能读，但转写只做逐字转录：不翻译、不总结、不补内容",
    )
    return state


def node_fetch_url(state: JDAgentState) -> JDAgentState:
    """网址分支：抓一次 HTML -> Markdown（严格口径，噪声多就交给 relax）。"""
    source = _take(state, SOURCE_URL)
    if source is None:
        return state
    strict = not state.relaxed
    cached = state.pages.get(source.raw)
    try:
        if cached is None:
            html, final_url = fetch_html(source.raw, opener=state.fetcher)
            cached = html_to_markdown(html)
            state.pages[source.raw] = cached
            fetched = f"抓取 1 次 HTTP（{final_url}）"
        else:
            fetched = "复用已抓到的内容（不再发请求）"
        markdown, dropped = clean_lines(cached, strict=strict)
    except SourceError as exc:
        state.chunks.append(
            JDChunk(SOURCE_URL, source.ref, source.label, key=source.raw, note=str(exc))
        )
        state.notes.append(f"{source.ref}：{exc}")
        _record(
            state,
            "FetchUrl",
            f"抓取失败：{_clip(exc)}",
            "state.chunks += 网址 1 段（空）；state.notes += 1",
            DECISION_CONTINUE,
            "页面抓不到不是输入错：先跑完别的来源，最后如实写进 StopReason，并提示改给文本或截图",
        )
        return state

    chunk = JDChunk(
        kind=SOURCE_URL,
        ref=source.ref,
        label=source.label,
        markdown=markdown,
        key=source.raw,
        raw_lines=count_lines(cached),
        dropped=dropped,
    )
    state.chunks.append(chunk)
    weak = strict and chunk.line_count < MIN_URL_LINES
    chunk.weak = weak
    state.needs_relax = weak
    observation = (
        f"{fetched}：HTML 转 Markdown 后有效 {chunk.line_count} 行（丢掉 {dropped} 行页面噪音）"
    )
    if weak:
        observation += f"；有效行不足 {MIN_URL_LINES} 行，准备换放宽口径"
    _record(
        state,
        "FetchUrl",
        observation,
        f"state.chunks += 网址 1 段（{chunk.line_count} 行）；state.needs_relax = {weak}",
        DECISION_CONTINUE,
        "严格口径先只留像 JD 的正文：噪声少、召回低，剩下太少再放宽（下一步 Adjust）",
    )
    return state


def node_relax(state: JDAgentState) -> JDAgentState:
    """Adjust：严格口径留下的正文太少，换放宽口径从同一份内容里重抽（不再发请求）。"""
    changed: List[str] = []
    for chunk in state.chunks:
        if chunk.kind != SOURCE_URL or not chunk.weak:
            continue
        before = chunk.line_count
        chunk.markdown, chunk.dropped = clean_lines(state.pages.get(chunk.key, ""), strict=False)
        chunk.weak = False
        changed.append(f"{chunk.label} {before} -> {chunk.line_count} 行")
    state.relaxed = True
    state.needs_relax = False
    _record(
        state,
        "RelaxFetch",
        f"换放宽口径重抽 {len(changed)} 个页面：{'、'.join(changed) or '无可重抽的页面'}"
        f"（用的是已抓到的内容，没再发请求）",
        f"state.relaxed = True；{len(changed)} 段网址正文被替换",
        DECISION_ADJUST,
        "严格口径只留下零星几行：换放宽口径先把页面文字都留下，宁可多留噪声，也不漏掉 JD 正文",
    )
    return state


# ---- 节点：结构化与收尾 -----------------------------------------------------

def node_structure(state: JDAgentState) -> JDAgentState:
    """把取到的原文交给 JD 结构化 Tool：切岗位 + 分章节 + 出模板。"""
    merged = "\n\n".join(chunk.markdown for chunk in state.usable_chunks)
    source_line = _source_line(state)
    structured = structure_jd_text(merged, source_label=source_line, title=state.request.title)
    state.structured = structured
    state.markdown = render_jd_markdown(
        structured, source_line=source_line, generated_at=state.generated_at
    )
    _record(
        state,
        "StructureJD",
        f"JD 结构化 Tool（{JD_STRUCT_TOOL.slug}）把 {len(state.usable_chunks)} 段原文"
        f"（有效 {count_lines(merged)} 行）规整成 {structured.position_count} 个岗位 / "
        f"{structured.bullet_count} 条要点；命中能力 {len(structured.capabilities)} 项",
        f"state.structured = {structured.position_count} 个岗位 / {structured.bullet_count} 条要点；"
        f"state.markdown = {count_lines(state.markdown)} 行",
        DECISION_CONTINUE,
        "章节与岗位只认规则能认出来的：认不出的章节原样保留，不往「任职要求」里塞",
    )
    return state


def node_ask(state: JDAgentState) -> JDAgentState:
    """Ask：能读的来源都太薄，停下来问一个最有价值的问题。"""
    state.question, reason = _ask_question(state)
    state.question_reason = reason
    bullets = state.structured.bullet_count if state.structured else 0
    _record(
        state,
        "AskSource",
        f"取到的内容只够 {bullets} 条要点（门槛 {MIN_JD_BULLETS} 条）、{state.line_count} 行有效正文，"
        f"转不出可信的 JD",
        "state.question = 一个待回答的问题；没有写出 Markdown",
        DECISION_ASK,
        reason,
    )
    return state


def node_finalize(state: JDAgentState) -> JDAgentState:
    """写出 Markdown 并收尾（Stop）。"""
    state.written = _write_outputs(state)
    state.stop_reason = _stop_reason(state)
    structured = state.structured
    positions = structured.position_count if structured else 0
    bullets = structured.bullet_count if structured else 0
    target = "、".join(state.written) if state.written else "state.markdown（没有要求落盘）"
    _record(
        state,
        "WriteMarkdown",
        f"写出 JD Markdown：{count_lines(state.markdown)} 行 / {positions} 个岗位 / {bullets} 条要点"
        + (f" -> {target}" if state.written else ""),
        f"state.written = {len(state.written)} 个文件；state.stop_reason 已写入",
        DECISION_STOP,
        state.stop_reason,
    )
    return state


def _vision_model(state: JDAgentState) -> str:
    model = getattr(state.settings, "vision_model", "") or ""
    return model or DEFAULT_QWEN_VL_MODEL


def _source_line(state: JDAgentState) -> str:
    labels = [source.label for source in state.sources if source.usable]
    return "、".join(dict.fromkeys(labels)) or "（没有可用来源）"


def _write_outputs(state: JDAgentState) -> List[str]:
    if not state.output_dir or not state.markdown.strip():
        return []
    directory = Path(state.output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{state.stem or DEFAULT_STEM}.md"
    target.write_text(state.markdown, encoding="utf-8")
    return [str(target)]


def _ask_question(state: JDAgentState) -> Tuple[str, str]:
    """只问一个最关键的问题，并写清为什么要问。"""
    if not state.usable_chunks:
        return (
            "没有取到任何 JD 内容。把 JD 文字粘过来（--jd-text），或给一张截图（--jd-image），"
            "我按你给的这份重转一次。",
            "能读的来源都是空的：网址可能是前端渲染 / 需要登录，图片可能没配 key 或读不出来 —— "
            "这些只有你能补上。",
        )
    return (
        f"只取到 {state.line_count} 行、{state.structured.bullet_count if state.structured else 0} 条要点，"
        "看不出完整的岗位要求。这份 JD 还有别的页面或截图吗？",
        f"要点少于 {MIN_JD_BULLETS} 条，转出来的 JD 撑不住下游的岗位判断；"
        "再读一遍现有来源不会产生新内容，缺的是只有你知道的那部分。",
    )


def _stop_reason(state: JDAgentState) -> str:
    structured = state.structured
    parts = [
        f"{len(state.usable_chunks)} 个来源取到 {state.line_count} 行有效正文，"
        f"转成 {structured.position_count if structured else 0} 个岗位 / "
        f"{structured.bullet_count if structured else 0} 条要点"
    ]
    if state.relaxed:
        parts.append("网址改用过放宽口径")
    failed = state.failed_chunks
    if failed:
        parts.append(
            f"{len(failed)} 个来源没取到（{'、'.join(chunk.ref for chunk in failed)}）"
        )
    parts.append("再读一遍不会产生新内容，停在这里")
    return "；".join(parts) + "。"


# ---- 路由 -------------------------------------------------------------------

def route_next(state: JDAgentState) -> str:
    """下一个该走哪个节点：队头来源的类型，或者来源都取完了 -> structure。"""
    return state.pending[0].kind if state.pending else "structure"


def route_after_url(state: JDAgentState) -> str:
    """刚抓完的页面需要放宽口径就先 Adjust，否则按正常顺序往下走。"""
    return "relax" if state.needs_relax else route_next(state)


def route_after_structure(state: JDAgentState) -> str:
    """转出来的东西够不够：够就写文件（Stop），不够就问一句（Ask）。"""
    return "ask" if state.thin else "finalize"


@lru_cache(maxsize=1)
def compiled_graph():
    """编译好的 JD Agent 图（无状态：所有依赖都在 state 里，所以只编译一次）。"""
    graph = StateGraph(JDAgentState)
    graph.add_node("detect", node_detect)
    graph.add_node("read_text", node_read_text)
    graph.add_node("parse_file", node_parse_file)
    graph.add_node("ocr_image", node_ocr_image)
    graph.add_node("fetch_url", node_fetch_url)
    graph.add_node("relax", node_relax)
    graph.add_node("structure", node_structure)
    graph.add_node("finalize", node_finalize)
    graph.add_node("ask", node_ask)
    graph.set_entry_point("detect")
    for name in ("detect", "read_text", "parse_file", "ocr_image"):
        graph.add_conditional_edges(name, route_next, ROUTES)
    graph.add_conditional_edges("fetch_url", route_after_url, {**ROUTES, "relax": "relax"})
    graph.add_conditional_edges("relax", route_next, ROUTES)
    graph.add_conditional_edges("structure", route_after_structure, {"finalize": "finalize", "ask": "ask"})
    graph.add_edge("finalize", END)
    graph.add_edge("ask", END)
    return graph.compile()


# ---- 对外入口 ---------------------------------------------------------------

def _rebuild(original: JDAgentState, result: Dict[str, object]) -> JDAgentState:
    """把 LangGraph 返回的字段字典还原成状态对象（缺的字段沿用原对象）。"""
    values = {field.name: getattr(original, field.name) for field in dataclasses.fields(JDAgentState)}
    values.update({key: value for key, value in result.items() if key in values})
    return JDAgentState(**values)


def run_jd_agent(
    request: JDRequest,
    settings: Optional[LLMSettings] = None,
    vision_client: Optional[OpenAICompatClient] = None,
    fetcher: Optional[Callable] = None,
    sources: Optional[Sequence[JDSource]] = None,
    out_dir=None,
    stem: str = DEFAULT_STEM,
    generated_at: str = "",
) -> JDAgentState:
    """跑一次完整的 JD Agent Loop，返回带着 Trace 的状态。

    out_dir 给了就落一份 Markdown；vision_client / fetcher 可以注入替身（测试或二次开发）。
    sources 给了就用它（调用方已经识别过一次输入类型，避免重复识别）；
    不给网址就一个 HTTP 都不发，不给图片就一次模型都不调。
    """
    vision_ready = vision_client is not None or bool(getattr(settings, "vision_ready", False))
    if sources is None:
        sources = detect_sources(
            text=request.text,
            files=request.files,
            images=request.images,
            urls=request.urls,
            vision_ready=vision_ready,
        )
    state = JDAgentState(
        request=request,
        sources=list(sources),
        output_dir=str(out_dir) if out_dir else "",
        stem=stem or DEFAULT_STEM,
        generated_at=generated_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
        settings=settings,
        vision_client=vision_client,
        fetcher=fetcher,
    )
    return _rebuild(state, compiled_graph().invoke(state))


def render_jd_console(state: JDAgentState) -> List[str]:
    """终端输出：先把 Trace 铺满，再给结论（写得出去就给文件，写不出去就提问）。"""
    lines: List[str] = ["", "=" * 72, "运行 Trace（JD Agent · LangGraph）", "=" * 72]
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
            "需要你回答 1 个问题（内容太少，不替你编）",
            "=" * 72,
            f"  {state.question}",
            f"  为什么问：{state.question_reason}",
            "",
            '  继续方式：python main.py --jd-agent --jd-text "你的 JD 文本"',
        ]
    else:
        lines += ["转换完成", "=" * 72]
        lines.append(f"  {state.markdown.splitlines()[0] if state.markdown else '（没有内容）'}")
        lines.append(f"  停在这里的原因：{state.stop_reason}")
    for note in state.notes:
        lines.append(f"  备注：{note}")
    if state.written:
        lines.append("")
        lines.append("已生成：")
        lines += [f"  - {path}" for path in state.written]
    lines.append("")
    return lines


def render_jd_trace(state: JDAgentState) -> str:
    """把完整 Trace 落成一份 Markdown（--jd-trace）。"""
    lines = [
        "# JD Agent 运行 Trace",
        "",
        f"- 来源：{_source_line(state)}",
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
    return "\n".join(lines) + "\n"
