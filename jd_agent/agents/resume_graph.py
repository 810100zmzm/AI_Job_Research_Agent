"""Resume Agent（LangGraph）：把简历拆成事实条目，逐条判证据等级。

图长这样（节点名就是 Trace 里的 Action，箭头上写着走法）：

    detect ─┬─ 文本 ─► read_text  ─┐
            ├─ 文件 ─► parse_file ┤
            ├─ 图片 ─► ocr_image  ┤──►（队列里还有来源就继续，没有了）──► structure ─► screen ─► evidence ─┬─► finalize
            └─ 网址 ─► fetch_url  ─┘                                                                        └─► ask
                                                                    严格口径下一条硬证据都没有 ─► relax ─► evidence

每走一步都往 state.trace 里落一条，四要素缺一不可：
    Action（这一步做了什么）-> Observation（真实看到了什么，带来源与行数）
      -> State Update（状态被改成了什么）-> Decision（下一步只能 Continue / Adjust / Ask / Stop）

判断顺序是这一层的重点 —— **先挡伪装，再判等级**：

    structure（简历结构化 Tool）  原文 -> 章节 + 事实条目（带原文行号）
    screen（否定 / 背景识别 Tool） 先挑出「写了但没做过 / 只是背景」的条目，不许它们当证据
    evidence（证据等级判定 Tool） 剩下的条目才判 有结果 / 有动作 / 仅提及

顺序反了就会把「没做过 Docker 部署」判成「有动作」。否定与背景**放宽口径也不放**。

另外两条纪律：
  * 「来源取不到」只是降级，不是崩溃：记一条 note 继续跑剩下的来源，最后如实写进 StopReason；
  * 内容太少、或者一条能当证据的都没有，就 Ask（只问一个最有价值的问题），不替用户编经历。

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

from ..domain.jd import display_path
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
from ..domain.resume_facts import (
    EVIDENCE_LABEL,
    FACT_BLOCK_LABEL,
    EvidenceSummary,
    ResumeDoc,
    ResumeFact,
    extract_resume,
    summarize,
    table_lines,
)
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
from ..tools.resume_evidence import (
    LEVEL_ORDER,
    RESUME_EVIDENCE_TOOL,
    render_blocked_table,
    render_evidence_table,
)
from ..tools.resume_negation import RESUME_NEGATION_TOOL, ScreenReport, screen_facts
from ..tools.resume_struct import RESUME_STRUCT_TOOL

# 判定口径（写死在这里，方便直接检查，也方便日后调）
MIN_FACTS = 3           # 可当证据的事实条目少于这个数：内容太少，Ask 一次
MIN_SOLID_FACTS = 1     # 严格口径下硬证据少于这个数（一条都没有）：换放宽口径重判一次
DEFAULT_STEM = "resume_facts"

NODE_NAMES = (
    "detect",       # 识别简历来源
    "read_text",    # 直接读取简历文本
    "parse_file",   # 解析简历文件
    "ocr_image",    # 用视觉模型转录简历图片
    "fetch_url",    # 抓取简历网页
    "structure",    # 结构化简历章节与事实
    "screen",       # 拦截否定 / 背景条目
    "evidence",     # 判定证据等级
    "relax",        # 放宽证据口径后重判
    "finalize",     # 生成并写出事实清单
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

# 简历截图只有模型能读：提示词与 JD 那份的区别只在「转写什么」上，纪律一样（逐字、不补内容）
RESUME_IMAGE_SYSTEM_PROMPT = (
    "你是严谨的简历转录助手。把图片里的简历逐字转写成 Markdown：\n"
    "1. 原文有章节标题（例如「教育背景」「项目经历」「专业技能」「奖项荣誉」）就按原层级写成 ## / ###；\n"
    "2. 每条经历 / 技能单独一行，写成 `- ` 开头的列表；\n"
    "3. 表格转成 Markdown 表格，保留表头；\n"
    "4. 只转录图里能看清的字，不翻译、不总结、不补内容、不改顺序；看不清的字写 [看不清]；\n"
    "5. 只输出 Markdown 正文，不要解释，不要代码块围栏。"
)
RESUME_IMAGE_USER_PROMPT = "这是简历的第 {index} 张图片（来源：{ref}）。请把图中的内容逐字转写成 Markdown。"


def _clip(text: object, limit: int = 120) -> str:
    """压成一行并截断：Trace 里的报错信息不该把一行撑爆。"""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ---- 状态 -------------------------------------------------------------------

@dataclass
class ResumeRequest:
    """用户给的输入：四种来源（文本 / 文件 / 图片 / 网址）+ 一个可选的简历标题。"""

    text: str = ""
    files: Tuple[str, ...] = ()
    images: Tuple[str, ...] = ()
    urls: Tuple[str, ...] = ()
    title: str = ""

    @property
    def empty(self) -> bool:
        return not (self.text.strip() or self.files or self.images or self.urls)


@dataclass
class ResumeChunk:
    """一个来源取到的 Markdown 原文（note 非空表示这个来源没取到）。"""

    kind: str
    ref: str
    label: str
    markdown: str = ""
    key: str = ""              # 原始 key（路径 / 网址），出错时用来回查
    note: str = ""
    raw_lines: int = 0         # 清理前的行数
    dropped: int = 0           # 清理时丢掉的行数

    @property
    def line_count(self) -> int:
        return count_lines(self.markdown)

    @property
    def empty(self) -> bool:
        return not self.markdown.strip()


@dataclass
class ResumeAgentState:
    """Resume Agent 的完整状态，也是唯一的数据出口（Markdown / 终端 / Trace 都由它渲染）。"""

    request: ResumeRequest
    sources: List[JDSource] = field(default_factory=list)
    pending: List[JDSource] = field(default_factory=list)
    chunks: List[ResumeChunk] = field(default_factory=list)
    pages: Dict[str, str] = field(default_factory=dict)      # 网址 -> 已抓到的 Markdown
    trace: List[TraceStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    doc: Optional[ResumeDoc] = None
    screen: Optional[ScreenReport] = None
    summary: Optional[EvidenceSummary] = None
    markdown: str = ""
    question: str = ""
    question_reason: str = ""
    stop_reason: str = ""
    relaxed: bool = False          # 是否已经换过放宽口径重判证据
    needs_relax: bool = False      # 刚判完，严格口径下一条硬证据都没有
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
    def facts(self) -> List[ResumeFact]:
        return list(self.doc.facts) if self.doc else []

    @property
    def usable_chunks(self) -> List[ResumeChunk]:
        return [chunk for chunk in self.chunks if not chunk.empty]

    @property
    def failed_chunks(self) -> List[ResumeChunk]:
        return [chunk for chunk in self.chunks if chunk.note]

    @property
    def line_count(self) -> int:
        return sum(chunk.line_count for chunk in self.usable_chunks)

    @property
    def evidence_count(self) -> int:
        return len(self.summary.evidence) if self.summary else 0

    @property
    def usable_count(self) -> int:
        """能拿来当证据的条目数（硬 + 弱，不含被挡掉的）。"""
        return len(self.summary.usable) if self.summary else 0

    @property
    def solid_count(self) -> int:
        return len(self.summary.solid) if self.summary else 0

    @property
    def blocked_count(self) -> int:
        return len(self.summary.blocked) if self.summary else 0

    @property
    def thin(self) -> bool:
        """可当证据的事实太少：撑不住下游的岗位判断。

        口径是「能当证据的条目数」（usable，不含被挡掉的），不是严格口径下的硬证据数：
        一份全是「了解 / 熟悉」的简历仍然有 3 条以上可当弱证据的条目，不该被当成内容太少；
        「硬证据一条都没有」是另一道门（MIN_SOLID_FACTS，先 Adjust 一次）。
        """
        return self.usable_count < MIN_FACTS


def _record(
    state: ResumeAgentState, action: str, observation: str, state_update: str, decision: str, reason: str
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

def _take(state: ResumeAgentState, kind: str) -> Optional[JDSource]:
    """取出下一个待处理的来源（route_next 已经保证队头就是这一类）。"""
    for index, source in enumerate(state.pending):
        if source.kind == kind:
            return state.pending.pop(index)
    return None


def node_detect(state: ResumeAgentState) -> ResumeAgentState:
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


def node_read_text(state: ResumeAgentState) -> ResumeAgentState:
    """文本分支：直接读取（不联网、不调模型）。"""
    source = _take(state, SOURCE_TEXT)
    if source is None:
        return state
    markdown, dropped = clean_lines(source.raw, strict=False)
    chunk = ResumeChunk(
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


def node_parse_file(state: ResumeAgentState) -> ResumeAgentState:
    """文件分支：交给工具区的文件解析 Tool（纯规则、不联网）。"""
    source = _take(state, SOURCE_FILE)
    if source is None:
        return state
    try:
        parsed = parse_file_markdown(source.raw)
    except ValueError as exc:
        state.chunks.append(ResumeChunk(SOURCE_FILE, source.ref, source.label, note=str(exc)))
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
    chunk = ResumeChunk(
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


def node_ocr_image(state: ResumeAgentState) -> ResumeAgentState:
    """图片分支：Qwen-VL 逐字转成 Markdown（联网）。"""
    source = _take(state, SOURCE_IMAGE)
    if source is None:
        return state
    index = 1 + len([chunk for chunk in state.chunks if chunk.kind == SOURCE_IMAGE])
    model = _vision_model(state)
    try:
        markdown = image_to_markdown(
            source.raw,
            state.vision_client,
            model,
            ref=source.ref,
            index=index,
            system_prompt=RESUME_IMAGE_SYSTEM_PROMPT,
            user_prompt=RESUME_IMAGE_USER_PROMPT,
        )
    except SourceError as exc:
        state.chunks.append(ResumeChunk(SOURCE_IMAGE, source.ref, source.label, note=str(exc)))
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
    chunk = ResumeChunk(
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


def node_fetch_url(state: ResumeAgentState) -> ResumeAgentState:
    """网址分支：抓一次 HTML -> Markdown（简历页不套 JD 那套 UI 噪声词表）。"""
    source = _take(state, SOURCE_URL)
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
        state.chunks.append(
            ResumeChunk(SOURCE_URL, source.ref, source.label, key=source.raw, note=str(exc))
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

    chunk = ResumeChunk(
        kind=SOURCE_URL,
        ref=source.ref,
        label=source.label,
        markdown=markdown,
        key=source.raw,
        raw_lines=count_lines(cached),
        dropped=dropped,
    )
    state.chunks.append(chunk)
    _record(
        state,
        "FetchUrl",
        f"{fetched}：HTML 转 Markdown 后有效 {chunk.line_count} 行（丢掉 {dropped} 行空行 / 重复行）",
        f"state.chunks += 网址 1 段（{chunk.line_count} 行）",
        DECISION_CONTINUE,
        "简历页面不是招聘网站，不套 JD 那套 UI 噪声词表：页面上的文字尽量留下，宁可多留也不漏",
    )
    return state


# ---- 节点：拆事实、挡伪装、判等级 -------------------------------------------

def node_structure(state: ResumeAgentState) -> ResumeAgentState:
    """把取到的原文交给简历结构化 Tool：切章节 + 认抬头 + 抽事实条目。"""
    merged = "\n\n".join(chunk.markdown for chunk in state.usable_chunks)
    source_file = _fact_source(state)
    doc = extract_resume(merged, source_file=source_file, title=state.request.title)
    state.doc = doc
    roles = "、".join(dict.fromkeys(section.role for section in doc.sections))
    if len(state.usable_chunks) > 1:
        state.notes.append(
            f"{len(state.usable_chunks)} 个来源合并后一起拆：条目行号按合并后的文本计"
        )
    _record(
        state,
        "StructureResume",
        f"简历结构化 Tool（{RESUME_STRUCT_TOOL.slug}）把 {len(state.usable_chunks)} 段原文"
        f"（有效 {count_lines(merged)} 行）拆成 {len(doc.sections)} 个章节 / {doc.bullet_count} 条事实，"
        f"另外读到 {len(doc.meta)} 项抬头信息；章节角色：{roles or '—'}",
        f"state.doc = {len(doc.sections)} 个章节 / {doc.bullet_count} 条事实",
        DECISION_CONTINUE,
        "章节怎么切、抬头信息怎么认，只认规则能认出来的：认不出的章节原样保留，一条不丢",
    )
    return state


def node_screen(state: ResumeAgentState) -> ResumeAgentState:
    """先挡伪装：否定与背景识别 Tool 把「写了但没做过 / 只是背景」的条目挑出来。"""
    report = screen_facts(state.facts)
    state.screen = report
    examples = "；".join(f"「{fact.quote_short}」" for fact in report.blocked[:2])
    observation = (
        f"否定 / 背景识别 Tool（{RESUME_NEGATION_TOOL.slug}）过了一遍 {len(report.facts)} 条事实，"
        f"挡掉 {len(report.blocked)} 条（{FACT_BLOCK_LABEL['negated']} {len(report.negated)} / "
        f"{FACT_BLOCK_LABEL['background']} {len(report.background)}）"
    )
    if examples:
        observation += f"：{examples}"
    _record(
        state,
        "ScreenFacts",
        observation,
        f"state.screen = 挡住 {len(report.blocked)} 条；剩下 {len(report.kept)} 条可以判等级",
        DECISION_CONTINUE,
        "顺序不能反：先把「没做过」挑出来，再判等级 —— 否则「没做过 Docker 部署」会被动词骗成「有动作」",
    )
    return state


def node_evidence(state: ResumeAgentState) -> ResumeAgentState:
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
        f"state.summary = 可当证据 {len(summary.usable)} 条 / 硬证据 {len(summary.solid)} 条"
        f"；state.needs_relax = {state.needs_relax}",
        DECISION_CONTINUE,
        "严格口径只认「有动作 / 有结果」：挑得准、但可能把「了解 / 熟悉」级的东西全漏掉，所以留一次 Adjust",
    )
    return state


def node_relax(state: ResumeAgentState) -> ResumeAgentState:
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
        "放宽只放宽「证据强弱」：仅提及也能算弱证据；否定与背景永远不放 —— 「没做过」写多少遍都不会变成「做过」",
    )
    return state


# ---- 节点：收尾 -------------------------------------------------------------

def node_ask(state: ResumeAgentState) -> ResumeAgentState:
    """Ask：拆出来的东西撑不住，停下来问一个最有价值的问题。"""
    state.question, reason = _ask_question(state)
    state.question_reason = reason
    _record(
        state,
        "AskSource",
        f"拆出 {len(state.facts)} 条事实，其中可当证据 {state.evidence_count} 条"
        f"（门槛 {MIN_FACTS} 条）、被挡掉 {state.blocked_count} 条，撑不起一份可信的证据清单",
        "state.question = 一个待回答的问题；没有写出 Markdown",
        DECISION_ASK,
        reason,
    )
    return state


def node_finalize(state: ResumeAgentState) -> ResumeAgentState:
    """写出 Markdown 并收尾（Stop）。"""
    state.markdown = render_resume_markdown(
        state.doc,
        state.summary,
        source_line=_source_line(state),
        generated_at=state.generated_at,
    )
    state.written = _write_outputs(state)
    state.stop_reason = _stop_reason(state)
    target = "、".join(state.written) if state.written else "state.markdown（没有要求落盘）"
    _record(
        state,
        "WriteFacts",
        f"写出简历事实清单：{count_lines(state.markdown)} 行 / {len(state.facts)} 条事实 / "
        f"可当证据 {state.usable_count} 条（硬证据 {state.solid_count} 条）"
        + (f" -> {target}" if state.written else ""),
        f"state.written = {len(state.written)} 个文件；state.stop_reason 已写入",
        DECISION_STOP,
        state.stop_reason,
    )
    return state


def _vision_model(state: ResumeAgentState) -> str:
    model = getattr(state.settings, "vision_model", "") or ""
    return model or DEFAULT_QWEN_VL_MODEL


def _source_line(state: ResumeAgentState) -> str:
    labels = [source.label for source in state.sources if source.usable]
    return "、".join(dict.fromkeys(labels)) or "（没有可用来源）"


def _fact_source(state: ResumeAgentState) -> str:
    """事实条目的「出处」怎么写：只有一个文件来源时就用文件路径，行号就是文件里的行号。"""
    chunks = state.usable_chunks
    if len(chunks) == 1 and chunks[0].kind == SOURCE_FILE:
        return display_path(chunks[0].key) or chunks[0].label
    return _source_line(state)


def _write_outputs(state: ResumeAgentState) -> List[str]:
    if not state.output_dir or not state.markdown.strip():
        return []
    directory = Path(state.output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{state.stem or DEFAULT_STEM}.md"
    target.write_text(state.markdown, encoding="utf-8")
    return [str(target)]


def _ask_question(state: ResumeAgentState) -> Tuple[str, str]:
    """只问一个最关键的问题，并写清为什么要问。"""
    if not state.usable_chunks:
        return (
            "没有取到任何简历内容。把简历文字粘过来（--cv-text）、给一份文件（--cv-file），"
            "或给一张截图（--cv-image），我按你给的这份重拆一次。",
            "能读的来源都是空的：网址可能是前端渲染 / 需要登录，图片可能没配 key 或读不出来 —— "
            "这些只有你能补上。",
        )
    if state.summary is not None and not state.summary.usable:
        blocked = "；".join(f"「{fact.quote_short}」" for fact in state.summary.blocked[:2])
        return (
            f"这份简历拆出 {len(state.facts)} 条事实，但一条能当证据的都没有 —— 全被否定或背景挡住了"
            f"（比如 {blocked}）。有没有能写成「我做了什么 + 做出什么结果」的经历？",
            "被挡住的条目写的是「没做过」或「只是背景」，按定义不能算「已具备」；"
            "补一句真实做过的动作与结果，就能变成硬证据。",
        )
    return (
        f"按{'放宽' if state.relaxed else '严格'}口径算下来，只有 {state.evidence_count} 条事实能当证据"
        f"（门槛 {MIN_FACTS} 条），看不出这份简历的完整经历。还有别的版本、别的项目描述吗？",
        f"能当证据的事实少于 {MIN_FACTS} 条，拆出来的清单撑不住下游的岗位判断；"
        "再拆一遍现有来源不会产生新内容，缺的是只有你知道的那部分。",
    )


def _stop_reason(state: ResumeAgentState) -> str:
    summary = state.summary
    parts = [
        f"{len(state.usable_chunks)} 个来源取到 {state.line_count} 行有效正文，"
        f"拆成 {len(state.doc.sections) if state.doc else 0} 个章节 / {len(state.facts)} 条事实，"
        f"其中可当证据 {state.usable_count} 条（硬证据 {state.solid_count} 条）"
    ]
    if state.relaxed:
        parts.append("证据改用过放宽口径")
    if state.blocked_count:
        parts.append(
            f"{state.blocked_count} 条被否定 / 背景挡住"
            "（「没做过」与「只是背景」都不算已具备）"
        )
    failed = state.failed_chunks
    if failed:
        parts.append(f"{len(failed)} 个来源没取到（{'、'.join(chunk.ref for chunk in failed)}）")
    parts.append("再读一遍不会产生新内容，停在这里")
    return "；".join(parts) + "。"


# ---- 渲染 -------------------------------------------------------------------

def render_resume_markdown(
    doc: Optional[ResumeDoc],
    summary: Optional[EvidenceSummary],
    source_line: str = "",
    generated_at: str = "",
) -> str:
    """最终的简历事实清单：抬头信息 + 一句话结论 + 按章节的等级表 + 被挡掉的条目。"""
    if doc is None or summary is None:
        return ""
    counts = summary.counts
    lines = [f"# {doc.name} · 简历事实清单", ""]
    note = ["由 Resume Agent（LangGraph）生成"]
    if source_line:
        note.append(f"来源：{source_line}")
    if generated_at:
        note.append(f"生成时间：{generated_at}")
    note.append("事实与等级由规则判定，一字未改")
    lines += ["> " + "｜".join(note), ""]

    for key, value in doc.meta[:6]:
        lines.append(f"**{key}**：{value}")
    if doc.meta:
        lines.append("")
    lines += [
        f"**可当证据的事实**：{len(summary.usable)} 条"
        f"（硬证据 {len(summary.solid)} / 弱证据 {len(summary.weak)}）"
        f"｜**被挡掉**：{len(summary.blocked)} 条"
        f"（{FACT_BLOCK_LABEL['negated']} {len(summary.negated)} / "
        f"{FACT_BLOCK_LABEL['background']} {len(summary.background)}）",
        "",
        f"**一句话结论**：{summary.conclusion}",
        "",
        "## 事实清单",
        "",
    ]

    for section in doc.sections:
        if not section.facts:
            continue
        lines.append(f"### {section.title}")
        lines.append("")
        if section.background:
            lines.append("> 整节是背景 / 简介：这里的条目都不算「我做过什么」。")
            lines.append("")
        for subsection in dict.fromkeys(fact.subsection for fact in section.facts if fact.subsection):
            group = [fact for fact in section.facts if fact.subsection == subsection and fact.usable]
            if not group:
                continue
            lines.append(f"#### {subsection}")
            lines.append("")
            lines += render_evidence_table(group)
            lines.append("")
        rest = [fact for fact in section.facts if not fact.subsection and fact.usable]
        if rest:
            lines += render_evidence_table(rest)
            lines.append("")

    lines += ["## 不能当证据的内容（挡住「写了但没做过」）", ""]
    lines.append(
        f"> 否定与背景是两个**不放宽**的口径：「没做过」写多少遍都不会变成「做过」，"
        f"它们永远不算证据（本次 {len(summary.blocked)} 条）。"
    )
    lines.append("")
    lines += render_blocked_table(summary.blocked)
    return "\n".join(lines).rstrip() + "\n"


def render_resume_console(state: ResumeAgentState) -> List[str]:
    """终端输出：先把 Trace 铺满，再给结论（写得出去就给文件，写不出去就提问）。"""
    lines: List[str] = ["", "=" * 72, "运行 Trace（Resume Agent · LangGraph）", "=" * 72]
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
            "需要你回答 1 个问题（内容太少 / 全被挡住，不替你编经历）",
            "=" * 72,
            f"  {state.question}",
            f"  为什么问：{state.question_reason}",
            "",
            '  继续方式：python main.py --resume-agent --cv-text "你的简历文本"',
        ]
    else:
        lines += ["拆完了", "=" * 72]
        if state.summary is not None:
            lines.append(f"  {state.summary.conclusion}")
        lines.append(f"  停在这里的原因：{state.stop_reason}")
    for note in state.notes:
        lines.append(f"  备注：{note}")
    if state.written:
        lines.append("")
        lines.append("已生成：")
        lines += [f"  - {path}" for path in state.written]
    lines.append("")
    return lines


def render_resume_trace(state: ResumeAgentState) -> str:
    """把完整 Trace 落成一份 Markdown（--resume-trace）。"""
    lines = [
        "# Resume Agent 运行 Trace",
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
    if state.summary is not None:
        lines += ["", "## 事实条目（按判定结果）", ""]
        lines += render_evidence_table(state.summary.usable)
        lines += ["", "## 被挡住的内容", "", *render_blocked_table(state.summary.blocked)]
    return "\n".join(lines) + "\n"


# ---- 路由 -------------------------------------------------------------------

def route_next(state: ResumeAgentState) -> str:
    """下一个该走哪个节点：队头来源的类型，或者来源都取完了 -> structure。"""
    return state.pending[0].kind if state.pending else "structure"


def route_after_evidence(state: ResumeAgentState) -> str:
    """硬证据一条都没有就先 Adjust 一次，否则够不够：够就写文件（Stop），不够就问一句（Ask）。"""
    if state.needs_relax:
        return "relax"
    return "ask" if state.thin else "finalize"


@lru_cache(maxsize=1)
def compiled_graph():
    """编译好的 Resume Agent 图（无状态：所有依赖都在 state 里，所以只编译一次）。"""
    graph = StateGraph(ResumeAgentState)
    graph.add_node("detect", node_detect)
    graph.add_node("read_text", node_read_text)
    graph.add_node("parse_file", node_parse_file)
    graph.add_node("ocr_image", node_ocr_image)
    graph.add_node("fetch_url", node_fetch_url)
    graph.add_node("structure", node_structure)
    graph.add_node("screen", node_screen)
    graph.add_node("evidence", node_evidence)
    graph.add_node("relax", node_relax)
    graph.add_node("finalize", node_finalize)
    graph.add_node("ask", node_ask)
    graph.set_entry_point("detect")
    for name in ("detect", "read_text", "parse_file", "ocr_image", "fetch_url"):
        graph.add_conditional_edges(name, route_next, ROUTES)
    graph.add_edge("structure", "screen")
    graph.add_edge("screen", "evidence")
    graph.add_conditional_edges(
        "evidence", route_after_evidence, {"relax": "relax", "finalize": "finalize", "ask": "ask"}
    )
    graph.add_edge("relax", "evidence")
    graph.add_edge("finalize", END)
    graph.add_edge("ask", END)
    return graph.compile()


# ---- 对外入口 ---------------------------------------------------------------

def _rebuild(original: ResumeAgentState, result: Dict[str, object]) -> ResumeAgentState:
    """把 LangGraph 返回的字段字典还原成状态对象（缺的字段沿用原对象）。"""
    values = {field.name: getattr(original, field.name) for field in dataclasses.fields(ResumeAgentState)}
    values.update({key: value for key, value in result.items() if key in values})
    return ResumeAgentState(**values)


def run_resume_agent(
    request: ResumeRequest,
    settings: Optional[LLMSettings] = None,
    vision_client: Optional[OpenAICompatClient] = None,
    fetcher: Optional[Callable] = None,
    sources: Optional[Sequence[JDSource]] = None,
    out_dir=None,
    stem: str = DEFAULT_STEM,
    generated_at: str = "",
) -> ResumeAgentState:
    """跑一次完整的 Resume Agent Loop，返回带着 Trace 的状态。

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
    state = ResumeAgentState(
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
