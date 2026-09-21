"""Streamlit 前端：把 Agent 的判断结果与简历排版结果渲染成网页。

启动（项目根目录）：
    .venv\\Scripts\\python.exe -m streamlit run streamlit_app.py

三条与主流程一致的设计约束：
  * 渲染复用 jd_agent.render —— 网页里的 Markdown / JSON / HTML 与命令行产物同源，前端不重写判定逻辑；
  * 大模型与图片解析都是显式开关，默认关闭；开启会联网，失败只降级成一块「调用失败」的说明；
  * 前端不往磁盘写任何东西：下载走内存，报告要落盘请用命令行（python main.py）。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jd_agent import tools  # noqa: E402
from jd_agent.agent import MAX_ROUNDS, build_text_budget, run_agent  # noqa: E402
from jd_agent.cli import INPUT_DIR, JD_DIR, PROJECT_DIR, RESUME_DIR, report_stem  # noqa: E402
from jd_agent.jd import load_jd, load_jd_positions  # noqa: E402
from jd_agent.llm import DEFAULT_CALL_LIMIT  # noqa: E402
from jd_agent.render import render_html, render_json, render_markdown  # noqa: E402
from jd_agent.schema import (  # noqa: E402
    VERDICT_EDIT,
    VERDICT_NO,
    VERDICT_YES,
    AgentState,
    JobPosting,
)
from jd_agent.settings import DEFAULT_ENV_FILE, LLMSettings, load_env, resolve_settings  # noqa: E402

st.set_page_config(page_title="AI 求职尽调 Agent", page_icon=":material/analytics:", layout="wide")

VERDICT_BADGE = {VERDICT_YES: "green", VERDICT_EDIT: "orange", VERDICT_NO: "red"}


# ---- 只读的数据加载（缓存，不写盘） -----------------------------------------


def md_files(directory: Path) -> List[str]:
    return [str(item) for item in sorted(directory.glob("*.md"))] if directory.is_dir() else []


def _label(path_text: str) -> str:
    """下拉框里显示相对项目根的短路径。"""
    path = Path(path_text)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def input_candidates() -> List[str]:
    """第二个输入（项目描述 / 简历）能选的 md：input/project → input/profile → input 根目录。"""
    ordered: List[str] = []
    for group in (md_files(PROJECT_DIR), md_files(RESUME_DIR), md_files(INPUT_DIR)):
        ordered += [item for item in group if item not in ordered]
    return ordered


@st.cache_data(show_spinner=False)
def postings_of(path_text: str) -> List[JobPosting]:
    return load_jd_positions(Path(path_text))


@st.cache_data(show_spinner=False)
def first_posting_of(path_text: str) -> JobPosting:
    return load_jd(Path(path_text))


@st.cache_data(show_spinner=False)
def env_settings() -> LLMSettings:
    """读 .env 并解析配置；只在打开大模型 / 图片解析时才调用。"""
    loaded = load_env(DEFAULT_ENV_FILE)
    return resolve_settings(
        env_file=loaded.path,
        env_keys=tuple(loaded.applied) + tuple(loaded.skipped),
        text_model="",
        vision_model="",
    )


@st.cache_data(show_spinner=False)
def build_resume_cached(resume_file: str, merges: Tuple[str, ...], style: str, stamp: str) -> tools.Resume:
    """stamp（文件 mtime）只用来让缓存失效；换风格要重排，所以 style 也进缓存 key。"""
    return tools.build_resume(Path(resume_file), [Path(item) for item in merges], style=style)


def _mtime(*paths: str) -> str:
    marks = []
    for item in paths:
        try:
            marks.append(f"{Path(item).stat().st_mtime_ns}")
        except OSError:
            marks.append("0")
    return "|".join(marks)


def _refs(refs: Sequence[str]) -> str:
    return "、".join(f"`{ref}`" for ref in refs) if refs else "—"


# ---- 侧边栏：输入与开关 ------------------------------------------------------


def pick_inputs() -> Tuple[List[Tuple[Path, JobPosting]], str, dict]:
    """返回 (待分析的岗位列表, 项目描述路径, 开关字典)。"""
    jd_file_list = md_files(JD_DIR)
    candidates = input_candidates()
    options = {"llm": False, "vision": False, "llm_max_calls": DEFAULT_CALL_LIMIT, "max_rounds": MAX_ROUNDS}

    with st.sidebar:
        st.markdown("#### 输入")
        mode = st.radio(
            "JD 来源",
            ("input/jd 下的全部岗位", "指定 JD 文件"),
            help="默认把 input/jd 下每个文件的每个岗位都跑一遍，一个岗位一份报告。",
        )
        selected: List[Tuple[Path, JobPosting]] = []
        if mode == "指定 JD 文件":
            if not jd_file_list:
                st.error(f"{_label(str(JD_DIR))} 下没有 .md 文件")
            else:
                chosen_file = st.selectbox("JD 文件", jd_file_list, format_func=_label)
                postings = postings_of(chosen_file)
                titles = [f"{index + 1}. {item.title}" for index, item in enumerate(postings)]
                choice = st.selectbox("岗位", ["全部岗位"] + titles)
                picked = postings if choice == "全部岗位" else [postings[titles.index(choice)]]
                selected = [(Path(chosen_file), item) for item in picked]
        else:
            for item in jd_file_list:
                selected += [(Path(item), posting) for posting in postings_of(item)]

        project = ""
        if candidates:
            project = st.selectbox(
                "项目描述 / 简历 md（第二个输入）",
                candidates,
                format_func=_label,
                help="命令行里对应 --project：可以指向 input/profile 下你自己的简历 md。",
            )
        else:
            st.error(f"{_label(str(INPUT_DIR))} 下没有 .md 文件")

        st.markdown("#### 能力开关")
        options["llm"] = st.toggle(
            "启用大模型（DeepSeek）",
            value=False,
            help="报告定稿后才介入：建议写法草稿 + 报告润色。会联网，失败只降级成一块说明。",
        )
        options["llm_max_calls"] = int(
            st.number_input(
                "LLM 调用上限（本次运行共用）",
                min_value=1,
                max_value=50,
                value=DEFAULT_CALL_LIMIT,
                step=1,
                disabled=not options["llm"],
            )
        )
        options["vision"] = st.toggle(
            "启用图片解析（Qwen-VL）", value=False, help="把项目描述里引用的本地图片读成文字事实，会联网。"
        )
        options["max_rounds"] = int(
            st.number_input("最多检索轮数", min_value=1, max_value=5, value=MAX_ROUNDS, step=1)
        )
        options["run"] = st.button("开始分析", type="primary", icon=":material/play_arrow:", width="stretch")

    return selected, project, options


# ---- 跑一遍 Agent（只读，不写盘） --------------------------------------------


def run_one(
    jd_path: Path,
    posting: JobPosting,
    project: str,
    options: dict,
    answer: str = "",
    budget=None,
) -> AgentState:
    """跑一个岗位；answer 非空时先把回答并入证据池，再重新检索与判定。"""
    return run_agent(
        jd_path,
        Path(project),
        answer=answer,
        max_rounds=options["max_rounds"],
        llm=options["llm"],
        vision=options["vision"],
        llm_max_calls=options["llm_max_calls"],
        settings=env_settings() if (options["llm"] or options["vision"]) else None,
        posting=posting,
        llm_budget=budget,
    )


def run_selected(
    selected: Sequence[Tuple[Path, JobPosting]], project: str, options: dict
) -> List[Tuple[JobPosting, AgentState, str]]:
    settings = env_settings() if (options["llm"] or options["vision"]) else None
    budget = None
    if options["llm"]:
        # 多个岗位共用一份调用账本：一次运行的调用总数仍然 ≤ 上限
        budget = build_text_budget(settings, None, limit=options["llm_max_calls"])
        if budget is None:
            st.warning(
                "没有读到 DEEPSEEK_API_KEY：LLM 两块会显示「未启用」，规则报告照跑。",
                icon=":material/key_off:",
            )
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(selected)
    progress = st.progress(0.0, text="正在跑规则流程……")
    results: List[Tuple[JobPosting, AgentState, str]] = []
    for index, (jd_path, posting) in enumerate(selected, start=1):
        state = run_one(jd_path, posting, project, options, budget=budget)
        results.append((posting, state, stamp))
        progress.progress(index / total, text=f"已分析 {index}/{total}：{posting.title}")
    progress.empty()
    return results


def apply_answer(index: int, answer: str) -> None:
    """把页面上写的回答补进去，重跑第 index 个岗位，然后整体重绘。

    和命令行的 `--answer` 完全等价：回答会作为新材料并入证据池，再走一遍检索与判定。
    """
    analysis = st.session_state.get("analysis")
    if not analysis:
        return
    jd_path = analysis["selected"][index - 1][0]
    posting = analysis["results"][index - 1][0]
    state = run_one(jd_path, posting, analysis["project"], analysis["options"], answer=answer)
    analysis["results"][index - 1] = (posting, state, analysis["stamp"])
    st.session_state["analysis"] = analysis
    st.rerun()


# ---- 结果渲染（全部复用 render.py 的产物） -----------------------------------


def _render_verdict(state: AgentState) -> None:
    verdict = state.verdict
    columns = st.columns([2, 1, 1, 1])
    with columns[0]:
        st.badge(verdict.call, icon=":material/gavel:", color=VERDICT_BADGE.get(verdict.call, "blue"))
        st.markdown(verdict.headline)
    columns[1].metric("核心覆盖", f"{len(state.solid)}/{len(state.core_requirements)}")
    columns[2].metric("结果级证据", state.result_match_count)
    columns[3].metric("缺口", len(state.missing))

    st.markdown(f"**为什么（{len(verdict.reasons)} 条 · [规则]）**")
    for order, reason in enumerate(verdict.reasons, start=1):
        st.markdown(f"{order}. {reason.text}")
        st.caption(f"JD {reason.jd_ref}　｜　项目 {_refs(reason.project_refs)}")
    st.markdown(f"**风险（{len(verdict.risks)} 条 · [规则]）**")
    for risk in verdict.risks:
        st.markdown(f"- 【{risk.kind}】{risk.text}")
        st.caption(f"出处：{_refs(risk.refs)}")
    if verdict.rewrite:
        st.markdown("**建议写法 · [规则]**")
        st.write(verdict.rewrite)
    st.caption(f"StopReason：{verdict.stop_reason}")


def _render_llm(state: AgentState) -> None:
    report = state.llm_report
    if report is None or report.is_empty:
        return
    st.markdown("**LLM 生成内容 · [LLM]**")
    st.caption(
        f"模型 {report.model or '未指定'}　｜　调用 {report.calls}/{report.call_limit} 次　｜　"
        f"单次超时 {report.timeout:g}s　｜　重试 {report.retries} 次"
        + (f"　｜　{report.note}" if report.note else "")
    )
    st.caption("这些内容由模型生成，不是判定依据；结论与 Decision 仍来自规则段落。")
    for block in report.blocks:
        with st.expander(
            f"[LLM] {block.title}（source: {block.source}｜状态：{block.status}）", expanded=block.ok
        ):
            if block.text:
                st.write(block.text)
            for order, item in enumerate(block.items, start=1):
                st.markdown(f"{order}. {item.get('question', '')}")
                if item.get("target"):
                    st.caption(f"追问点：{item['target']}")
                if item.get("prepare"):
                    st.caption(f"怎么准备：{item['prepare']}")
            for note in block.notes:
                st.caption(note)
            if block.error:
                st.caption(f"说明：{block.error}")


def _render_evidence(state: AgentState) -> None:
    with st.expander(f"岗位要求 vs 项目证据（{len(state.matches)} 项 · [规则]）", icon=":material/table_chart:"):
        rows = [
            {
                "要求": match.requirement_name,
                "JD 层级": match.requirement.level_label,
                "证据等级": match.level_label,
                "说明": match.note,
                "JD 出处": match.requirement.ref,
                "项目出处": "、".join(fact.ref for fact in match.facts) or "—",
            }
            for match in state.matches
        ]
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_trace(state: AgentState) -> None:
    with st.expander(f"运行 Trace（{len(state.trace)} 步 · 四要素）", icon=":material/route:"):
        for step in state.trace:
            st.markdown(
                f"**[{step.index}] {step.action}**\n\n"
                f"- Observation：{step.observation}\n"
                f"- State Update：{step.state_update}\n"
                f"- Decision：`{step.decision}` —— {step.decision_reason}"
            )


def _render_answer_form(index: int) -> None:
    """Ask 分支：在页面上直接把答案写进来，效果等同命令行的 --answer。"""
    answer = st.text_area(
        "你的回答（写清做了什么、结果是多少，有数字就给数字）",
        key=f"answer-{index}",
        height=110,
        placeholder="例：上线 30 天内 120 人使用，问答命中率从 60% 提到 85%（人工抽检 100 条）。",
    )
    st.caption(
        "提交后这段材料会并入证据池，重新走一遍检索与判定（等同一份新报告，LLM 预算重新计算）；"
        "命令行等价写法：python main.py --answer \"…\""
    )
    if st.button(
        "提交回答并重新判断", type="primary", icon=":material/send:", key=f"answer-submit-{index}"
    ):
        if not answer.strip():
            st.warning("先写点内容再提交。", icon=":material/warning:")
        else:
            with st.spinner("把回答并入证据池后重新检索……"):
                apply_answer(index, answer)


def render_result(posting: JobPosting, state: AgentState, stamp: str, index: int, total: int) -> None:
    title = posting.title if total == 1 else f"[{index}/{total}] {posting.title}"
    st.subheader(title, icon=":material/checklist:")
    st.caption(
        f"JD `{posting.source_file}`　｜　项目 `{state.project.source_file}`　｜　"
        f"生成时间 {stamp}　｜　最终 Decision：**{state.decision}**"
    )
    with st.container(border=True):
        if state.needs_answer:
            st.markdown("**资料不足，需要你先回答一个问题**（不猜；这一轮一次模型都没调）")
            st.markdown(f"> {state.question}")
            st.caption(f"为什么问：{state.question_reason}")
            _render_answer_form(index)
        else:
            _render_verdict(state)
            _render_llm(state)
    _render_evidence(state)
    _render_trace(state)

    stem = report_stem(posting, total)
    buttons = st.columns(3)
    for column, (label, name, text, mime) in zip(
        buttons,
        (
            ("下载 Markdown", f"{stem}.md", render_markdown(state, stamp), "text/markdown"),
            ("下载 HTML", f"{stem}.html", render_html(state, stamp), "text/html"),
            ("下载 JSON", f"{stem}.json", render_json(state, stamp), "application/json"),
        ),
    ):
        column.download_button(
            label, text, file_name=name, mime=mime, icon=":material/download:", key=f"dl-{name}"
        )


# ---- 简历排版 ---------------------------------------------------------------


def render_resume_tab() -> None:
    st.subheader("简历排版", icon=":material/description:")
    st.caption("纯规则：章节按模板重排、列表与表格统一、空白规整；不新增任何事实，也不联网。")
    resumes = md_files(RESUME_DIR)
    if not resumes:
        st.info(f"{_label(str(RESUME_DIR))} 下还没有简历 md。", icon=":material/info:")
        return

    left, right = st.columns([2, 1])
    with left:
        resume_file = st.selectbox("简历 md", resumes, format_func=_label, key="resume-file")
    with right:
        merges = st.multiselect(
            "并入「项目经历」的项目描述（可多选）",
            md_files(PROJECT_DIR),
            format_func=_label,
            key="resume-merges",
        )

    style_key = st.radio(
        "排版风格",
        [style.key for style in tools.resume_styles.all_styles()],
        format_func=lambda key: tools.resume_styles.get_style(key).label,
        horizontal=True,
        key="resume-style",
    )
    st.caption(tools.resume_styles.get_style(style_key).summary)

    resume = build_resume_cached(resume_file, tuple(merges), style_key, _mtime(resume_file, *merges))
    st.caption("　｜　".join(f"{key}：{value}" for key, value in tools.resume_summary(resume)))
    if resume.notes:
        st.caption("排版规整：" + "；".join(resume.notes))

    stem = tools.default_stem(style_key)
    markdown_text = tools.render_markdown(resume)
    html_text = tools.render_html(resume)
    buttons = st.columns(2)
    buttons[0].download_button(
        "下载 Markdown",
        markdown_text,
        file_name=f"{stem}.md",
        mime="text/markdown",
        icon=":material/download:",
        key="resume-dl-md",
    )
    buttons[1].download_button(
        "下载 HTML",
        html_text,
        file_name=f"{stem}.html",
        mime="text/html",
        icon=":material/download:",
        key="resume-dl-html",
    )
    preview, source = st.tabs([f"排版预览（{len(resume.sections)} 节）", "Markdown 源码"])
    with preview:
        # 预览用 standalone=False：样式都限定在 .resume-doc 里，不会影响 Streamlit 自己的界面
        st.html(tools.render_html(resume, standalone=False), width="stretch")
    with source:
        st.code(markdown_text, language="markdown")


def main() -> None:
    st.title("AI 求职尽调 Agent", icon=":material/analytics:")
    st.caption("输入一份 JD + 一个项目描述：这个项目值不值得写进简历。规则优先、来源可区分、失败可降级。")

    selected, project, options = pick_inputs()
    if options["run"]:
        if not selected or not project:
            st.error("先选好 JD 与项目描述 md。", icon=":material/error:")
        else:
            results = run_selected(selected, project, options)
            # 存下这次运行的全部输入，Ask 分支要在页面上重跑（等价于命令行的 --answer）
            st.session_state["analysis"] = {
                "results": results,
                "selected": list(selected),
                "project": project,
                "options": options,
                "stamp": results[0][2] if results else datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

    analysis, resume = st.tabs(["分析结果", "简历排版"])
    with analysis:
        results = (st.session_state.get("analysis") or {}).get("results") or []
        if not results:
            st.info(
                "左侧选好输入后点「开始分析」。默认纯规则、不联网，判据与命令行完全一致。",
                icon=":material/info:",
            )
        for index, (posting, state, stamp) in enumerate(results, start=1):
            render_result(posting, state, stamp, index, len(results))
    with resume:
        render_resume_tab()


main()
