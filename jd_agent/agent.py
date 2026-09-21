"""Agent Loop：从「一份 JD + 一个项目描述」到「这个项目值不值得写进简历」。

每一步都必须落一条 Trace，四要素缺一不可：
    Action（这一步做了什么）
      -> Observation（真实看到了什么，带数字和行号）
      -> State Update（状态被改成了什么）
      -> Decision（下一步只能选 Continue / Adjust / Ask / Stop）

两条纪律：
  * 资料不足时不许猜 —— Decision=Ask，只问一个最有价值的问题，把判断交回给用户；
  * 证据够了就停 —— Decision=Stop，并写明 StopReason。

v1.1 的 LLM 装饰层（--llm，可选）：只在「规则报告已生成」之后介入，追加三块规则做不到的内容
（llm-suggest 建议写法草稿 / llm-interview 面试追问预演 / llm-polish 报告润色），每块都带 source。
LLM 不改结论、不改 Decision、不改词典；失败只降级成一块「调用失败」的文字。
不带 --llm 时这条链路一步都不走，Trace 与输出与 v1.0 完全一致。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .settings import LLMSettings
from .evidence import match_requirements
from .generate import (
    CallBudget,
    IDLE_REASON,
    TASK_ACTION,
    TASK_DUTY,
    failed_llm_report,
    generate_llm_report,
    unavailable_llm_report,
)
from .jd import load_jd
from .llm import (
    DEFAULT_CALL_LIMIT,
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_QWEN_VL_MODEL,
    DEFAULT_RETRIES,
    OpenAICompatClient,
)
from .project import apply_answer, load_project
from .schema import (
    DECISION_ADJUST,
    DECISION_ASK,
    DECISION_CONTINUE,
    DECISION_STOP,
    DECISIONS,
    EVIDENCE_ACTION,
    EVIDENCE_MENTION,
    EVIDENCE_NONE,
    EVIDENCE_RESULT,
    LEVEL_MUST,
    RISK_CONFLICT,
    RISK_GAP,
    RISK_PROBE,
    RISK_WORDING,
    VERDICT_EDIT,
    VERDICT_NO,
    VERDICT_YES,
    AgentState,
    EvidenceMatch,
    JobPosting,
    LlmBlock,
    LlmReport,
    Project,
    Reason,
    Risk,
    TraceStep,
    Verdict,
    VisionReport,
)
from .vision import collect_images, enrich_project

# 判定口径（全部写死在这里，方便直接检查，也方便日后调）
MAX_ROUNDS = 2          # 最多 Ask 一轮，问完还不足就给保守结论，不无限追问
MIN_FACTS = 3           # 项目描述少于 3 条事实，认为信息密度不够
COVERAGE_OK = 0.6       # 核心要求覆盖到这个比例，才敢说「值得写」
COVERAGE_MIN = 0.35     # 低于这个比例直接判「暂不建议写」


def run_agent(
    jd_path,
    project_path,
    answer: str = "",
    select_title: str = "",
    max_rounds: int = MAX_ROUNDS,
    llm: bool = False,
    vision: bool = False,
    llm_max_calls: int = DEFAULT_CALL_LIMIT,
    settings: Optional[LLMSettings] = None,
    text_client: Optional[OpenAICompatClient] = None,
    vision_client: Optional[OpenAICompatClient] = None,
    images: Sequence[str] = (),
    posting: Optional[JobPosting] = None,
    llm_budget: Optional[CallBudget] = None,
) -> AgentState:
    """跑完一次完整的 Agent Loop，返回带着 Trace 的状态。

    llm / vision 默认关闭：不传这两个开关时完全不联网，行为与纯规则版本一致。
    llm_max_calls 是 LLM 装饰层的调用上限（默认 10 次，含重试）。
    posting 用于批量模式：直接喂已解析好的岗位，跳过 load_jd；
    llm_budget 让多个岗位共用一份调用账本（一次运行的总调用数仍 ≤ llm_max_calls）。
    text_client / vision_client 用于注入替身（测试或二次开发），优先级高于 settings。
    """
    if max_rounds < 1:
        raise ValueError("max_rounds 至少为 1")

    trace: List[TraceStep] = []

    def record(action: str, observation: str, state_update: str, decision: str, reason: str) -> None:
        if decision not in DECISIONS:
            raise ValueError(f"Decision 只能是 {DECISIONS}，收到：{decision}")
        trace.append(
            TraceStep(
                index=len(trace) + 1,
                action=action,
                observation=observation,
                state_update=state_update,
                decision=decision,
                decision_reason=reason,
            )
        )

    # ---- 1. 读 JD ---------------------------------------------------------
    posting = posting if posting is not None else load_jd(jd_path, select_title)
    record(
        "ReadJD",
        f"读取 {Path(str(jd_path)).name}（{posting.line_count} 行），识别到 {posting.position_count} 个岗位，"
        f"锁定「{posting.title}」；该岗位 {posting.bullet_count} 行与能力要求相关",
        f"state.jd = 「{posting.title}」（{posting.company or '未标注公司'}）",
        DECISION_CONTINUE,
        "JD 可读，已定位到一个具体岗位",
    )

    # ---- 2. 读项目描述 ----------------------------------------------------
    project = load_project(project_path)
    counts = _level_counts(project)
    image_refs, _ = collect_images(project.text, project_path)
    picture_hint = (
        f"；另外引用了 {len(image_refs)} 张本地图片（本次未解析：加 --vision 可读成文字事实）"
        if image_refs and not vision
        else ""
    )
    record(
        "ReadProject",
        f"读取 {Path(str(project_path)).name}（{project.line_count} 行），解析出 {len(project.facts)} 条事实："
        f"有结果 {counts[EVIDENCE_RESULT]} / 有动作 {counts[EVIDENCE_ACTION]} / 仅提及 {counts[EVIDENCE_MENTION]}；"
        f"能看出「这是我做的」{len(project.ownership_facts)} 条{picture_hint}",
        f"state.project = {len(project.facts)} 条事实",
        DECISION_CONTINUE,
        "项目描述已拆成可核验的事实条目",
    )

    # ---- 3. 图片取证（可选；只有 --vision 才联网）-------------------------
    vision_refs, vision_notes = collect_images(project.text, project_path, tuple(images) if vision else ())
    vision_report = VisionReport(enabled=False, images=vision_refs, notes=list(vision_notes))

    if vision:
        vision_model = settings.vision_model if settings else DEFAULT_QWEN_VL_MODEL
        client = vision_client or _build_client(settings, "vision")
        before = len(project.facts)
        listed = "、".join(item.ref for item in vision_refs) or "无"

        if client is None:
            vision_report.error = "未配置 DASHSCOPE_API_KEY / QWEN_API_KEY"
            record(
                "EnrichWithVision",
                f"已开启 --vision，但没有可用的 Key（DASHSCOPE_API_KEY / QWEN_API_KEY）：本次不联网；"
                f"描述里发现 {len(vision_refs)} 张本地图片（{listed}）",
                "state.vision = 缺 Key，没有发出任何请求，证据池未变化",
                DECISION_CONTINUE,
                "没有 Key 就保持离线，规则链路与结论完全不受影响",
            )
        elif not vision_refs:
            record(
                "EnrichWithVision",
                "已开启 --vision，但项目描述里没有本地图片引用，也没有 --image 传入：没有可解析的内容",
                "state.vision = 0 张图片",
                DECISION_CONTINUE,
                "没有图片可读，直接进入能力抽取",
            )
        else:
            report = enrich_project(project, vision_refs, client, vision_model)
            report.notes = list(vision_notes) + list(report.notes)
            vision_report = report
            added_counts = Counter(fact.level for fact in project.facts[before:])
            note_tail = f"；{report.notes[0]}" if report.notes else ""
            record(
                "EnrichWithVision",
                f"用 {vision_model} 解析 {report.parsed}/{len(vision_refs)} 张本地图片，新增 {report.facts_added} 条文字事实"
                f"（有结果 {added_counts.get(EVIDENCE_RESULT, 0)} / 有动作 {added_counts.get(EVIDENCE_ACTION, 0)} / "
                f"仅提及 {added_counts.get(EVIDENCE_MENTION, 0)}）—— 等级一律由 classify() 判定，不采信模型自述"
                + (f"；{report.error}" if report.error else "")
                + note_tail,
                f"state.project.facts {before} -> {len(project.facts)}（新增事实的 source_file 形如「图片:相对路径」）",
                DECISION_CONTINUE,
                "图片事实已并入同一证据池，继续用规则检索"
                if report.facts_added
                else "图片没有提供可用事实，按原证据继续检索",
            )

    rounds = 1

    # ---- 2.5 用户补充（回答上一轮的 Ask）---------------------------------
    if answer.strip():
        added = apply_answer(project, answer)
        rounds += 1
        record(
            "ApplyUserAnswer",
            f"收到用户补充（{len(answer.strip())} 字），拆成 {added} 条新材料；"
            f"事实总数 {len(project.facts) - added} -> {len(project.facts)}",
            f"state.project.facts += {added}（来源：补充说明）",
            DECISION_CONTINUE,
            "新事实进入证据池，需要重新检索一遍",
        )

    # ---- 3. 抽岗位真正需要的能力 ------------------------------------------
    requirements = posting.requirements
    core = [item for item in requirements if item.provable]
    skipped = [item for item in requirements if not item.provable]
    must_count = len([item for item in core if item.level == LEVEL_MUST])
    record(
        "ExtractRequirements",
        f"{len(requirements)} 项能力里，{len(skipped)} 项属于门槛 / 素质类"
        f"（{'、'.join(item.name for item in skipped) or '无'}）——项目无法证明，移出对比；"
        f"剩下 {len(core)} 项进入检索，其中必备 {must_count} 项",
        f"state.core_requirements = {len(core)} 项，state.skipped = {len(skipped)} 项",
        DECISION_CONTINUE if core else DECISION_STOP,
        "已得到「项目能证明什么」的能力清单"
        if core
        else "这份 JD 里没有项目能证明的能力要求，继续检索没有意义",
    )

    state = AgentState(
        jd=posting,
        project=project,
        core_requirements=core,
        skipped_requirements=skipped,
        matches=[],
        trace=trace,
        rounds=rounds,
        vision=vision_report,
    )

    if not core:
        state.sufficiency = "no_requirement"
        state.verdict = Verdict(
            call=VERDICT_NO,
            headline="这份 JD 的要点全是学历 / 届别这类门槛条件，没有项目能证明的能力项，无法判断项目价值。",
            risks=[
                Risk(
                    text="把门槛条件当成项目判断依据会误导投递决策，建议换一份技术岗 JD 再跑一遍。",
                    kind=RISK_GAP,
                    refs=[posting.source_file],
                )
            ],
            stop_reason="核心要求清单为空，检索与判定都没有输入，直接停止。",
        )
        return state

    # ---- 4. 严格检索证据 --------------------------------------------------
    matches = match_requirements(core, project, allow_mention=False)
    strict_hits = [item for item in matches if item.has_evidence]
    missing = [item for item in matches if not item.has_evidence]
    result_hits = [item for item in strict_hits if item.level == EVIDENCE_RESULT]
    record(
        "RetrieveEvidence",
        f"严格匹配（只认「有动作 / 有结果」）：{len(strict_hits)}/{len(core)} 项命中，"
        f"其中带量化结果 {len(result_hits)} 项；无证据："
        f"{'、'.join(item.requirement_name for item in missing) or '无'}",
        f"state.matches = {len(matches)} 条（命中 {len(strict_hits)}）",
        DECISION_ADJUST if missing else DECISION_CONTINUE,
        f"{len(missing)} 项在严格规则下没证据，可能只是写成了技术栈 → 放宽规则再查一轮"
        if missing
        else "全部核心要求都已有「动作及以上」证据，不需要放宽规则",
    )

    # ---- 5. 放宽规则重查（Adjust）----------------------------------------
    if missing:
        matches = match_requirements(core, project, allow_mention=True)
        relaxed = [item for item in matches if item.relaxed]
        still_missing = [item for item in matches if not item.has_evidence]
        record(
            "AdjustEvidence",
            f"放宽后（允许「仅提及」级计入弱证据）新增 {len(relaxed)} 项弱证据"
            f"（{'、'.join(item.requirement_name for item in relaxed) or '无'}）；"
            f"仍有 {len(still_missing)} 项完全找不到："
            f"{'、'.join(item.requirement_name for item in still_missing) or '无'}",
            f"state.matches 更新为 {_match_brief(matches)}，relaxed = {len(relaxed)}",
            DECISION_CONTINUE,
            "证据已尽量取全，可以判断资料够不够用",
        )

    state.matches = matches

    # ---- 6. 资料够不够下结论？--------------------------------------------
    insufficient, question, why, diagnosis = assess_sufficiency(state)
    can_ask = bool(question) and rounds < max_rounds and not answer.strip()

    if insufficient and can_ask:
        state.sufficiency = "insufficient"
        state.question = question
        state.question_reason = why
        record(
            "CheckSufficiency",
            f"资料不足：{diagnosis}",
            f"state.sufficiency = insufficient；state.question = 「{question}」",
            DECISION_ASK,
            why,
        )
        return state

    state.sufficiency = "insufficient_after_ask" if insufficient else "enough"

    # ---- 7. 规则下结论：报告内容定稿（结论 / 理由 / 风险 / 建议写法 / StopReason）----
    verdict = build_verdict(state, insufficient=insufficient)
    state.verdict = verdict
    if insufficient:
        observation = (
            f"已问过一轮仍缺关键信息（{diagnosis}）；现有证据：核心覆盖 {len(state.solid)}/{len(core)}，"
            f"结果级证据 {state.result_match_count} 条"
        )
    else:
        observation = (
            f"核心覆盖 {len(state.solid)}/{len(core)}（{state.coverage:.0%}），"
            f"结果级证据 {state.result_match_count} 条，仍缺 {len(state.missing)} 项："
            f"{'、'.join(item.requirement_name for item in state.missing) or '无'}"
        )
    state_update = (
        f"state.verdict = 「{verdict.call}」；理由 {len(verdict.reasons)} 条、风险 {len(verdict.risks)} 条"
    )

    if not llm:
        # 不带 --llm：Judge 就是最后一步，与 v1.0 一字不差（完全不联网）
        record("Judge", observation, state_update, DECISION_STOP, verdict.stop_reason)
        return state

    record(
        "Judge",
        observation + "；以上由规则产出、已经定稿，LLM 不会改动它",
        state_update,
        DECISION_CONTINUE,
        "证据够了、结论已定；下一步只让 LLM 在这个已生成的报告上补三块规则做不到的内容",
    )

    # ---- 8~10. LLM 装饰层：报告已生成之后才介入，不改结论、不改 Decision ----
    report = _decorate_with_llm(state, text_client, settings, llm_max_calls, llm_budget)
    verdict.llm_report = report
    for block in report.blocks:
        record(
            TASK_ACTION.get(block.source, block.source),
            _llm_observation(report, block),
            _llm_state_update(report, block),
            DECISION_CONTINUE,
            "LLM 只往报告里追加带 source 的内容，不参与判定",
        )

    # ---- 11. 收尾：Stop 并写明 StopReason ---------------------------------
    record(
        "Finish",
        f"报告定稿：规则 {len(verdict.reasons)} 条理由 / {len(verdict.risks)} 条风险 + "
        f"LLM {report.ok_count}/{len(report.blocks)} 块（调用 {report.calls}/{report.call_limit} 次，"
        f"单次超时 {report.timeout:g}s，重试 {report.retries} 次）；"
        f"降级明细：{'、'.join(f'{item.source}={item.status}' for item in report.degraded) or '无'}",
        "state.final = 报告（规则段 + LLM 段，逐段带 [规则] / [LLM] 标记）",
        DECISION_STOP,
        verdict.stop_reason,
    )
    return state


def _build_client(
    settings: Optional[LLMSettings], layer: str, timeout: Optional[float] = None
) -> Optional[OpenAICompatClient]:
    """按配置建客户端；没配 key 就返回 None（缺 key 绝不发请求）。"""
    if settings is None:
        return None
    seconds = settings.timeout if timeout is None else timeout
    if layer == "text":
        if not settings.text_ready:
            return None
        return OpenAICompatClient(
            settings.text_base_url, settings.text_api_key, settings.text_model, seconds
        )
    if not settings.vision_ready:
        return None
    return OpenAICompatClient(
        settings.vision_base_url, settings.vision_api_key, settings.vision_model, seconds
    )


def _decorate_with_llm(
    state: AgentState,
    client: Optional[OpenAICompatClient],
    settings: Optional[LLMSettings],
    max_calls: int,
    budget: Optional[CallBudget] = None,
) -> LlmReport:
    """报告已生成之后调 DeepSeek，产出三块内容（草稿 / 追问预演 / 润色）。

    任何失败都只降级成块级文字：没有 Key → 三块「未启用」；调用失败 → 该块「调用失败」。
    budget 由调用方传进来时（批量跑多个岗位）多份报告共用同一份调用账本，总调用数不变。
    """
    model = settings.text_model if settings else DEFAULT_DEEPSEEK_MODEL
    limit = max(1, int(max_calls or DEFAULT_CALL_LIMIT))
    ledger = budget if budget is not None else build_text_budget(settings, client, limit=limit)
    if ledger is None:
        return unavailable_llm_report(model, "未配置 DEEPSEEK_API_KEY（没有联网）")
    try:
        return generate_llm_report(state, ledger)
    except Exception as exc:  # 兜底：装饰层不许打断主流程
        return failed_llm_report(ledger.model, f"生成层异常：{exc.__class__.__name__}: {exc}")


def build_text_budget(
    settings: Optional[LLMSettings],
    text_client: Optional[OpenAICompatClient] = None,
    limit: int = DEFAULT_CALL_LIMIT,
) -> Optional[CallBudget]:
    """建一份可复用的调用账本（批量跑多个岗位时共用同一份预算）。

    没有可用客户端（缺 Key）时返回 None，调用方按「未启用」处理，一次请求都不发。
    """
    model = settings.text_model if settings else DEFAULT_DEEPSEEK_MODEL
    target = text_client or _build_client(settings, "text", timeout=DEFAULT_CALL_TIMEOUT)
    if target is None:
        return None
    return CallBudget(
        target,
        model,
        limit=max(1, int(limit or DEFAULT_CALL_LIMIT)),
        timeout=DEFAULT_CALL_TIMEOUT,
        retries=DEFAULT_RETRIES,
    )


def _llm_observation(report: LlmReport, block: LlmBlock) -> str:
    duty = TASK_DUTY.get(block.source, "")
    if block.idle:
        return (
            f"{duty}：已闲置（{IDLE_REASON}），本次没有发出调用；"
            "报告里仍保留这一块，并标注「已闲置」"
        )
    if block.skipped:
        return f"{duty}：没有调用（{block.error}），该块标「已达调用上限」"
    if block.error and not block.calls:
        return f"{duty}：没有发出调用（{block.error}）"
    if block.error:
        return (
            f"{duty}：调用 {block.calls} 次后失败（{block.error}）；"
            f"该块显示「{block.status}」，上面的规则报告不受影响"
        )
    return (
        f"{duty}：产出 {block.summary()}（source={block.source}，模型 {block.model}）；"
        "内容与规则段落分开标注，规则结论未被替换"
    )


def _llm_state_update(report: LlmReport, block: LlmBlock) -> str:
    return (
        f"state.llm_report.blocks[{block.source}] = {block.status}"
        f"（本次累计调用 {report.calls}/{report.call_limit}）"
    )


def assess_sufficiency(state: AgentState) -> Tuple[bool, str, str, str]:
    """资料够不够下结论？返回 (是否不足, 要问的问题, 为什么问, 现状一句话)。"""
    project = state.project
    hit = [item for item in state.matches if item.has_evidence]
    missing_must = [item for item in state.missing if item.requirement.level == LEVEL_MUST]
    thin = len(project.facts) < MIN_FACTS

    if not project.actionable_facts:
        return (
            True,
            "这个项目里你个人具体做了什么？哪几个模块是你独立完成的？",
            "项目描述里看不出你的贡献，替你猜会直接把结论带偏",
            f"{len(project.facts)} 条事实里没有一条能看出「我做了什么」，全是背景介绍与技术栈罗列",
        )

    if not hit:
        target = missing_must[0] if missing_must else state.missing[0]
        return (
            True,
            f"岗位在 {target.requirement.ref} 要求「{target.requirement_name}」，项目里没提到 —— "
            f"这块你实际做过吗？做到什么程度？",
            "项目与岗位核心要求零交集，通常说明项目描述写得太粗，而不是你真的没做过",
            f"核心要求 {len(state.core_requirements)} 项，项目里命中 0 项",
        )

    if thin and not (project.result_facts and state.coverage >= 0.5):
        return (
            True,
            "能不能再补 3 行：这个项目你负责哪一块、用了什么方法、最后结果如何？",
            "信息密度太低，任何判断都会变成猜测",
            f"项目描述只有 {len(project.facts)} 条事实",
        )

    if not project.result_facts and state.coverage < 0.5:
        return (
            True,
            "这个项目最后跑出来的结果是什么？（准确率 / 召回率 / 耗时 / 用户数，有一个数字就够）",
            "没有结果级证据，写进简历只能证明「做过」，撑不起「做好了」",
            f"没有一条带量化结果的事实，核心覆盖只有 {state.coverage:.0%}",
        )

    return False, "", "", ""


def build_verdict(state: AgentState, insufficient: bool = False) -> Verdict:
    """把证据状态翻译成：值不值得写 + 2~3 条理由 + 1~2 个风险 + StopReason。"""
    core = state.core_requirements
    solid = sorted(state.solid, key=lambda item: -item.score)
    result_matches = [item for item in solid if item.level == EVIDENCE_RESULT]
    missing_must = [item for item in state.missing if item.requirement.level == LEVEL_MUST]
    coverage = state.coverage
    jd_title = state.jd.name

    if not solid or coverage < COVERAGE_MIN:
        call = VERDICT_NO
    elif coverage >= COVERAGE_OK and result_matches and not missing_must:
        call = VERDICT_YES
    else:
        call = VERDICT_EDIT

    headline = _headline(call, state, jd_title, solid, result_matches, coverage)
    if insufficient:
        call = VERDICT_NO
        missing_names = '、'.join(item.requirement_name for item in state.missing)
        headline = (
            "补充材料后仍然缺少关键事实（"
            + (missing_names or "项目的个人贡献与结果")
            + "），按现有信息只能给保守结论：暂不建议写。"
        )

    return Verdict(
        call=call,
        headline=headline,
        reasons=_build_reasons(state, solid),
        risks=_build_risks(state, solid),
        rewrite=_build_rewrite(state, solid),
        stop_reason=_stop_reason(state, call, insufficient),
    )


def _headline(call, state, jd_title, solid, result_matches, coverage) -> str:
    if call == VERDICT_YES:
        return (
            f"能撑住「{jd_title}」{len(solid)}/{len(state.core_requirements)} 项核心要求"
            f"（覆盖 {coverage:.0%}），其中 {len(result_matches)} 项带量化结果，可以写进简历。"
        )
    if call == VERDICT_EDIT:
        return (
            f"只撑得住 {len(solid)}/{len(state.core_requirements)} 项核心要求（覆盖 {coverage:.0%}），"
            f"还差 {len(state.missing)} 项；可以写，但必须限定在你真做过、讲得清的部分。"
        )
    return (
        f"与「{jd_title}」的核心要求只对上 {len(solid)}/{len(state.core_requirements)} 项"
        f"（覆盖 {coverage:.0%}），写进简历容易在追问里露馅，建议换项目或先补做。"
    )


def _build_reasons(state: AgentState, solid: Sequence[EvidenceMatch]) -> List[Reason]:
    reasons: List[Reason] = []
    used_facts = set()
    used_jd_lines = set()
    picked = set()

    # 第一轮：尽量让理由来自不同的 JD 行和不同的项目证据，避免三条理由说同一件事
    for match in solid:
        if len(reasons) >= 3:
            break
        fact = match.best_fact
        if fact is None or fact.ref in used_facts or match.requirement.ref in used_jd_lines:
            continue
        used_facts.add(fact.ref)
        used_jd_lines.add(match.requirement.ref)
        picked.add(match.requirement_name)
        reasons.append(_reason(match, fact))

    # 第二轮：证据行数不够时放宽去重，但同一条要求不重复出现
    if len(reasons) < 3:
        for match in solid:
            if len(reasons) >= 3:
                break
            fact = match.best_fact
            if fact is None or match.requirement_name in picked:
                continue
            picked.add(match.requirement_name)
            reasons.append(_reason(match, fact))

    # 强证据不足 2 条时用「仅提及」级证据补上，但明确标注是弱证据
    if len(reasons) < 2:
        for match in [item for item in state.matches if item.level == EVIDENCE_MENTION]:
            if len(reasons) >= 2:
                break
            fact = match.best_fact
            if fact is None or match.requirement_name in picked:
                continue
            picked.add(match.requirement_name)
            reasons.append(_reason(match, fact, weak=True))

    if not reasons:
        reasons.append(
            Reason(
                text=f"对照了 JD 的 {len(state.core_requirements)} 项核心要求，项目里没有任何一条能对得上的证据",
                requirement_name="（无匹配）",
                jd_ref=state.jd.source_file,
                project_refs=[],
            )
        )
    return reasons


def _reason(match: EvidenceMatch, fact, weak: bool = False) -> Reason:
    requirement = match.requirement
    prefix = "（弱证据）" if weak else ""
    middle = "项目里只是在技术栈里出现过" if weak else "项目里有对应证据"
    return Reason(
        text=(
            f"{prefix}JD 的{requirement.level_label}提到「{requirement.name}」：{requirement.quote_short}"
            f" → {middle}：{fact.quote_short}（{fact.ref}，{fact.level_label}）"
        ),
        requirement_name=requirement.name,
        jd_ref=requirement.ref,
        project_refs=[item.ref for item in match.facts[:2]],
    )


def _build_risks(state: AgentState, solid: Sequence[EvidenceMatch]) -> List[Risk]:
    risks: List[Risk] = []

    for match in [item for item in state.missing if item.requirement.level == LEVEL_MUST]:
        if len(risks) >= 2:
            break
        risks.append(
            Risk(
                text=(
                    f"「{match.requirement_name}」是 JD 的{match.requirement.level_label}（{match.requirement.ref}），"
                    f"这段项目里没有对应证据：别把它包装成这条能力；面试前要么补做，要么准备好承认没做过。"
                ),
                kind=RISK_GAP,
                refs=[match.requirement.ref],
            )
        )

    for match in state.missing:
        if len(risks) >= 2:
            break
        if not match.note.startswith("项目描述里明确写了"):
            continue
        facts = state.project.facts_for(match.requirement.capability_key)
        refs = [fact.ref for fact in facts[:2]] or [match.requirement.ref]
        risks.append(
            Risk(
                text=(
                    f"项目描述里写了没做过「{match.requirement_name}」（{'、'.join(refs)}），"
                    f"简历里若出现相关表述就是自相矛盾。"
                ),
                kind=RISK_CONFLICT,
                refs=refs,
            )
        )

    for match in [item for item in solid if item.level == EVIDENCE_ACTION]:
        if len(risks) >= 2:
            break
        fact = match.best_fact
        refs = [fact.ref] if fact else [match.requirement.ref]
        risks.append(
            Risk(
                text=(
                    f"「{match.requirement_name}」只有动作、没有结果（{refs[0]}），"
                    f"面试官会追问指标：提前准备一句量化对比，哪怕只是小样本。"
                ),
                kind=RISK_PROBE,
                refs=refs,
            )
        )

    for match in [item for item in state.matches if item.level == EVIDENCE_MENTION]:
        if len(risks) >= 2:
            break
        fact = match.best_fact
        refs = [fact.ref] if fact else [match.requirement.ref]
        risks.append(
            Risk(
                text=(
                    f"「{match.requirement_name}」只在技术栈里出现过（{refs[0]}），"
                    f"只能写「用过」，不能写「熟练 / 精通」。"
                ),
                kind=RISK_WORDING,
                refs=refs,
            )
        )

    if not risks:
        fact = state.project.result_facts[0] if state.project.result_facts else None
        refs = [fact.ref] if fact else [state.project.source_file]
        risks.append(
            Risk(
                text=(
                    f"证据链完整，但面试大概率会往实现细节追问（例如 {refs[0]} 那句）："
                    f"提前准备 3 句话讲清做法与取舍。"
                ),
                kind=RISK_PROBE,
                refs=refs,
            )
        )
    return risks[:2]


def _build_rewrite(state: AgentState, solid: Sequence[EvidenceMatch]) -> str:
    """按已核验的证据拼一版简历写法草稿 —— 只用事实，不加形容词。"""
    top = [item for item in solid if item.level == EVIDENCE_RESULT][:2] or list(solid[:2])
    parts: List[str] = []
    for match in top:
        fact = match.best_fact
        if fact is None:
            continue
        parts.append(f"{match.requirement_name}：{fact.quote_short}（{fact.ref}）")
    if not parts:
        return ""
    metrics = [metric for fact in state.project.result_facts for metric in fact.metrics][:3]
    tail = f"；量化结果：{'、'.join(metrics)}" if metrics else ""
    return f"建议写法（只用已核验证据拼成）：{state.project.name}｜" + "；".join(parts) + tail


def _stop_reason(state: AgentState, call: str, insufficient: bool) -> str:
    if insufficient:
        missing = '、'.join(item.requirement_name for item in state.missing)
        return (
            f"已经问过一轮，补充后仍缺「{missing or '项目的个人贡献与结果'}」；"
            f"再问下去只是在拖延，按现有信息给「{call}」并停止。"
        )
    return (
        f"{len(state.core_requirements)} 项核心要求已逐条核对完（覆盖 {state.coverage:.0%}、"
        f"结果级证据 {state.result_match_count} 条、缺口 {len(state.missing)} 项），"
        f"证据足以支撑「{call}」这个结论，继续检索不会产生新信息。"
    )


def _level_counts(project: Project) -> Counter:
    return Counter(fact.level for fact in project.facts)


def _match_brief(matches: Sequence[EvidenceMatch]) -> str:
    counts = Counter(item.level for item in matches)
    return (
        f"有结果 {counts.get(EVIDENCE_RESULT, 0)} / 有动作 {counts.get(EVIDENCE_ACTION, 0)} / "
        f"仅提及 {counts.get(EVIDENCE_MENTION, 0)} / 无证据 {counts.get(EVIDENCE_NONE, 0)}"
    )
