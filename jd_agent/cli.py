"""命令行入口：把「一份 JD + 一个项目描述」喂给 Agent，输出判断与运行 Trace。

    python main.py                                   # 不指定 --jd 时：读 input/jd 下所有文件的全部岗位
    python main.py --jd input/jd/xx.md               # 只跑这个文件（文件里多个岗位时默认只跑第一个）
    python main.py --jd input/jd/xx.md --jd-title 岗位二   # 指定这个文件里的哪个岗位
    python main.py --project input/profile/我的简历.md     # 第二个输入可以是任何 md，包括简历
    python main.py --answer "这个项目的结果是……"      # 回答 Agent 提出的那一个问题

简历排版工具（纯规则、不联网、不调用大模型；三套风格 classic / compact / accent）：
    python main.py --build-resume                    # 把 input/profile 下的简历排成 output/resume.md + .html
    python main.py --build-resume --resume-style compact     # 换风格 → output/resume-compact.md + .html
    python main.py --build-resume --resume-file input/profile/个人简历1.md
    python main.py --build-resume --resume-project input/project/示例项目1-AI周报助手.md

可选的大模型能力（默认关闭，不联网；先配好项目根 .env）：
    python main.py --llm                            # 报告生成后：写法草稿 + 报告润色（两块都带 source）
    python main.py --llm --llm-max-calls 3          # 限制本次运行的 LLM 调用次数（默认 10，多岗位共用）
    python main.py --vision                         # Qwen-VL 把项目里引用的图片读成文字事实
    python main.py --llm --vision --check-llm       # 先自检 key / 网络 / 模型名

退出码：0 = 已给出判断（Stop）｜3 = 需要你回答一个问题（Ask）｜2 = 输入有误
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import __version__
from .agent import MAX_ROUNDS, build_text_budget, run_agent
from .jd import load_jd, load_jd_positions
from .tools import RESUME_FORMATS, RESUME_TOOL, resume_styles, tool_help_lines
from .schema import JobPosting
from .settings import DEFAULT_ENV_FILE, EnvLoadResult, LLMSettings, load_env, resolve_settings
from .llm import DEFAULT_CALL_LIMIT, OpenAICompatClient, check_llm
from .render import render_console, render_html, render_json, render_markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
JD_DIR = INPUT_DIR / "jd"
PROJECT_DIR = INPUT_DIR / "project"
RESUME_DIR = INPUT_DIR / "profile"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_STEM = "resume_decision"
FORMATS = ("md", "json", "html")
EXIT_OK = 0
EXIT_ERROR = 2
EXIT_ASK = 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-project-judge",
        description="AI 求职尽调 Agent：读一份 JD + 一个项目描述，判断这个项目值不值得写进简历",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py                       # 不指定 --jd：跑 input/jd 下所有文件的全部岗位\n"
            "  python main.py --jd input/jd/xx.md --jd-title 岗位二\n"
            "  python main.py --answer \"周报助手的结果是召回率 0.82\"\n"
            "  python main.py --project input/profile/资料不足项目测试简历1.md\n"
            "  python main.py --build-resume        # 简历排版：output/resume.md + output/resume.html\n"
            "  python main.py --build-resume --resume-style compact   # 换风格 → resume-compact.*\n"
            "\n可用工具（jd_agent/tools/）：\n"
            + "".join(f"  {line}\n" for line in tool_help_lines())
        ),
    )
    parser.add_argument("--jd", help="JD 文件（不指定时读 input/jd/ 下所有文件的全部岗位）")
    parser.add_argument("--jd-title", dest="jd_title", help="JD 文件里有多个岗位时，用标题关键字指定（如「岗位二」）")
    parser.add_argument(
        "--project",
        help="项目描述文件（默认取 input/project/ 下的第一个 .md；也可以是 input/profile/ 下的简历 md）",
    )
    parser.add_argument(
        "--build-resume",
        dest="build_resume",
        action="store_true",
        help="简历排版：把简历 md 重排成简约大方的 md / html（纯规则，不联网、不调大模型）",
    )
    parser.add_argument(
        "--resume-file",
        dest="resume_file",
        help=f"简历 md（默认取 {RESUME_DIR} 下第一个非「示例」文件）",
    )
    parser.add_argument(
        "--resume-project",
        dest="resume_projects",
        action="append",
        help="排版时额外并入「项目经历」的项目描述文件，可重复",
    )
    parser.add_argument(
        "--resume-format",
        dest="resume_formats",
        default=",".join(RESUME_FORMATS),
        help="简历输出格式，逗号分隔：md,html（默认两个都出）",
    )
    parser.add_argument(
        "--resume-style",
        dest="resume_style",
        default=resume_styles.DEFAULT_STYLE,
        help="简历排版风格："
        + " / ".join(f"{style.key} {style.name}" for style in resume_styles.all_styles())
        + f"（默认 {resume_styles.DEFAULT_STYLE}）",
    )
    parser.add_argument(
        "--resume-name",
        dest="resume_name",
        help="简历文件名（默认按风格取名：resume / resume-compact / resume-accent）",
    )
    parser.add_argument("--answer", help="回答 Agent 提出的问题；回答会作为新材料并入证据池后重新检索")
    parser.add_argument("--max-rounds", dest="max_rounds", type=int, default=MAX_ROUNDS, help=f"最多检索轮数（默认 {MAX_ROUNDS}）")
    parser.add_argument("--out", help=f"输出目录（默认 {OUTPUT_DIR}）")
    parser.add_argument("--formats", default=",".join(FORMATS), help="输出格式，逗号分隔：md,json,html")
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "可选：规则报告生成之后调用 DeepSeek（deepseek-chat），做规则做不好的事——"
            "建议写法草稿（llm-suggest）/ 报告润色（llm-polish）；面试追问预演（llm-interview）当前已闲置；"
            "每块都带 source 标记，不改结论、不改 Decision"
        ),
    )
    parser.add_argument(
        "--llm-max-calls",
        dest="llm_max_calls",
        type=int,
        default=DEFAULT_CALL_LIMIT,
        help=f"LLM 装饰层的调用上限，含重试（默认 {DEFAULT_CALL_LIMIT}）",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="可选：调用 Qwen-VL（qwen-vl-max）把项目描述里引用的本地图片解析成文字事实；证据等级仍由规则判定",
    )
    parser.add_argument("--env", help=f"API Key 所在的 .env 文件（默认 {DEFAULT_ENV_FILE}）")
    parser.add_argument("--text-model", dest="text_model", help="覆盖文本层模型名（默认 DEEPSEEK_MODEL 或 deepseek-chat）")
    parser.add_argument("--vision-model", dest="vision_model", help="覆盖多模态层模型名（默认 QWEN_VL_MODEL 或 qwen-vl-max）")
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        help="额外指定要解析的本地图片，可重复；会和项目描述里引用的图片一起处理",
    )
    parser.add_argument(
        "--check-llm",
        dest="check_llm",
        action="store_true",
        help="自检：给启用的层各发一条最小请求，确认 key / 网络 / 模型名可用（加了 --vision 就一起检查多模态层）",
    )
    parser.add_argument("--quiet", action="store_true", help="不在终端打印 Trace 与结论")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _console_safe(text: str) -> str:
    """避免在 GBK 控制台上因 emoji 直接崩溃。"""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _say(text: str = "") -> None:
    print(_console_safe(text))


def _md_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.md"))


def _first_md(directory: Path) -> Optional[Path]:
    files = _md_files(directory)
    return files[0] if files else None


def resolve_jd_path(explicit: Optional[str]) -> Path:
    """只回答「第一个 JD 文件是谁」：显式给了就用它，否则 input/jd 下第一个 .md。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 JD 文件：{path}")
        return path
    found = _first_md(JD_DIR) or _first_md(INPUT_DIR)
    if found is None:
        raise FileNotFoundError(f"没有找到 JD 文件。请把岗位描述放进 {JD_DIR}，或用 --jd 指定。")
    return found


def resolve_jd_files(explicit: Optional[str]) -> Tuple[List[Path], bool]:
    """返回 (JD 文件列表, 是否没指定 --jd)。

    没指定 --jd 时读 input/jd 下全部 .md（批量模式）；指定了就只跑那一个文件，行为与 v1.0 一致。
    """
    if explicit:
        return [resolve_jd_path(explicit)], False
    files = _md_files(JD_DIR)
    if not files:
        raise FileNotFoundError(f"没有找到 JD 文件。请把岗位描述放进 {JD_DIR}，或用 --jd 指定。")
    return files, True


def collect_postings(
    files: Sequence[Path], select_title: str, all_positions: bool
) -> List[Tuple[Path, JobPosting]]:
    """把 JD 文件展开成（文件, 岗位）列表。

    批量模式下一个文件里的每个岗位都跑；--jd 指定单个文件时保持 v1.0 行为——默认只跑第一个岗位，
    用 --jd-title 才切换。
    """
    collected: List[Tuple[Path, JobPosting]] = []
    for path in files:
        if all_positions:
            collected += [(path, posting) for posting in load_jd_positions(path)]
        else:
            collected.append((path, load_jd(path, select_title)))
    return collected


def resolve_resume_path(explicit: Optional[str]) -> Path:
    """简历 md：显式给了就用它，否则 input/profile 下第一个非「示例」文件。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到简历文件：{path}")
        return path
    real = [item for item in _md_files(RESUME_DIR) if "示例" not in item.name]
    found = real[0] if real else _first_md(RESUME_DIR)
    if found is None:
        raise FileNotFoundError(f"没有找到简历文件。请把简历 md 放进 {RESUME_DIR}，或用 --resume-file 指定。")
    return found


def _slug(text: str) -> str:
    """岗位名 -> 文件名：非法字符与空白换成下划线。"""
    slug = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text).replace("#", "_"))
    return slug.strip("_.") or "jd"


def report_stem(posting: JobPosting, total: int) -> str:
    """只有一个岗位时沿用 v1.0 的 resume_decision.*，多个岗位时按岗位分文件。"""
    return REPORT_STEM if total <= 1 else f"{REPORT_STEM}_{_slug(posting.jd_id)}"


def resolve_project_path(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到项目描述文件：{path}")
        return path
    found = _first_md(PROJECT_DIR)
    if found is None:
        raise FileNotFoundError(f"没有找到项目描述文件。请把项目描述放进 {PROJECT_DIR}，或用 --project 指定。")
    return found


def _prepare_settings(args) -> Tuple[Optional[LLMSettings], EnvLoadResult, str]:
    """读 .env 并解析配置；返回 (配置, 加载结果, 错误信息)。日志里只有键名，没有值。"""
    env_path = Path(args.env).expanduser() if args.env else DEFAULT_ENV_FILE
    if args.env and not env_path.is_file():
        return None, EnvLoadResult(path=str(env_path)), f"找不到 .env 文件：{env_path}"
    loaded = load_env(env_path)
    if loaded.error:
        return None, loaded, loaded.error
    settings = resolve_settings(
        env_file=loaded.path,
        env_keys=tuple(loaded.applied) + tuple(loaded.skipped),
        text_model=args.text_model or "",
        vision_model=args.vision_model or "",
    )
    return settings, loaded, ""


def run_build_resume(args) -> int:
    """简历排版模式：读简历 md，输出排好版的 md / html（纯规则，不联网、不调大模型）。

    真正的活交给 jd_agent.tools 里的 RESUME_TOOL；这里只负责挑默认路径、打印结果。
    """
    try:
        path = resolve_resume_path(args.resume_file)
    except FileNotFoundError as exc:
        _say(f"[错误] {exc}")
        return EXIT_ERROR

    formats = [item.strip().lower() for item in args.resume_formats.split(",") if item.strip()]
    projects = [Path(item).expanduser() for item in (args.resume_projects or ())]
    result = RESUME_TOOL.run(
        resume_file=path,
        out_dir=Path(args.out).expanduser() if args.out else OUTPUT_DIR,
        formats=formats,
        stem=args.resume_name or None,
        style=args.resume_style,
        projects=projects,
    )
    if not result.ok:
        _say(f"[错误] {result.error}")
        return EXIT_ERROR

    if not args.quiet:
        _say("")
        _say("简历排版完成（纯规则：只重排结构、规整空白，不新增任何事实）")
        _say("=" * 72)
        for key, value in result.summary:
            _say(f"  {key}：{value}")
        if result.notes:
            _say("")
            _say("排版规整：")
            for note in result.notes:
                _say(f"  - {note}")
        _say("")
        _say("已生成：")
        for item in result.written:
            _say(f"  - {item}")
        _say("")
    return EXIT_OK


def main(
    argv: Optional[Sequence[str]] = None,
    text_client: Optional[OpenAICompatClient] = None,
    vision_client: Optional[OpenAICompatClient] = None,
) -> int:
    """命令行入口。

    text_client / vision_client 用于注入替身（测试或二次开发）；默认 None 时按 .env 建真实客户端。
    """
    args = build_arg_parser().parse_args(argv)
    use_llm = bool(args.llm)
    use_vision = bool(args.vision)
    settings: Optional[LLMSettings] = None

    if args.build_resume:
        return run_build_resume(args)

    if use_llm or use_vision or args.check_llm:
        settings, loaded, error = _prepare_settings(args)
        if error:
            _say(f"[错误] {error}")
            return EXIT_ERROR
        if not args.quiet:
            _say(f"[配置] .env：{loaded.summary()}")
            for line in settings.describe():
                _say(f"[配置] {line}")
            _say("")

    if args.check_llm:
        results = check_llm(
            settings, text_client=text_client, vision_client=vision_client, check_vision=use_vision
        )
        failed = 0
        for item in results:
            _say(f"[{'通过' if item.ok else '失败'}] {item.label}（{item.model}）：{item.detail}")
            failed += 0 if item.ok else 1
        _say("")
        if failed:
            _say(f"自检有 {failed} 项没通过：先按上面的报错检查 key / 网络 / 模型名，再加 --llm / --vision 跑正式流程。")
            return EXIT_ERROR
        _say("自检通过：key、网络与模型名都可用。")
        return EXIT_OK

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unknown = [item for item in formats if item not in FORMATS]
    if unknown:
        _say(f"[错误] 不支持的输出格式：{', '.join(unknown)}（可选 md / json / html）")
        return EXIT_ERROR

    try:
        jd_files, batch = resolve_jd_files(args.jd)
        project_path = resolve_project_path(args.project)
        collected = collect_postings(jd_files, args.jd_title or "", batch)
    except (FileNotFoundError, ValueError) as exc:  # 例如 --jd-title 没匹配到任何岗位
        _say(f"[错误] {exc}")
        return EXIT_ERROR

    usable = [(path, posting) for path, posting in collected if posting.requirements]
    empty = [posting for _, posting in collected if not posting.requirements]
    if not usable:
        _say(f"[错误] 没有识别到任何岗位要求：{'、'.join(item.source_file for item in empty) or '无'}")
        return EXIT_ERROR

    # 一次运行里所有岗位共用同一份调用账本，总调用数仍然 ≤ --llm-max-calls
    budget = build_text_budget(settings, text_client, limit=max(1, args.llm_max_calls)) if use_llm else None

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    renderers = {
        "md": (render_markdown, "md"),
        "json": (render_json, "json"),
        "html": (render_html, "html"),
    }
    results = []  # [(岗位, 运行结果, 写出的文件)]
    total = len(usable)
    for index, (jd_path, posting) in enumerate(usable, start=1):
        state = run_agent(
            jd_path,
            project_path,
            answer=args.answer or "",
            select_title=args.jd_title or "",
            max_rounds=max(1, args.max_rounds),
            llm=use_llm,
            vision=use_vision,
            llm_max_calls=max(1, args.llm_max_calls),
            settings=settings,
            text_client=text_client,
            vision_client=vision_client,
            images=tuple(args.images or ()),
            posting=posting,
            llm_budget=budget,
        )
        written: List[Path] = []
        for name in formats:
            renderer, suffix = renderers[name]
            path = out_dir / f"{report_stem(posting, total)}.{suffix}"
            path.write_text(renderer(state, stamp), encoding="utf-8")
            written.append(path)
        results.append((posting, state, written))
        if not args.quiet:
            if total > 1:
                _say("")
                _say("=" * 72)
                _say(f"[{index}/{total}] {posting.name}（{posting.source_file}）")
            for line in render_console(state):
                _say(line)

    if not args.quiet:
        if total > 1:
            _say("")
            _say("=" * 72)
            _say(f"批量结果汇总（{total} 个岗位，读自 {len(jd_files)} 个 JD 文件）")
            _say("=" * 72)
            for index, (posting, state, _) in enumerate(results, start=1):
                verdict = state.verdict
                call = verdict.call if verdict else "（资料不足，等你的回答）"
                _say(
                    f"  [{index}] {posting.name}｜核心覆盖 {len(state.solid)}/{len(state.core_requirements)}"
                    f"｜{call}｜Decision={state.decision}"
                )
        if empty:
            _say("")
            _say("跳过（没有识别到岗位要求）：")
            for posting in empty:
                _say(f"  - {posting.source_file}（{posting.name}）")
        _say("")
        _say("已生成：")
        for _, _, written in results:
            for path in written:
                _say(f"  - {path}")
        asked = [(posting, state) for posting, state, _ in results if state.needs_answer]
        if asked and total > 1:  # 单岗位时上面的 render_console 已经把这个问题和原因打全了
            _say("")
            for posting, state in asked:
                _say(f"[需要你回答] {posting.name}：{state.question}")
            _say(f"  答完后再跑一次（退出码 {EXIT_ASK}）：python main.py --answer \"你的回答\"")
        _say("")

    return EXIT_ASK if any(state.needs_answer for _, state, _ in results) else EXIT_OK
