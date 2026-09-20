"""命令行入口：把「一份 JD + 一个项目描述」喂给 Agent，输出判断与运行 Trace。

    python main.py                                   # 自动读 input/jd 与 input/project
    python main.py --jd input/jd/xx.md --project input/project/xx.md
    python main.py --jd-title 岗位二                 # JD 文件里有多个岗位时选一个
    python main.py --answer "这个项目的结果是……"      # 回答 Agent 提出的那一个问题

退出码：0 = 已给出判断（Stop）｜3 = 需要你回答一个问题（Ask）｜2 = 输入有误
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .agent import MAX_ROUNDS, run_agent
from .render import render_console, render_html, render_json, render_markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
JD_DIR = INPUT_DIR / "jd"
PROJECT_DIR = INPUT_DIR / "project"
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
            "  python main.py\n"
            "  python main.py --jd-title 岗位二\n"
            "  python main.py --answer \"周报助手的结果是召回率 0.82\"\n"
            "  python main.py --project input/project/示例项目2-资料不足.md\n"
        ),
    )
    parser.add_argument("--jd", help="JD 文件（默认取 input/jd/ 下的第一个 .md）")
    parser.add_argument("--jd-title", dest="jd_title", help="JD 文件里有多个岗位时，用标题关键字指定（如「岗位二」）")
    parser.add_argument("--project", help="项目描述文件（默认取 input/project/ 下的第一个 .md）")
    parser.add_argument("--answer", help="回答 Agent 提出的问题；回答会作为新材料并入证据池后重新检索")
    parser.add_argument("--max-rounds", dest="max_rounds", type=int, default=MAX_ROUNDS, help=f"最多检索轮数（默认 {MAX_ROUNDS}）")
    parser.add_argument("--out", help=f"输出目录（默认 {OUTPUT_DIR}）")
    parser.add_argument("--formats", default=",".join(FORMATS), help="输出格式，逗号分隔：md,json,html")
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


def _first_md(directory: Path) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.md"))
    return files[0] if files else None


def resolve_jd_path(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 JD 文件：{path}")
        return path
    found = _first_md(JD_DIR) or _first_md(INPUT_DIR)
    if found is None:
        raise FileNotFoundError(f"没有找到 JD 文件。请把岗位描述放进 {JD_DIR}，或用 --jd 指定。")
    return found


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        jd_path = resolve_jd_path(args.jd)
        project_path = resolve_project_path(args.project)
    except FileNotFoundError as exc:
        _say(f"[错误] {exc}")
        return EXIT_ERROR

    try:
        state = run_agent(
            jd_path,
            project_path,
            answer=args.answer or "",
            select_title=args.jd_title or "",
            max_rounds=max(1, args.max_rounds),
        )
    except ValueError as exc:  # 例如 --jd-title 没匹配到任何岗位
        _say(f"[错误] {exc}")
        return EXIT_ERROR

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unknown = [item for item in formats if item not in FORMATS]
    if unknown:
        _say(f"[错误] 不支持的输出格式：{', '.join(unknown)}（可选 md / json / html）")
        return EXIT_ERROR

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    renderers = {
        "md": (render_markdown, "md"),
        "json": (render_json, "json"),
        "html": (render_html, "html"),
    }
    for name in formats:
        renderer, suffix = renderers[name]
        path = out_dir / f"{REPORT_STEM}.{suffix}"
        path.write_text(renderer(state, stamp), encoding="utf-8")
        written.append(path)

    if not args.quiet:
        for line in render_console(state):
            _say(line)
        _say("")
        _say("已生成：")
        for path in written:
            _say(f"  - {path}")
        if state.needs_answer:
            _say("")
            _say(f"[需要你回答] 上面那个问题答完后再跑一次（退出码 {EXIT_ASK}）")
        _say("")

    return EXIT_ASK if state.needs_answer else EXIT_OK
