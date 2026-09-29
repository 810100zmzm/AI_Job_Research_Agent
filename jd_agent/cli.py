"""命令行入口：把「一份 JD + 一个项目描述」喂给 Agent，输出判断与运行 Trace。

    python main.py                                   # 不指定 --jd 时：读 input/jd 下所有文件的全部岗位
    python main.py --jd input/jd/xx.md               # 只跑这个文件（文件里多个岗位时默认只跑第一个）
    python main.py --jd input/jd/xx.md --jd-title 岗位二   # 指定这个文件里的哪个岗位
    python main.py --project input/profile/我的简历.md     # 第二个输入可以是任何 md，包括简历
    python main.py --answer "这个项目的结果是……"      # 回答 Agent 提出的那一个问题

简历排版工具（纯规则、不联网、不调用大模型；三套风格 classic / structure / accent）：
    python main.py --build-resume                    # 把 input/profile 下的简历排成 output/resume.md + .html
    python main.py --build-resume --resume-style structure   # 换风格 → output/resume-structure.md + .html
    python main.py --build-resume --resume-file input/profile/个人简历1.md
    python main.py --build-resume --resume-project input/project/示例项目1-AI周报助手.md

JD Agent（LangGraph）：把 JD（文本 / 图片 / 网址）转成统一模板的 Markdown：
    python main.py --jd-agent --jd-text "岗位职责：……"       # 文本直接读取（不联网）
    python main.py --jd-agent --jd-file input/jd/xx.md        # 文件走文件解析 Tool（纯规则）
    python main.py --jd-agent --jd-image assets/jd.png        # 图片走 Qwen-VL（需 .env）
    python main.py --jd-agent --jd-url "https://.../job/1"    # 网址抓一次 HTML → Markdown
    python main.py --jd-agent --jd-url "https://..." --jd-trace   # 额外写出完整 Trace

Resume Agent（LangGraph）：把简历拆成事实条目，逐条判证据等级：
    python main.py --resume-agent --cv-file input/profile/xx.md   # 文件：走文件解析 Tool（纯规则）
    python main.py --resume-agent --cv-text "教育背景：……"         # 文本：直接读取（不联网）
    python main.py --resume-agent --cv-image assets/cv.png         # 图片：Qwen-VL 逐字转录（需 .env）
    python main.py --resume-agent --cv-file input/profile/xx.md --cv-trace   # 额外写出完整 Trace

    判定顺序是「先挡伪装，再判等级」：否定（「没做过 Docker 部署」）与背景（「旨在…」「计划学习…」）
    先被挡掉，剩下的条目才判 有结果 / 有动作 / 仅提及。输入的四个开关用 --cv-*，避开排版模式的 --resume-file。

Polish Agent（LangGraph）：基于「已核验事实 + JD 要求」给建议写法 / 追问预演 / 润色（JD 与简历各给一路）：
    python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md
    python main.py --polish-agent --jd-text "岗位职责：……" --cv-file input/profile/xx.md
    python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md --polish-trace

    规则先跑、模型后介入：工具先把「要求 ↔ 事实」钉死（结构化 / 否定背景 / 证据等级 / 能力词典匹配），
    模型只读这份简报写三块内容。没有文本层 key 时三块标「未启用」（简报照出，退出码仍为 0）；
    少给一路来源直接退出码 2，撑不起一份建议时退出码 3。

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
from .agents.agent import MAX_ROUNDS, build_text_budget, run_agent
from .domain.jd import load_jd, load_jd_positions
from .services.jd_source import SOURCE_URL, describe_sources, detect_sources, unusable_notes
from .tools import RESUME_FORMATS, RESUME_TOOL, resume_styles, tool_help_lines
from .core.schema import JobPosting
from .core.settings import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_FILE,
    EnvLoadResult,
    LLMSettings,
    StorageSettings,
    load_env,
    resolve_settings,
    resolve_storage_settings,
)
from .knowledge import KNOWLEDGE_TIERS, L1_STATIC, L2_SEMI_STATIC, L3_DYNAMIC, KnowledgeBase, build_knowledge_base
from .memory import MemoryManager, build_memory_manager
from .core.llm import DEFAULT_CALL_LIMIT, OpenAICompatClient, build_client, check_llm
from .agents.render import render_console, render_html, render_json, render_markdown

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
            "  python main.py --enable-memory --session-id demo\n"
            "  python main.py --index-only                         # 只增量建知识索引，然后退出\n"
            "  python main.py --enable-knowledge --knowledge-tiers L1,L2\n"
            "  python main.py --build-resume        # 简历排版：output/resume.md + output/resume.html\n"
            "  python main.py --build-resume --resume-style structure # 换风格 → resume-structure.*\n"
            "  python main.py --jd-agent --jd-url \"https://…\"   # JD Agent：JD（文本/图片/网址）-> Markdown\n"
            "  python main.py --resume-agent --cv-file input/profile/xx.md   # Resume Agent：简历 -> 事实条目 + 证据等级\n"
            "  python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md\n"
            "   # Polish Agent：JD + 简历 -> 建议写法 / 追问预演 / 润色\n"
            "\n可用工具（jd_agent/tools/）：\n"
            + "".join(f"  {line}\n" for line in tool_help_lines())
        ),
    )
    parser.add_argument("--jd", help="JD 文件（不指定时读 input/jd/ 下所有文件的全部岗位）")
    parser.add_argument("--jd-title", dest="jd_title", help="JD 文件里有多个岗位时，用标题关键字指定（如「岗位二」）")
    parser.add_argument(
        "--jd-agent",
        dest="jd_agent",
        action="store_true",
        help="JD Agent（LangGraph）：把 JD（文本 / 图片 / 网址）转成 Markdown，写出 output/jd.md",
    )
    parser.add_argument(
        "--jd-text",
        dest="jd_text",
        help="直接给的 JD 文本（多行就用引号包起来；整段是一个网址时自动按网址处理）",
    )
    parser.add_argument(
        "--jd-file",
        dest="jd_files",
        action="append",
        help="本地 JD 文件（md / txt / html / json / yaml，可重复）；走文件解析 Tool，不联网",
    )
    parser.add_argument(
        "--jd-image",
        dest="jd_images",
        action="append",
        help="本地 JD 截图（png / jpg…，可重复）；走 Qwen-VL 逐字转录，需要 .env 里的 key",
    )
    parser.add_argument(
        "--jd-url",
        dest="jd_urls",
        action="append",
        help="JD 网址（可重复）；每个地址发 1 次 HTTP GET，只抓你给的那个页面",
    )
    parser.add_argument("--jd-name", dest="jd_name", help="这份 JD 的标题（默认取原文里第一个标题）")
    parser.add_argument("--jd-out", dest="jd_out", help=f"输出的 Markdown 路径（默认 {OUTPUT_DIR / 'jd.md'}）")
    parser.add_argument(
        "--jd-trace",
        dest="jd_trace",
        action="store_true",
        help="额外写出带完整 Trace 的 Markdown（与输出同名，后缀 .trace.md）",
    )
    parser.add_argument(
        "--resume-agent",
        dest="resume_agent",
        action="store_true",
        help="Resume Agent（LangGraph）：把简历拆成事实条目并判证据等级，写出 output/resume_facts.md",
    )
    parser.add_argument(
        "--cv-text",
        dest="cv_text",
        help="直接给的简历文本（多行就用引号包起来；整段是一个网址时自动按网址处理）",
    )
    parser.add_argument(
        "--cv-file",
        dest="cv_files",
        action="append",
        help="本地简历文件（md / txt / html / json / yaml，可重复）；走文件解析 Tool，不联网",
    )
    parser.add_argument(
        "--cv-image",
        dest="cv_images",
        action="append",
        help="本地简历截图（png / jpg…，可重复）；走 Qwen-VL 逐字转录，需要 .env 里的 key",
    )
    parser.add_argument(
        "--cv-url",
        dest="cv_urls",
        action="append",
        help="在线简历网址（可重复）；每个地址发 1 次 HTTP GET，只抓你给的那个页面",
    )
    parser.add_argument("--cv-title", dest="cv_title", help="这份简历的标题（默认取原文里第一个标题）")
    parser.add_argument(
        "--cv-out", dest="cv_out", help=f"输出的 Markdown 路径（默认 {OUTPUT_DIR / 'resume_facts.md'}）"
    )
    parser.add_argument(
        "--cv-trace",
        dest="cv_trace",
        action="store_true",
        help="额外写出带完整 Trace 的 Markdown（与输出同名，后缀 .trace.md）",
    )
    parser.add_argument(
        "--polish-agent",
        dest="polish_agent",
        action="store_true",
        help=(
            "Polish Agent（LangGraph）：JD + 简历 -> 建议写法 / 追问预演 / 润色，"
            f"写出 {OUTPUT_DIR / 'polish.md'}"
        ),
    )
    parser.add_argument(
        "--polish-out", dest="polish_out", help=f"输出的 Markdown 路径（默认 {OUTPUT_DIR / 'polish.md'}）"
    )
    parser.add_argument(
        "--polish-trace",
        dest="polish_trace",
        action="store_true",
        help="额外写出带完整 Trace 的 Markdown（与输出同名，后缀 .trace.md）",
    )
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
        help="简历文件名（默认按风格取名：resume / resume-structure / resume-accent）",
    )
    parser.add_argument("--answer", help="回答 Agent 提出的问题；回答会作为新材料并入证据池后重新检索")
    parser.add_argument("--max-rounds", dest="max_rounds", type=int, default=MAX_ROUNDS, help=f"最多检索轮数（默认 {MAX_ROUNDS}）")
    parser.add_argument("--session-id", dest="session_id", default="default", help="记忆会话 ID（默认 default）")
    parser.add_argument(
        "--enable-memory",
        dest="enable_memory",
        action="store_true",
        help="启用 L0/L1/L2 记忆；L1 默认内存实现，L2 默认写入项目 data/ 下的 JSONL",
    )
    parser.add_argument(
        "--enable-knowledge",
        dest="enable_knowledge",
        action="store_true",
        help="启用 L1/L2/L3 知识库检索；未索引时先跑 --index-knowledge",
    )
    parser.add_argument(
        "--index-knowledge",
        dest="index_knowledge",
        action="store_true",
        help="把 knowledge/、input/、output/ 按 L1/L2/L3 层级增量写入知识索引",
    )
    parser.add_argument(
        "--index-only",
        dest="index_only",
        action="store_true",
        help="只建立 / 更新知识索引，不读取 JD，也不生成分析报告",
    )
    parser.add_argument(
        "--knowledge-tiers",
        dest="knowledge_tiers",
        default="L1,L2,L3",
        help="检索哪些知识层级，逗号分隔（L1 静态 / L2 半静态 / L3 动态；默认全部）",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default="",
        help=f"运行时数据目录（默认 {DEFAULT_DATA_DIR}，始终建议放在 D 盘项目内）",
    )
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


def _prepare_storage_settings(args) -> Tuple[StorageSettings, EnvLoadResult, str]:
    env_path = Path(args.env).expanduser() if args.env else DEFAULT_ENV_FILE
    if args.env and not env_path.is_file():
        return StorageSettings(), EnvLoadResult(path=str(env_path)), f"找不到 .env 文件：{env_path}"
    loaded = load_env(env_path)
    if loaded.error:
        return StorageSettings(), loaded, loaded.error
    settings = resolve_storage_settings(data_dir=args.data_dir or "")
    return settings, loaded, ""


def _parse_knowledge_tiers(raw: str) -> Tuple[str, ...]:
    aliases = {
        "l1": L1_STATIC,
        "static": L1_STATIC,
        "静态": L1_STATIC,
        "l2": L2_SEMI_STATIC,
        "semi-static": L2_SEMI_STATIC,
        "半静态": L2_SEMI_STATIC,
        "l3": L3_DYNAMIC,
        "dynamic": L3_DYNAMIC,
        "动态": L3_DYNAMIC,
    }
    values = [item.strip().casefold() for item in str(raw or "").split(",") if item.strip()]
    if not values or "all" in values or "全部" in values:
        return KNOWLEDGE_TIERS
    tiers: List[str] = []
    for value in values:
        tier = aliases.get(value)
        if tier is None:
            raise ValueError(f"未知知识层级：{value}（可选 L1 / L2 / L3）")
        if tier not in tiers:
            tiers.append(tier)
    return tuple(tiers)


def _build_context_stores(
    settings: StorageSettings,
    *,
    memory_enabled: bool,
    knowledge_enabled: bool,
) -> Tuple[Optional[MemoryManager], Optional[KnowledgeBase]]:
    memory = None
    knowledge = None
    if memory_enabled:
        memory = build_memory_manager(
            settings.data_dir,
            short_term_ttl_seconds=settings.l1_ttl_seconds,
            short_term_max_items=settings.l1_max_items,
            long_term_backend=settings.long_term_backend,
            mongodb_uri=settings.mongodb_uri,
            mongodb_database=settings.mongodb_database,
        )
    if knowledge_enabled:
        knowledge = build_knowledge_base(
            settings.data_dir,
            embedding_backend=settings.embedding_backend,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            index_backend=settings.index_backend,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            qdrant_collection=settings.qdrant_collection,
        )
    return memory, knowledge


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


def run_jd_agent_mode(args, vision_client: Optional[OpenAICompatClient] = None) -> int:
    """JD Agent 模式：把 JD（文本 / 图片 / 网址）转成 Markdown（LangGraph 编排）。

    真正的活交给 jd_agent.agents.jd_graph；这里只做三件事：挑来源、按需读 .env、打印 Trace 与结果。
    只有图片分支需要 key（Qwen-VL），文本 / 文件 / 网址三种来源都不读 .env。
    """
    try:                       # 只有这个模式需要 langgraph：没装也不影响其它模式
        from .agents.jd_graph import JDRequest, render_jd_console, render_jd_trace, run_jd_agent
    except ImportError:
        _say("[错误] --jd-agent 需要 langgraph：pip install langgraph（其它模式不受影响）")
        return EXIT_ERROR

    request = JDRequest(
        text=args.jd_text or "",
        files=tuple(args.jd_files or ()),
        images=tuple(args.jd_images or ()),
        urls=tuple(args.jd_urls or ()),
        title=args.jd_name or "",
    )
    if request.empty:
        _say("[错误] --jd-agent 至少要给一个 JD 来源：--jd-text / --jd-file / --jd-image / --jd-url")
        return EXIT_ERROR

    settings: Optional[LLMSettings] = None
    client = vision_client
    if request.images:
        settings, loaded, error = _prepare_settings(args)
        if error:
            _say(f"[错误] {error}")
            return EXIT_ERROR
        if client is None:
            client = build_client(settings, "vision")
        if not args.quiet:
            _say(f"[配置] .env：{loaded.summary()}")
            for line in settings.describe():
                _say(f"[配置] {line}")
            _say("")

    sources = detect_sources(
        text=request.text,
        files=request.files,
        images=request.images,
        urls=request.urls,
        vision_ready=client is not None,
    )
    usable = [source for source in sources if source.usable]
    if not usable:
        _say("[错误] 没有一个能读的 JD 来源：")
        for note in unusable_notes(sources):
            _say(f"  - {note}")
        return EXIT_ERROR

    out_path = Path(args.jd_out).expanduser() if args.jd_out else OUTPUT_DIR / "jd.md"
    if not args.quiet:
        _say(f"[输入] {len(usable)} 个可用来源：{describe_sources(usable)}")
        if any(source.kind == SOURCE_URL for source in usable):
            _say("[输入] 网址抓取：每个地址发 1 次 HTTP GET，只抓你给的那个页面")
        _say("")

    state = run_jd_agent(
        request,
        settings=settings,
        vision_client=client,
        sources=sources,
        out_dir=out_path.parent,
        stem=out_path.stem,
    )
    if not args.quiet:
        for line in render_jd_console(state):
            _say(line)
    if args.jd_trace:
        trace_path = out_path.with_suffix(".trace.md")
        trace_path.write_text(render_jd_trace(state), encoding="utf-8")
        if not args.quiet:
            _say(f"Trace 已写出：{trace_path}")
            _say("")
    return EXIT_ASK if state.needs_answer else EXIT_OK


def run_resume_agent_mode(args, vision_client: Optional[OpenAICompatClient] = None) -> int:
    """Resume Agent 模式：把简历拆成事实条目并判证据等级（LangGraph 编排）。

    真正的活交给 jd_agent.agents.resume_graph；这里只做三件事：挑来源、按需读 .env、打印 Trace 与结果。
    只有图片分支需要 key（Qwen-VL），文本 / 文件 / 网址三种来源都不读 .env。
    输入的四个开关用 --cv-* 前缀（CV = 简历），避开排版模式已经在用的 --resume-file。
    """
    try:                       # 只有这个模式需要 langgraph：没装也不影响其它模式
        from .agents.resume_graph import (
            ResumeRequest,
            render_resume_console,
            render_resume_trace,
            run_resume_agent,
        )
    except ImportError:
        _say("[错误] --resume-agent 需要 langgraph：pip install langgraph（其它模式不受影响）")
        return EXIT_ERROR

    request = ResumeRequest(
        text=args.cv_text or "",
        files=tuple(args.cv_files or ()),
        images=tuple(args.cv_images or ()),
        urls=tuple(args.cv_urls or ()),
        title=args.cv_title or "",
    )
    if request.empty:
        _say("[错误] --resume-agent 至少要给一个简历来源：--cv-text / --cv-file / --cv-image / --cv-url")
        return EXIT_ERROR

    settings: Optional[LLMSettings] = None
    client = vision_client
    if request.images:
        settings, loaded, error = _prepare_settings(args)
        if error:
            _say(f"[错误] {error}")
            return EXIT_ERROR
        if client is None:
            client = build_client(settings, "vision")
        if not args.quiet:
            _say(f"[配置] .env：{loaded.summary()}")
            for line in settings.describe():
                _say(f"[配置] {line}")
            _say("")

    sources = detect_sources(
        text=request.text,
        files=request.files,
        images=request.images,
        urls=request.urls,
        vision_ready=client is not None,
    )
    usable = [source for source in sources if source.usable]
    if not usable:
        _say("[错误] 没有一个能读的简历来源：")
        for note in unusable_notes(sources):
            _say(f"  - {note}")
        return EXIT_ERROR

    out_path = Path(args.cv_out).expanduser() if args.cv_out else OUTPUT_DIR / "resume_facts.md"
    if not args.quiet:
        _say(f"[输入] {len(usable)} 个可用来源：{describe_sources(usable)}")
        _say("[输入] 判定顺序：先挡「写了但没做过」（否定 / 背景），再判证据等级")
        _say("")

    state = run_resume_agent(
        request,
        settings=settings,
        vision_client=client,
        sources=sources,
        out_dir=out_path.parent,
        stem=out_path.stem,
    )
    if not args.quiet:
        for line in render_resume_console(state):
            _say(line)
    if args.cv_trace:
        trace_path = out_path.with_suffix(".trace.md")
        trace_path.write_text(render_resume_trace(state), encoding="utf-8")
        if not args.quiet:
            _say(f"Trace 已写出：{trace_path}")
            _say("")
    return EXIT_ASK if state.needs_answer else EXIT_OK


def run_polish_agent_mode(
    args,
    text_client: Optional[OpenAICompatClient] = None,
    vision_client: Optional[OpenAICompatClient] = None,
) -> int:
    """Polish Agent 模式：基于「已核验事实 + JD 要求」给出建议写法 / 追问预演 / 润色（LangGraph 编排）。

    真正的活交给 jd_agent.agents.polish_graph；这里只做三件事：挑两路来源、读 .env、打印 Trace 与结果。
    与 JD / Resume Agent 的一处区别：这两路必须同时给（一份 JD + 一份简历），所以一律读 .env ——
    三块内容要文本层 key；给了图片还要多模态 key。没有文本层 key 时三块统一标「未启用」，简报照出。
    """
    try:                       # 只有这个模式需要 langgraph：没装也不影响其它模式
        from .agents.polish_graph import (
            PolishRequest,
            render_polish_console,
            render_polish_trace,
            run_polish_agent,
        )
    except ImportError:
        _say("[错误] --polish-agent 需要 langgraph：pip install langgraph（其它模式不受影响）")
        return EXIT_ERROR

    request = PolishRequest(
        jd_text=args.jd_text or "",
        jd_files=tuple(args.jd_files or ()),
        jd_images=tuple(args.jd_images or ()),
        jd_urls=tuple(args.jd_urls or ()),
        jd_title=args.jd_name or "",
        cv_text=args.cv_text or "",
        cv_files=tuple(args.cv_files or ()),
        cv_images=tuple(args.cv_images or ()),
        cv_urls=tuple(args.cv_urls or ()),
        cv_title=args.cv_title or "",
    )
    missing = request.missing_sides
    if missing:                # 只有一路就没有对照：光有 JD 只能得到要求清单，光有简历只能得到事实清单
        _say(f"[错误] --polish-agent 要同时给 JD 与简历，现在缺：{'、'.join(missing)}")
        _say("       JD 用 --jd-text / --jd-file / --jd-image / --jd-url；简历用 --cv-* 那一组")
        return EXIT_ERROR

    settings, loaded, error = _prepare_settings(args)   # 三块内容要用文本层 key，所以这一模式一律读 .env
    if error:
        _say(f"[错误] {error}")
        return EXIT_ERROR
    client = vision_client
    if (request.jd_images or request.cv_images) and client is None:
        client = build_client(settings, "vision")
    if not args.quiet:
        _say(f"[配置] .env：{loaded.summary()}")
        for line in settings.describe():
            _say(f"[配置] {line}")
        _say("")

    vision_ready = client is not None or bool(getattr(settings, "vision_ready", False))
    jd_sources = detect_sources(
        text=request.jd_text,
        files=request.jd_files,
        images=request.jd_images,
        urls=request.jd_urls,
        vision_ready=vision_ready,
    )
    cv_sources = detect_sources(
        text=request.cv_text,
        files=request.cv_files,
        images=request.cv_images,
        urls=request.cv_urls,
        vision_ready=vision_ready,
    )
    usable_jd = [source for source in jd_sources if source.usable]
    usable_cv = [source for source in cv_sources if source.usable]
    if not usable_jd or not usable_cv:      # 有一路一条都读不出来：这不是「问一句」能解决的，当输入错误
        _say("[错误] 至少有一路没有能读的来源：")
        skipped = [s for s in jd_sources + cv_sources if not s.usable]
        for note in unusable_notes(skipped):
            _say(f"  - {note}")
        return EXIT_ERROR

    out_path = Path(args.polish_out).expanduser() if args.polish_out else OUTPUT_DIR / "polish.md"
    if not args.quiet:
        _say(f"[输入] JD 侧 {len(usable_jd)} 个可用来源：{describe_sources(usable_jd)}")
        _say(f"[输入] 简历侧 {len(usable_cv)} 个可用来源：{describe_sources(usable_cv)}")
        _say("[输入] 顺序：先挡「写了没做过」（否定 / 背景），再判证据等级，最后才让模型写")
        _say("")

    state = run_polish_agent(
        request,
        settings=settings,
        text_client=text_client,
        vision_client=client,
        jd_sources=jd_sources,
        cv_sources=cv_sources,
        out_dir=out_path.parent,
        stem=out_path.stem,
        llm_max_calls=max(1, args.llm_max_calls),
    )
    if not args.quiet:
        for line in render_polish_console(state):
            _say(line)
    if args.polish_trace:
        trace_path = out_path.with_suffix(".trace.md")
        trace_path.write_text(render_polish_trace(state), encoding="utf-8")
        if not args.quiet:
            _say(f"Trace 已写出：{trace_path}")
            _say("")
    return EXIT_ASK if state.needs_answer else EXIT_OK


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
    use_memory = bool(args.enable_memory)
    use_knowledge = bool(args.enable_knowledge or args.index_knowledge or args.index_only)
    settings: Optional[LLMSettings] = None

    if args.build_resume:
        return run_build_resume(args)

    if args.jd_agent:
        return run_jd_agent_mode(args, vision_client=vision_client)

    if args.resume_agent:
        return run_resume_agent_mode(args, vision_client=vision_client)

    if args.polish_agent:
        return run_polish_agent_mode(args, text_client=text_client, vision_client=vision_client)

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

    memory_manager: Optional[MemoryManager] = None
    knowledge_base: Optional[KnowledgeBase] = None
    knowledge_tiers: Tuple[str, ...] = ()
    if use_memory or use_knowledge:
        try:
            storage_settings, storage_loaded, storage_error = _prepare_storage_settings(args)
            if storage_error:
                _say(f"[错误] {storage_error}")
                return EXIT_ERROR
            knowledge_tiers = _parse_knowledge_tiers(args.knowledge_tiers)
            memory_manager, knowledge_base = _build_context_stores(
                storage_settings,
                memory_enabled=use_memory,
                knowledge_enabled=use_knowledge,
            )
        except (ImportError, ValueError) as exc:
            _say(f"[错误] 记忆 / 知识库初始化失败：{exc}")
            return EXIT_ERROR
        if not args.quiet:
            _say(f"[配置] .env：{storage_loaded.summary()}")
            for line in storage_settings.describe():
                _say(f"[配置] {line}")
            if knowledge_base is not None:
                stats = knowledge_base.stats()
                for line in stats.lines():
                    _say(f"[知识库] {line}")
            _say("")
        if (args.index_knowledge or args.index_only) and knowledge_base is not None:
            try:
                changed = knowledge_base.ensure_default_index(PROJECT_ROOT)
            except Exception as exc:
                _say(f"[错误] 知识索引失败：{exc.__class__.__name__}: {exc}")
                return EXIT_ERROR
            if not args.quiet:
                _say(f"[知识库] 增量索引完成：写入 / 更新 {changed} 个切片")
                _say("")

        if args.index_only:
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
            memory=memory_manager,
            knowledge=knowledge_base,
            session_id=args.session_id or "default",
            knowledge_tiers=knowledge_tiers,
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
