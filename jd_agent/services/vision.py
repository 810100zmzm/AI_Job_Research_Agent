"""多模态层（可选）：把项目描述里引用的本地图片解析成文字事实。

边界写死在这里：
  * 模型只回答「图里有什么」，它自述的证据等级（哪怕写了 result）一律丢弃；
  * 每条事实都用 project.make_fact() 建 ProjectFact，等级由 classify() 规则判定；
  * source_file 记成「图片:相对路径」，报告里能回到具体是哪张图。
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ..domain.jd import PROJECT_ROOT, display_path
from ..core.llm import LLMError, OpenAICompatClient, parse_json_reply
from ..domain.project import make_fact
from ..core.schema import ImageRef, Project, VisionReport

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}

IMAGE_SECTION = "图片解析"   # 中性章节名，避免被 classify() 当成背景段落降级
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_FACTS_PER_IMAGE = 6
MAX_TOTAL_FACTS = 18
MAX_FACT_CHARS = 120

MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+)")
HTML_IMAGE_RE = re.compile(r"<img[^>]*?src=[\"'](?P<path>[^\"']+)[\"']", re.IGNORECASE)
BARE_IMAGE_RE = re.compile(
    r"(?P<path>[^\s`\"'<>()\[\]（）【】，,；;]+\.(?:png|jpe?g|webp|bmp|gif))", re.IGNORECASE
)

VISION_SYSTEM_PROMPT = (
    "你是严谨的图片取证助手。只描述图里能直接看到的东西：图表类型与坐标轴、数字、表格单元格、"
    "界面元素、代码或报错信息。不要推测，不要评价，不要给建议，也不要判断这张图对简历有没有用。\n"
    "输出严格 JSON：{\"facts\": [\"...\"]}；每条一句话，最多 6 条，每条不超过 60 字。"
)
VISION_USER_PROMPT = (
    "这张图片来自项目描述里的引用 {ref}（项目：{project}）。"
    "请把图中客观、可核验的事实写成条目。"
)


# ---- 1. 找出图片引用 --------------------------------------------------------

def _iter_references(text: str) -> Iterator[Tuple[int, str]]:
    """逐行找出图片引用：markdown / html / 裸文件名 三种写法都认。"""
    for line_no, raw in enumerate(str(text or "").splitlines(), start=1):
        seen: Set[str] = set()
        for pattern in (MARKDOWN_IMAGE_RE, HTML_IMAGE_RE, BARE_IMAGE_RE):
            for match in pattern.finditer(raw):
                value = match.group("path").strip()
                if value and value not in seen:
                    seen.add(value)
                    yield line_no, value


def _resolve(candidate: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    """把引用解析成本地文件；远程地址、非图片后缀、不存在的路径都返回 None。"""
    candidate = str(candidate or "").strip()
    if not candidate or "://" in candidate or candidate.startswith("data:"):
        return None
    path = Path(candidate)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    tries = [path] if path.is_absolute() else [base / path for base in search_dirs]
    for item in tries:
        try:
            if item.is_file():
                return item.resolve()
        except OSError:
            continue
    return None


def _ref_of(path: Path) -> str:
    return f"图片:{display_path(path)}"


def collect_images(
    text: str, project_path, extra_images: Sequence[str] = ()
) -> Tuple[List[ImageRef], List[str]]:
    """找出项目描述里引用的本地图片 + --image 显式传入的图片，返回 (图片清单, 备注)。"""
    path = Path(str(project_path))
    search_dirs = [path.parent, PROJECT_ROOT, Path.cwd()]
    refs: List[ImageRef] = []
    notes: List[str] = []
    seen: Set[str] = set()

    for line_no, candidate in _iter_references(text):
        resolved = _resolve(candidate, search_dirs)
        if resolved is None or str(resolved) in seen:
            continue
        seen.add(str(resolved))
        refs.append(ImageRef(raw=candidate, path=str(resolved), ref=_ref_of(resolved), line_no=line_no))

    for candidate in extra_images:
        resolved = _resolve(candidate, [Path.cwd(), path.parent, PROJECT_ROOT])
        if resolved is None:
            notes.append(f"--image 指定的图片不可用（不存在或不是支持的图片）：{candidate}")
            continue
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        refs.append(ImageRef(raw=candidate, path=str(resolved), ref=_ref_of(resolved), line_no=0))
    return refs, notes


# ---- 2. 调模型读图 ----------------------------------------------------------

def data_url(path: Path) -> str:
    """把图片读成 data URL（JD Agent 的图片分支复用同一套大小 / MIME 校验）。"""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"图片读不出来（{exc}）") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"图片过大（{len(data) / 1048576:.1f} MB > {MAX_IMAGE_BYTES // 1048576} MB）")
    mime = MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def build_vision_messages(ref: ImageRef, project_name: str) -> List[Dict[str, Any]]:
    """拼一条多模态消息（OpenAI 兼容的 content 数组格式）。"""
    return [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": VISION_USER_PROMPT.format(ref=ref.ref, project=project_name or "未命名项目"),
                },
                {"type": "image_url", "image_url": {"url": data_url(Path(ref.path))}},
            ],
        },
    ]


def parse_image_facts(reply: str) -> List[str]:
    """把模型返回解析成事实条目；解析不出来就返回空列表。"""
    payload = parse_json_reply(reply)
    items = payload.get("facts") if isinstance(payload, dict) else payload
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return _from_lines(reply)
    facts: List[str] = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("text", "")   # 模型自述的 level / evidence 一律不看
        text = " ".join(str(item or "").split())
        if text:
            facts.append(text[:MAX_FACT_CHARS])
    return facts or _from_lines(reply)


def _from_lines(reply: str) -> List[str]:
    """模型没按 JSON 输出时的兜底：按行取，去掉项目符号与编号。"""
    facts: List[str] = []
    for raw in str(reply or "").splitlines():
        line = re.sub(r"^\s*(?:[-*+•·]|\d{1,2}\s*[.、)])\s*", "", raw).strip()
        if len(line) > 1:
            facts.append(line[:MAX_FACT_CHARS])
    return facts


def enrich_project(
    project: Project, refs: Sequence[ImageRef], client: OpenAICompatClient, model: str
) -> VisionReport:
    """逐张解析图片，把事实并入项目证据池；失败只记 note，不抛异常。"""
    report = VisionReport(enabled=True, model=model, images=list(refs))
    seen: Set[str] = set()
    for ref in refs:
        if report.facts_added >= MAX_TOTAL_FACTS:
            report.notes.append(f"已达事实总量上限（{MAX_TOTAL_FACTS} 条），其余图片没有解析")
            break
        try:
            messages = build_vision_messages(ref, project.name)
        except ValueError as exc:
            report.notes.append(f"{ref.ref}：{exc}")
            continue
        try:
            reply = client.chat(messages, model=model, temperature=0)
        except LLMError as exc:
            report.error = f"{ref.ref}：{exc}"      # 一次失败只记一条 note，然后停手
            report.notes.append(f"{ref.ref}：调用失败（{exc}）")
            break
        except Exception as exc:  # 兜底：可选层不能带崩主流程
            report.error = f"{ref.ref}：{exc.__class__.__name__}: {exc}"
            report.notes.append(report.error)
            break

        facts = parse_image_facts(reply)
        if not facts:
            report.notes.append(f"{ref.ref}：没有解析出可用事实")
            continue
        added = _append_facts(project, ref, facts, seen, MAX_TOTAL_FACTS - report.facts_added)
        report.facts_added += added
        if added:
            report.parsed += 1
        else:
            report.notes.append(f"{ref.ref}：解析结果与已有事实重复")
    return report


def _append_facts(
    project: Project, ref: ImageRef, facts: Sequence[str], seen: Set[str], budget: int
) -> int:
    """把图片事实交给 make_fact 建条目 —— 等级始终由 classify() 判定。"""
    added = 0
    for text in list(facts)[:MAX_FACTS_PER_IMAGE]:
        if added >= budget or text in seen:
            continue
        seen.add(text)
        project.facts.append(make_fact(text, IMAGE_SECTION, 0, ref.ref))
        added += 1
    return added
