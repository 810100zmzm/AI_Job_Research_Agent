"""JD 来源层：把「文本 / 文件 / 图片 / 网址」四种输入取成 Markdown 原文。

四种取法（对应 JD Agent 的三种输入类型）：
    text   直接读取  —— 不联网、不调模型；整段就是一个网址时自动改走 url 分支
    file   文件解析  —— 交回工具区的文件解析 Tool（纯规则）
    image  图片转录  —— Qwen-VL 逐字转成 Markdown（联网，需 DASHSCOPE_API_KEY / QWEN_API_KEY）
    url    网址解析  —— 抓一次 HTTP，HTML -> Markdown（联网，只抓你给的那个页面）

纪律：不给网址就一个 HTTP 都不发，不给图片就一次模型都不调；任何失败抛 SourceError，
由 Agent 记成「这个来源没取到」并继续跑其它来源，而不是带崩整条链路。
"""
from __future__ import annotations

import gzip
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..domain.jd import display_path
from ..core.llm import LLMError, OpenAICompatClient
from .vision import data_url

# ---- 来源类型（Agent 的状态与 Trace 里都用这几个常量）------------------------
SOURCE_TEXT = "text"
SOURCE_FILE = "file"
SOURCE_IMAGE = "image"
SOURCE_URL = "url"

SOURCE_LABEL = {
    SOURCE_TEXT: "文本",
    SOURCE_FILE: "文件",
    SOURCE_IMAGE: "图片",
    SOURCE_URL: "网址",
}

SOURCE_UNKNOWN = "unknown"      # 后缀不认识：既不是文本类，也不是图片

TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".text", ".html", ".htm", ".xhtml", ".json", ".yaml", ".yml")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

# ---- 抓网页的纪律（常量，可直接检查）----------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 JD-Agent/1.0"
)
URL_TIMEOUT = 20.0                       # 单次抓取超时（秒）
MAX_PAGE_BYTES = 3 * 1024 * 1024         # 超过就截断，不为了一个页面把内存吃满
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
HOST_RE = re.compile(r"^[\w.\-]+\.[a-zA-Z]{2,}(?::\d+)?(?:[/?#]|$)")
META_CHARSET_RE = re.compile(rb"<meta[^>]+charset=[\"']?\s*(?P<charset>[\w-]+)", re.IGNORECASE)
CHARSET_RE = re.compile(r"charset=[\"']?\s*(?P<charset>[\w-]+)", re.IGNORECASE)


class SourceError(RuntimeError):
    """一个来源取不到内容（网络 / 页面格式 / 模型调用 / 文件读不出来）。"""


@dataclass(frozen=True)
class JDSource:
    """一个待处理的 JD 来源。note 非空表示这个来源不可用（原因写在这里）。"""

    kind: str
    raw: str            # 文本内容 / 文件路径 / 图片路径 / 网址
    ref: str            # 报告与 Trace 里的引用，例如「网址:https://x/y」
    label: str          # 给人看的短说明（文件名 / 域名 / 「粘贴的文本」）
    note: str = ""

    @property
    def usable(self) -> bool:
        return not self.note

    @property
    def kind_label(self) -> str:
        return SOURCE_LABEL.get(self.kind, self.kind)


# ---- 1. 识别输入类型 --------------------------------------------------------

def looks_like_url(value: str) -> bool:
    """整段就是一个网址（用来判断「粘贴的文本」其实是网址）。"""
    text = str(value or "").strip()
    if not text or " " in text or "\n" in text or len(text) > 500:
        return False
    return bool(URL_RE.match(text))


def normalize_url(value: str) -> str:
    """补全协议头并校验；不是一个能抓的网址就返回空串。"""
    text = str(value or "").strip()
    if not text or " " in text or "\n" in text:
        return ""
    if text.startswith("//"):
        text = "https:" + text
    if not SCHEME_RE.match(text):
        if not HOST_RE.match(text):
            return ""
        text = "https://" + text
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    return text


def classify_path(path) -> str:
    """按后缀判断这是哪一类来源：image / file / unknown。"""
    suffix = Path(str(path)).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return SOURCE_IMAGE
    if suffix in TEXT_SUFFIXES:
        return SOURCE_FILE
    return SOURCE_UNKNOWN


def _file_source(path) -> JDSource:
    """文本类文件的来源（图片走 _image_source，别在这里处理）。"""
    candidate = Path(str(path)).expanduser()
    label = candidate.name or str(candidate)
    ref = f"文件:{display_path(candidate)}"
    if not candidate.is_file():
        return JDSource(SOURCE_FILE, str(candidate), ref, label, note=f"找不到文件：{candidate}")
    if classify_path(candidate) == SOURCE_UNKNOWN:
        return JDSource(
            SOURCE_FILE,
            str(candidate),
            ref,
            label,
            note=f"不认识这个后缀（{candidate.suffix or '无后缀'}）：支持 md / txt / html / json / yaml 或图片",
        )
    return JDSource(SOURCE_FILE, str(candidate), ref, label)


def _image_source(path, vision_ready: bool) -> JDSource:
    candidate = Path(str(path)).expanduser()
    label = candidate.name or str(candidate)
    ref = f"图片:{display_path(candidate)}"
    if not candidate.is_file():
        return JDSource(SOURCE_IMAGE, str(candidate), ref, label, note=f"找不到图片：{candidate}")
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return JDSource(SOURCE_IMAGE, str(candidate), ref, label, note=f"不是支持的图片格式：{candidate.suffix}")
    if not vision_ready:
        return JDSource(
            SOURCE_IMAGE,
            str(candidate),
            ref,
            label,
            note="未配置 DASHSCOPE_API_KEY / QWEN_API_KEY（图片分支跳过）",
        )
    return JDSource(SOURCE_IMAGE, str(candidate), ref, label)


def _url_source(value) -> JDSource:
    normalized = normalize_url(value)
    if not normalized:
        return JDSource(SOURCE_URL, str(value), f"网址:{value}", str(value), note="不是一个能抓取的网址")
    host = urllib.parse.urlsplit(normalized).netloc
    return JDSource(SOURCE_URL, normalized, f"网址:{normalized}", host)


def detect_sources(
    text: str = "",
    files: Sequence = (),
    images: Sequence = (),
    urls: Sequence = (),
    vision_ready: bool = True,
) -> List[JDSource]:
    """识别输入类型并逐个校验：文本 / 文件 / 图片 / 网址，顺序固定，便于 Trace 阅读。"""
    sources: List[JDSource] = []
    pasted = str(text or "").strip()
    if pasted:
        if looks_like_url(pasted):          # 粘贴的总不能是网址：自动改走网址分支
            sources.append(_url_source(pasted))
        else:
            sources.append(
                JDSource(
                    SOURCE_TEXT,
                    pasted,
                    "文本",
                    "粘贴的文本",
                )
            )
    for item in files:
        # --jd-file 指到一张截图上，就按图片走（该需要 key 的照样需要 key）
        if classify_path(item) == SOURCE_IMAGE:
            sources.append(_image_source(item, vision_ready))
        else:
            sources.append(_file_source(item))
    for item in images:
        sources.append(_image_source(item, vision_ready))
    for item in urls:
        sources.append(_url_source(item))
    return sources


def describe_sources(sources: Sequence[JDSource]) -> str:
    """「文本 1、网址 2」这样的来源统计，给 Trace 的 Observation 用。"""
    counts: Dict[str, int] = {}
    for source in sources:
        counts[source.kind] = counts.get(source.kind, 0) + 1
    order = [SOURCE_TEXT, SOURCE_FILE, SOURCE_IMAGE, SOURCE_URL]
    return "、".join(f"{SOURCE_LABEL[kind]} {counts[kind]}" for kind in order if counts.get(kind))


def unusable_notes(sources: Sequence[JDSource]) -> List[str]:
    """不可用来源的原因，一条一行。"""
    return [f"{source.ref}：{source.note}" for source in sources if not source.usable]


# ---- 2. 抓网页（联网）-------------------------------------------------------

Opener = Callable[[str, float], Tuple[bytes, str, str]]


def _charset_of(content_type: str) -> str:
    match = CHARSET_RE.search(str(content_type or ""))
    return match.group("charset") if match else ""


def decode_body(body: bytes, charset: str = "") -> str:
    """按 charset -> utf-8 -> gbk -> big5 -> latin-1 的顺序解码，尽量不丢字。"""
    guesses = ([charset] if charset else []) + ["utf-8", "utf-8-sig", "gbk", "big5", "latin-1"]
    for encoding in guesses:
        try:
            text = body.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if encoding == "latin-1":           # 兜底：再从 <meta charset> 里碰一次运气
            sniffed = META_CHARSET_RE.search(body[:4096])
            if sniffed:
                try:
                    return body.decode(sniffed.group("charset").decode("ascii", "ignore"))
                except (UnicodeDecodeError, LookupError):
                    return text
        return text
    return body.decode("utf-8", errors="replace")


def _inflate(body: bytes) -> bytes:
    try:
        return zlib.decompress(body)
    except zlib.error:
        return zlib.decompress(body, -zlib.MAX_WBITS)


def _default_opener(url: str, timeout: float) -> Tuple[bytes, str, str]:
    """默认抓取器：标准库 urllib，带浏览器 UA、gzip 解压与大小截断。"""
    request = urllib.request.Request(url, headers=dict(REQUEST_HEADERS))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_PAGE_BYTES + 1)
            charset = _charset_of(response.headers.get("Content-Type", ""))
            encoding = str(response.headers.get("Content-Encoding") or "").lower()
            final_url = str(response.geturl() or url)
    except urllib.error.HTTPError as exc:
        raise SourceError(f"HTTP {exc.code}（{exc.reason}）") from exc
    except urllib.error.URLError as exc:
        raise SourceError(f"网络不可达：{exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise SourceError(f"请求超时（{timeout:g}s）") from exc
    except OSError as exc:
        raise SourceError(f"{exc.__class__.__name__}: {exc}") from exc
    if len(raw) > MAX_PAGE_BYTES:
        raw = raw[:MAX_PAGE_BYTES]
    if "gzip" in encoding:
        try:
            raw = gzip.decompress(raw)
        except OSError as exc:
            raise SourceError(f"页面解压失败：{exc}") from exc
    elif "deflate" in encoding:
        try:
            raw = _inflate(raw)
        except zlib.error as exc:
            raise SourceError(f"页面解压失败：{exc}") from exc
    return raw, charset, final_url


def fetch_html(url: str, timeout: float = URL_TIMEOUT, opener: Opener = None) -> Tuple[str, str]:
    """抓一次页面，返回 (HTML 文本, 最终网址)；失败抛 SourceError。

    opener 可以注入替身（测试用），签名是 (url, timeout) -> (body 字节, charset, 最终网址)。
    """
    fetcher: Opener = opener or _default_opener
    try:
        body, charset, final_url = fetcher(url, timeout)
    except SourceError:
        raise
    except Exception as exc:                # 注入的替身抛别的异常也照样降级
        raise SourceError(f"{exc.__class__.__name__}: {exc}") from exc
    if not body:
        raise SourceError("页面返回是空的")
    return decode_body(body, charset), final_url or url


# ---- 3. 图片转 Markdown（联网）----------------------------------------------

JD_IMAGE_SYSTEM_PROMPT = (
    "你是严谨的招聘信息转录助手。把图片里的 JD 逐字转写成 Markdown：\n"
    "1. 原文有章节标题（例如「岗位职责」「任职要求」「加分项」）就按原层级写成 ## / ###；\n"
    "2. 每条要求 / 职责单独一行，写成 `- ` 开头的列表；\n"
    "3. 表格转成 Markdown 表格，保留表头；\n"
    "4. 只转录图里能看清的字，不翻译、不总结、不补内容、不改顺序；看不清的字写 [看不清]；\n"
    "5. 只输出 Markdown 正文，不要解释，不要代码块围栏。"
)
JD_IMAGE_USER_PROMPT = "这是 JD 的第 {index} 张图片（来源：{ref}）。请把图中的招聘信息逐字转写成 Markdown。"


def parse_image_markdown(reply: str) -> str:
    """去掉模型爱加的代码块围栏与前后闲话，只留 Markdown 正文。"""
    text = str(reply or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def build_image_messages(
    path,
    ref: str,
    index: int = 1,
    system_prompt: str = JD_IMAGE_SYSTEM_PROMPT,
    user_prompt: str = JD_IMAGE_USER_PROMPT,
) -> List[Dict[str, Any]]:
    """拼一条多模态消息（OpenAI 兼容的 content 数组格式）。

    prompt 可以换：Resume Agent 转的是简历截图，用的就是自己那份提示词（纪律一样：逐字转录）。
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt.format(index=index, ref=ref)},
                {"type": "image_url", "image_url": {"url": data_url(Path(path))}},
            ],
        },
    ]


def image_to_markdown(
    path,
    client: OpenAICompatClient,
    model: str,
    ref: str = "图片",
    index: int = 1,
    system_prompt: str = JD_IMAGE_SYSTEM_PROMPT,
    user_prompt: str = JD_IMAGE_USER_PROMPT,
) -> str:
    """调 Qwen-VL 把一张截图逐字转成 Markdown；失败抛 SourceError。"""
    try:
        messages = build_image_messages(
            path, ref, index, system_prompt=system_prompt, user_prompt=user_prompt
        )
    except ValueError as exc:               # 图不存在 / 过大 / 读不出来
        raise SourceError(str(exc)) from exc
    try:
        reply = client.chat(messages, model=model, temperature=0)
    except LLMError as exc:
        raise SourceError(str(exc)) from exc
    except Exception as exc:                # 注入的替身抛别的异常也照样降级
        raise SourceError(f"{exc.__class__.__name__}: {exc}") from exc
    markdown = parse_image_markdown(reply)
    if not markdown:
        raise SourceError("模型没有返回可用的 Markdown")
    return markdown
