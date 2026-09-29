"""HTML -> Markdown：纯规则、只用标准库。

为什么自己写：这个项目只依赖 python-dotenv，抓回来的 JD 页面必须离线可解析，
所以这里用标准库 `html.parser` 做一次「保留结构」的转换：

    <h3>任职要求</h3>                    ->  ### 任职要求
    <li>熟练使用 Python</li>             ->  - 熟练使用 Python
    <div>两行文字</div>                  ->  两行文字（块与块之间空一行）
    <tr><td>地点</td><td>北京</td></tr>  ->  | 地点 | 北京 |

两个口径（对应 Agent Loop 里的 Continue / Adjust）：
  * 严格 strict=True：再丢掉 UI 噪声行（导航 / 按钮 / 备案号 / 重复行），只留像 JD 的正文；
  * 放宽 strict=False：只去空行与重复行，页面上出现过的文字都留下（严格口径剩下太少时用）。
两个口径都不改字：不翻译、不总结、不补内容；丢了多少行会如实报出来。
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Sequence, Tuple

# ---- 标签分类（都写在这里，方便直接检查）------------------------------------
SKIP_TAGS = frozenset({
    "script", "style", "noscript", "head", "meta", "link", "svg", "template",
    "iframe", "canvas", "video", "audio", "select", "option", "button", "form",
})
BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "main", "header", "footer", "aside", "nav",
    "dl", "dt", "dd", "blockquote", "pre", "figure", "figcaption", "table",
    "thead", "tbody", "tfoot", "ul", "ol", "label", "fieldset", "output", "address",
    "html", "body",
})
ROW_TAGS = ("td", "th")
# 自闭合（void）标签：写出来就没有内容，也永远等不到 </xxx>，不能算进 skip 深度
VOID_TAGS = frozenset({
    "meta", "link", "br", "hr", "img", "input", "source", "track", "wbr",
    "area", "base", "col", "embed", "param",
})
HEADING_RE = re.compile(r"^h([1-6])$")

# ---- 噪声词表：严格口径下命中即丢（都是招聘页面上的 UI 文字，不是 JD 内容）------
NOISE_WORDS = (
    "登录", "注册", "登入", "退出", "首页", "返回", "回到顶部", "返回顶部",
    "收藏", "分享", "举报", "投诉", "在线客服", "意见反馈", "帮助中心",
    "立即申请", "立即投递", "马上投递", "投递简历", "申请职位", "已投递",
    "上一篇", "下一篇", "相关推荐", "热门职位", "猜你喜欢", "相似职位", "更多职位",
    "下载APP", "下载客户端", "扫码", "关注我们", "微信公众号", "手机版", "电脑版",
    "版权所有", "备案", "京ICP", "沪ICP", "粤ICP", "增值电信业务", "营业执照",
    "隐私政策", "用户协议", "服务条款", "Cookie",
    "公司主页", "工商信息", "融资阶段", "企业规模",
)
NOISE_RE = re.compile("|".join(re.escape(word) for word in NOISE_WORDS), re.IGNORECASE)
NOISE_MAX_CHARS = 20          # 只对短行用词表过滤：长句里出现「登录」可能真是 JD 内容
STRUCT_PREFIX_RE = re.compile(r"^\s*(?:[-*+•·▪]|\d{1,2}\s*[.、)]|#{1,6}\s|\|)")
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
BLANK_BEFORE = frozenset({"heading", "para", "rule"})


def _collapse(parts: Sequence[str]) -> str:
    """把一串文本片段合成一个块：行内空白归一，块内换行保留。"""
    text = "".join(parts).replace("\u00a0", " ").replace("\u3000", " ")
    lines = [" ".join(item.split()) for item in text.splitlines()]
    return "\n".join(item for item in lines if item).strip()


class _MarkdownParser(HTMLParser):
    """把 HTML 收成一串块：(kind, level, text)，kind ∈ heading / list / para / row / rule。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[Tuple[str, int, str]] = []
        self._skip = 0
        self._kind = ""
        self._level = 0
        self._buf: List[str] = []
        self._cells: List[str] = []

    # -- 缓冲区 --
    def _flush(self) -> None:
        text = _collapse(self._buf)
        if text:
            self.blocks.append((self._kind or "para", self._level, text))
        self._buf = []
        self._kind = ""
        self._level = 0

    def _flush_cell(self) -> None:
        text = _collapse(self._buf)
        self._buf = []
        if text:
            self._cells.append(text)

    # -- HTMLParser 回调 --
    def handle_starttag(self, tag, attrs) -> None:
        if tag in SKIP_TAGS:
            if tag not in VOID_TAGS:      # void 标签没有内容、也没有对应的 </xxx>
                self._skip += 1
            return
        if self._skip:
            return
        if tag == "br":
            self._buf.append("\n")
            return
        if tag == "hr":
            self._flush()
            self.blocks.append(("rule", 0, ""))
            return
        if tag == "tr":
            self._flush()
            self._cells = []
            return
        if tag in ROW_TAGS:
            self._flush_cell()
            return
        heading = HEADING_RE.match(tag)
        if heading:
            self._flush()
            self._kind = "heading"
            self._level = int(heading.group(1))
            return
        if tag == "li":
            self._flush()
            self._kind = "list"
            return
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_startendtag(self, tag, attrs) -> None:
        if tag in SKIP_TAGS:
            return                        # <link /> 这类本来就是空的，不用记 skip
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag) -> None:
        if tag in SKIP_TAGS:
            if tag not in VOID_TAGS:
                self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "tr":
            self._flush_cell()
            if self._cells:
                self.blocks.append(("row", 0, " | ".join(self._cells)))
            self._cells = []
            return
        if tag in ROW_TAGS:
            self._flush_cell()
            return
        if tag in ("br", "hr"):
            return
        if tag == "li" or tag in BLOCK_TAGS or HEADING_RE.match(tag):
            self._flush()

    def handle_data(self, data) -> None:
        if self._skip:
            return
        self._buf.append(data)


def html_to_blocks(html_text: str) -> List[Tuple[str, int, str]]:
    """HTML -> 块列表（结构保留，不做任何删减）。"""
    parser = _MarkdownParser()
    parser.feed(str(html_text or ""))
    parser.close()
    return parser.blocks


def blocks_to_markdown(blocks: Sequence[Tuple[str, int, str]]) -> str:
    """块列表 -> Markdown：标题带 #，列表项带 -，表格行带 |。"""
    lines: List[str] = []
    previous = ""
    for kind, level, text in blocks:
        if lines and (kind in BLANK_BEFORE or kind != previous):
            lines.append("")
        if kind == "heading":
            lines.append(f"{'#' * max(1, min(6, level))} {text}")
        elif kind == "list":
            lines.append(f"- {text}")
        elif kind == "row":
            lines.append(f"| {text} |")
        elif kind == "rule":
            lines.append("---")
        else:
            lines.append(text)
        previous = kind
    return "\n".join(lines).strip()


def html_to_markdown(html_text: str) -> str:
    """HTML -> Markdown（只做标签到结构的映射，不清理噪声）。"""
    return blocks_to_markdown(html_to_blocks(html_text))


def count_lines(text: str) -> int:
    """有效行数：去掉空行以后还剩多少行。"""
    return len([line for line in str(text or "").splitlines() if line.strip()])


def is_noise(line: str) -> bool:
    """这一行像不像页面 UI（导航 / 按钮 / 备案号），而不是 JD 内容。"""
    body = STRUCT_PREFIX_RE.sub("", str(line or "")).strip()
    if not body:
        return True
    if SYMBOL_ONLY_RE.match(body):
        return True
    if len(body) <= NOISE_MAX_CHARS and NOISE_RE.search(body):
        return True
    return False


def clean_lines(text: str, strict: bool = True) -> Tuple[str, int]:
    """按行清理：空行压成一个、重复行只留第一次；strict=True 时再丢 UI 噪声行。

    返回 (清理后的文本, 丢掉的内容行数)。被丢掉的只有「行」，字一个不改。
    """
    kept: List[str] = []
    seen = set()
    dropped = 0
    blank = False
    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            blank = bool(kept)
            continue
        key = " ".join(line.split())
        if key in seen or (strict and is_noise(line)):
            dropped += 1
            continue
        seen.add(key)
        if blank:
            kept.append("")
            blank = False
        kept.append(line)
    return "\n".join(kept).strip(), dropped


def markdown_from_html(html_text: str, strict: bool = True) -> Tuple[str, int]:
    """HTML -> Markdown -> 按行清理；返回 (markdown, 丢掉的行数)。网址分支就走这一条。"""
    return clean_lines(html_to_markdown(html_text), strict=strict)
