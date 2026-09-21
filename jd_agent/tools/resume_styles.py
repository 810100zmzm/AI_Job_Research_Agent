"""三套简历排版风格：只改「怎么排」，不改「写什么」。

Markdown 本身没有排版能力，所以一套风格由两部分组成：
  1. CSS（作用于 `.resume-doc`，既用于下载的 HTML，也用于前端预览）；
  2. 两条 Markdown 约定 —— 章节之间插不插 `---`、子条目用 `###` 还是加粗行。

| key | 名字 | 适合场景 |
|-----|------|----------|
| classic | 经典简约 | 通用；细灰线 + 无衬线，粘到投递平台最稳。默认风格，输出与 v1.1 逐字一致 |
| compact | 紧凑一页 | 内容多、想压到一页；字号与行距更紧，章节标题带浅底（只换 CSS，不动 Markdown） |
| accent  | 强调竖线 | 想要一点设计感；章节标题带竖线，表格去掉竖格线，章节间加 `---`、子条目加粗 |
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

DEFAULT_STYLE = "classic"

# 只用于「下载下来单独打开」的那份 HTML：纸面底色与页边距。
# 嵌到别的页面预览时不加，免得影响宿主（样式全部限定在 .resume-doc 里）。
PAGE_CSS = "body { margin: 0; background: #ffffff; color: #1f2328; }\n"

# 三套风格共用的骨架：版心、字体、列表、表格、打印
BASE_CSS = """
.resume-doc { color-scheme: light; color: #1f2328; --line: #d0d7de; --muted: #57606a; --accent: #2b6cb0;
              max-width: 860px; margin: 0 auto; padding: 40px 28px 64px; line-height: 1.72;
              font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
              font-size: 14.5px; }
.resume-doc * { box-sizing: border-box; }
.resume-doc header { border-bottom: 2px solid var(--line); padding-bottom: 14px; margin-bottom: 10px; }
.resume-doc h1 { font-size: 29px; letter-spacing: .5px; margin: 0 0 8px; }
.resume-doc .tagline { color: var(--muted); font-size: 13.5px; margin: 0; }
.resume-doc .tagline span + span::before { content: "　｜　"; color: var(--line); }
.resume-doc section { margin-top: 30px; break-inside: avoid; }
.resume-doc h2 { font-size: 17px; margin: 0 0 12px; letter-spacing: .5px; }
.resume-doc h3 { font-size: 15px; margin: 18px 0 6px; }
.resume-doc p { margin: 6px 0; }
.resume-doc ul { margin: 6px 0; padding-left: 20px; }
.resume-doc li { margin: 3px 0; }
.resume-doc li::marker { color: var(--accent); }
.resume-doc strong { font-weight: 600; }
.resume-doc code { background: #f0f2f5; padding: 1px 5px; border-radius: 4px; font-size: 12.8px; }
.resume-doc table { border-collapse: collapse; width: 100%; margin: 10px 0 14px; font-size: 13.5px; }
.resume-doc th, .resume-doc td { border: 1px solid var(--line); padding: 6px 9px; text-align: left;
                                 vertical-align: top; }
.resume-doc th { background: #f6f8fa; font-weight: 600; }
.resume-doc blockquote { margin: 6px 0; padding-left: 12px; border-left: 3px solid var(--line);
                         color: var(--muted); }
.resume-doc .notes { margin-top: 36px; color: var(--muted); font-size: 12px; }
@media print {
  @page { margin: 14mm; }
  .resume-doc { max-width: none; padding: 0; font-size: 12.5pt; }
  .resume-doc section, .resume-doc h3, .resume-doc table { break-inside: avoid; }
  .resume-doc .notes { display: none; }
}
"""

# 经典简约：章节标题下一条细线，表格带浅底表头
CLASSIC_CSS = """
.resume-doc h2 { padding-bottom: 5px; border-bottom: 1px solid var(--line); }
"""

# 紧凑一页：整体收小，章节标题带浅底块
COMPACT_CSS = """
.resume-doc { max-width: 800px; padding: 30px 22px 40px; line-height: 1.6; font-size: 13px; }
.resume-doc h1 { font-size: 24px; margin: 0 0 6px; }
.resume-doc .tagline { font-size: 12.5px; }
.resume-doc section { margin-top: 20px; }
.resume-doc h2 { font-size: 14px; letter-spacing: 1px; margin: 0 0 8px; padding: 4px 8px;
                 background: #f2f4f7; border-radius: 4px; }
.resume-doc h3 { font-size: 14px; margin: 12px 0 4px; }
.resume-doc p, .resume-doc ul { margin: 4px 0; }
.resume-doc li { margin: 2px 0; }
.resume-doc table { font-size: 12.5px; margin: 8px 0 10px; }
.resume-doc th, .resume-doc td { padding: 4px 7px; }
@media print { .resume-doc { font-size: 10.5pt; } }
"""

# 强调竖线：章节标题带色条，表格去掉竖格线，求职意向做成一条浅底色块
ACCENT_CSS = """
.resume-doc { --accent: #0f766e; --line: #dbe3e6; --muted: #4b5563; }
.resume-doc header { border-bottom: 3px double var(--line); }
.resume-doc h1 { font-size: 27px; letter-spacing: 1.5px; }
.resume-doc .tagline { display: inline-block; background: #f1f5f9; color: #334155;
                       padding: 6px 12px; border-radius: 6px; }
.resume-doc h2 { border-left: 4px solid var(--accent); padding-left: 10px; letter-spacing: 1px; }
.resume-doc h3 { color: #0f172a; }
.resume-doc th, .resume-doc td { border: none; border-bottom: 1px solid var(--line); padding: 6px 4px; }
.resume-doc th { background: transparent; border-bottom: 2px solid var(--line); }
.resume-doc blockquote { border-left-color: var(--accent); }
"""


@dataclass(frozen=True)
class ResumeStyle:
    """一套排版风格。"""

    key: str
    name: str
    summary: str
    css: str
    rule_between_sections: bool = True   # Markdown：章节之间插 ---
    bold_sub_entries: bool = False       # Markdown：子条目用加粗行代替 ###

    @property
    def label(self) -> str:
        return f"{self.name}（{self.key}）"

    @property
    def help_text(self) -> str:
        return f"{self.label}——{self.summary}"


STYLES: Tuple[ResumeStyle, ...] = (
    ResumeStyle(
        key="classic",
        name="经典简约",
        summary="细灰线 + 无衬线，通用；粘到投递平台最稳（默认，输出与 v1.1 逐字一致）",
        css=BASE_CSS + CLASSIC_CSS,
        rule_between_sections=False,
    ),
    ResumeStyle(
        key="compact",
        name="紧凑一页",
        summary="字号与行距更紧、章节标题带浅底块，内容多时压到一页（只换样式，不改 Markdown）",
        css=BASE_CSS + COMPACT_CSS,
        rule_between_sections=False,
    ),
    ResumeStyle(
        key="accent",
        name="强调竖线",
        summary="章节标题带色条、表格去掉竖格线，章节间加分隔线、子条目加粗，多一点设计感",
        css=BASE_CSS + ACCENT_CSS,
        bold_sub_entries=True,
    ),
)

STYLE_BY_KEY: Dict[str, ResumeStyle] = {style.key: style for style in STYLES}
STYLE_KEYS: Tuple[str, ...] = tuple(style.key for style in STYLES)


def all_styles() -> Tuple[ResumeStyle, ...]:
    """按展示顺序返回全部风格。"""
    return STYLES


def get_style(key: str = DEFAULT_STYLE) -> ResumeStyle:
    """按 key 取风格；不认识的 key 直接报错，避免静默用错风格。"""
    name = (key or DEFAULT_STYLE).strip().lower()
    if name not in STYLE_BY_KEY:
        raise ValueError(f"没有这种排版风格：{key}（可选 {'、'.join(STYLE_KEYS)}）")
    return STYLE_BY_KEY[name]


def describe_styles() -> List[str]:
    """每套风格一行说明，给 --help 与前端用。"""
    return [style.help_text for style in STYLES]
