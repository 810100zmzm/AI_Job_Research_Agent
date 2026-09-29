"""工具区：与 Agent 主链路解耦的小工具，统一从这里取。

每个工具都是一个 `Tool` 子类（自带 slug / title / summary / usage），
命令行（jd_agent/cli.py）与网页前端（streamlit_app.py）都只依赖这一层，不各自写一套逻辑。

当前工具：
    resume（简历排版）—— 读简历 md → 有序的 md / html，三种风格可选
    jd-file（JD 文件解析）—— 读本地 JD 文件（md/txt/html/json/yaml）→ Markdown 草稿
    jd-struct（JD 结构化）—— JD 原文 → 统一模板的 JD Markdown（岗位职责 / 任职要求 / 加分项）
    resume-struct（简历结构化）—— 简历原文 → 章节 + 事实条目（带原文行号）
    resume-evidence（证据等级判定）—— 逐条判「有结果 / 有动作 / 仅提及」，严格与放宽两个口径
    resume-negation（否定 / 背景识别）—— 挡住「写了但没做过 / 只是背景」的条目
    cap-match（能力词典匹配）—— 把材料（JD / 简历）逐行映射到能力词典的能力项，两份材料还能出对照

加一个新工具只要三步：继承 `Tool`、实现 `run()` 返回 `ToolResult`、把实例追加进 `TOOLS`；
命令行与前端都是从这一层取工具，不需要各自再写一遍。
"""
from __future__ import annotations

from typing import Optional, Tuple

from . import resume_styles
from .base import Tool, ToolResult
from .cap_match import (
    CAP_MATCH_TOOL,
    CapMatchTool,
    compare_counts,
    render_cap_match_markdown,
)
from .jd_files import (
    JD_FILE_TOOL,
    ParsedFile,
    JDFileTool,
    classify_file,
    json_to_markdown,
    parse_file_markdown,
    parsed_summary,
    yaml_to_markdown,
)
from .file_parser import (
    SUPPORTED_SUFFIXES,
    detect_file_type,
    is_supported_file,
    parse_file_to_markdown,
)
from .jd_struct import (
    JD_STRUCT_TOOL,
    JDPosition,
    JDStructTool,
    StructuredJD,
    jd_summary,
    render_jd_markdown,
    structure_jd_text,
)
from .resume_evidence import (
    RESUME_EVIDENCE_TOOL,
    ResumeEvidenceTool,
    evidence_summary_rows,
    render_evidence_markdown,
    render_evidence_table,
    summarize_evidence,
)
from .resume_negation import (
    RESUME_NEGATION_TOOL,
    ResumeNegationTool,
    ScreenReport,
    explain_line,
    negation_summary,
    render_blocked_table,
    render_negation_markdown,
    screen_facts,
)
from .resume_struct import (
    RESUME_STRUCT_TOOL,
    ResumeStructTool,
    load_resume_doc,
    render_struct_markdown,
    struct_summary,
    structure_resume_text,
)
from .resume import (
    RESUME_FORMATS,
    RESUME_STEM,
    RESUME_TOOL,
    Node,
    Resume,
    ResumeTool,
    Section,
    Table,
    build_resume,
    default_stem,
    parse_resume,
    render_html,
    render_markdown,
    resume_summary,
    write_resume_files,
)

TOOLS: Tuple[Tool, ...] = (
    RESUME_TOOL,
    JD_FILE_TOOL,
    JD_STRUCT_TOOL,
    RESUME_STRUCT_TOOL,
    RESUME_EVIDENCE_TOOL,
    RESUME_NEGATION_TOOL,
    CAP_MATCH_TOOL,
)
TOOL_BY_SLUG = {tool.slug: tool for tool in TOOLS}


def get_tool(slug: str) -> Optional[Tool]:
    """按 slug 取工具；没有就返回 None。"""
    return TOOL_BY_SLUG.get(slug)


def tool_help_lines() -> Tuple[str, ...]:
    """每行一个工具说明，给 --help 用。"""
    return tuple(tool.help_line for tool in TOOLS)


__all__ = [
    "TOOLS",
    "TOOL_BY_SLUG",
    "Tool",
    "ToolResult",
    "get_tool",
    "tool_help_lines",
    "RESUME_TOOL",
    "ResumeTool",
    "RESUME_FORMATS",
    "RESUME_STEM",
    "default_stem",
    "Resume",
    "Section",
    "Node",
    "Table",
    "build_resume",
    "parse_resume",
    "render_markdown",
    "render_html",
    "resume_summary",
    "write_resume_files",
    "resume_styles",
    "JD_FILE_TOOL",
    "JDFileTool",
    "ParsedFile",
    "classify_file",
    "json_to_markdown",
    "yaml_to_markdown",
    "parse_file_markdown",
    "parsed_summary",
    "SUPPORTED_SUFFIXES",
    "detect_file_type",
    "is_supported_file",
    "parse_file_to_markdown",
    "JD_STRUCT_TOOL",
    "JDStructTool",
    "JDPosition",
    "StructuredJD",
    "structure_jd_text",
    "render_jd_markdown",
    "jd_summary",
    "RESUME_STRUCT_TOOL",
    "ResumeStructTool",
    "structure_resume_text",
    "load_resume_doc",
    "render_struct_markdown",
    "struct_summary",
    "RESUME_EVIDENCE_TOOL",
    "ResumeEvidenceTool",
    "summarize_evidence",
    "render_evidence_markdown",
    "render_evidence_table",
    "evidence_summary_rows",
    "RESUME_NEGATION_TOOL",
    "ResumeNegationTool",
    "screen_facts",
    "ScreenReport",
    "explain_line",
    "render_blocked_table",
    "render_negation_markdown",
    "negation_summary",
    "CAP_MATCH_TOOL",
    "CapMatchTool",
    "render_cap_match_markdown",
    "compare_counts",
]
