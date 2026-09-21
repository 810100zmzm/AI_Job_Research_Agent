"""工具区：与 Agent 主链路解耦的小工具，统一从这里取。

每个工具都是一个 `Tool` 子类（自带 slug / title / summary / usage），
命令行（jd_agent/cli.py）与网页前端（streamlit_app.py）都只依赖这一层，不各自写一套逻辑。

当前工具：
    resume（简历排版）—— 读简历 md → 有序的 md / html，三种风格可选

加一个新工具只要三步：继承 `Tool`、实现 `run()` 返回 `ToolResult`、把实例追加进 `TOOLS`；
命令行与前端都是从这一层取工具，不需要各自再写一遍。
"""
from __future__ import annotations

from typing import Optional, Tuple

from . import resume_styles
from .base import Tool, ToolResult
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

TOOLS: Tuple[Tool, ...] = (RESUME_TOOL,)
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
]
