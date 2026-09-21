"""工具的公共契约：命令行与网页前端用同一套方式调用每个工具。

三条约定：
  * 工具都是 `Tool` 的子类，自带 slug / title / summary / usage，--help 与前端直接列出来就行；
  * 工具只做一件事，结果统一用 `ToolResult` 表达；**输入有问题不抛异常**，用 ok=False + error 说明；
  * 工具不许读 .env、不许发网络请求 —— 需要联网的能力属于 Agent 主链路，不属于工具区。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class ToolResult:
    """一次工具运行的结果。"""

    ok: bool = True
    error: str = ""                                              # ok=False 时给人看的说明
    summary: List[Tuple[str, str]] = field(default_factory=list)  # (标签, 值)，给终端与前端显示
    notes: List[str] = field(default_factory=list)                # 做了什么规整 / 提示
    written: List[Path] = field(default_factory=list)             # 写出的文件

    @property
    def files(self) -> List[str]:
        return [str(item) for item in self.written]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "summary": [{"label": label, "value": value} for label, value in self.summary],
            "notes": list(self.notes),
            "files": self.files,
        }


class Tool(ABC):
    """工具基类：一个 slug + 一段说明 + 一个 run()。"""

    slug: str = ""
    title: str = ""
    summary: str = ""
    usage: str = ""

    @abstractmethod
    def run(self, **options) -> ToolResult:
        """跑一次工具；任何输入问题都通过 ToolResult(ok=False) 返回。"""

    @property
    def help_line(self) -> str:
        return f"{self.title}（{self.slug}）：{self.summary}"

    def __repr__(self) -> str:  # pragma: no cover - 只是让日志好读
        return f"<{self.__class__.__name__} slug={self.slug}>"
