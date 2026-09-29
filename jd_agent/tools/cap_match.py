"""能力词典匹配 Tool：把材料（JD / 简历）逐行映射到能力词典的能力项。

它回答的是「这份材料提到了哪些能力」这个问题，全部由规则给出：
  * 一份材料   → 命中的能力项 + 每项出现在第几行 + 是哪些关键词命中的，可直接回查原文；
  * 两份材料   → 再加一张对照表：哪些能力两边都有、哪些只有一边有（Polish Agent 用它把
                 「JD 要什么」和「简历里有什么」摆到同一张表上）。

词典本身在 `jd_agent/domain/lexicon.py`（主线解析 JD、解析项目、简历事实判定都用同一份词典），
这一层不新增任何词表，也不做语义猜测：词典没命中的能力项就是不出现。

工具区三条底线：不联网、不读 `.env`、不调用大模型。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..domain.jd import display_path, read_text
from ..agents.polish_brief import CapabilityHit, capability_hits, group_hits
from ..domain.resume_facts import table_lines
from .base import Tool, ToolResult
from .jd_files import parse_file_markdown

STEM = "cap_match"

# 聚合后的能力项：(key, 名字, 类别, 命中行号, 命中的关键词)
HitRow = Tuple[str, str, str, List[int], List[str]]


def _load(text: Optional[str], path, label: str) -> Tuple[str, str, str]:
    """读一份材料：返回 (正文, 给人看的来源, 错误)。文件走工具区的文件解析（纯规则）。"""
    if path:
        candidate = Path(str(path)).expanduser()
        if not candidate.is_file():
            return "", "", f"找不到文件：{candidate}"
        try:
            parsed = parse_file_markdown(candidate)
        except ValueError as exc:
            return "", "", str(exc)
        return parsed.markdown, label or display_path(candidate), ""
    return str(text or ""), label or "（粘贴的文本）", ""


def _table(rows: Sequence[HitRow], empty: str = "（没有命中任何能力项）") -> List[str]:
    if not rows:
        return [empty]
    body = [
        [name, category, "、".join(f"L{line}" for line in lines) or "—", "、".join(keywords) or "—"]
        for _, name, category, lines, keywords in rows
    ]
    return table_lines(("#", "能力项", "类别", "命中行号", "命中的关键词"), _numbered(body))


def _numbered(rows: Sequence[Sequence[object]]) -> List[List[object]]:
    return [[index, *row] for index, row in enumerate(rows, start=1)]


def _compare(left: Sequence[HitRow], right: Sequence[HitRow], left_label: str, right_label: str) -> List[str]:
    """对照表：并集为行，两边各自有没有命中。"""
    left_by_key = {row[0]: row for row in left}
    right_by_key = {row[0]: row for row in right}
    keys = [row[0] for row in left] + [key for key in (row[0] for row in right) if key not in left_by_key]
    rows: List[List[object]] = []
    for key in keys:
        in_left, in_right = key in left_by_key, key in right_by_key
        if in_left and in_right:
            verdict = "两边都有"
        elif in_left:
            verdict = f"只有{left_label}有"
        else:
            verdict = f"只有{right_label}有"
        name = (left_by_key.get(key) or right_by_key[key])[1]
        rows.append(
            [
                name,
                _lines_text(left_by_key.get(key)),
                _lines_text(right_by_key.get(key)),
                verdict,
            ]
        )
    if not rows:
        return ["（两份材料都没有命中能力项）"]
    return table_lines(("能力项", left_label, right_label, "结论"), rows)


def _lines_text(row: Optional[HitRow]) -> str:
    if not row or not row[3]:
        return "—"
    return "、".join(f"L{line}" for line in row[3])


def compare_counts(left: Sequence[HitRow], right: Sequence[HitRow]) -> Tuple[int, int, int]:
    """(两边都命中, 只有左边命中, 只有右边命中)。"""
    left_keys = {row[0] for row in left}
    right_keys = {row[0] for row in right}
    return (
        len(left_keys & right_keys),
        len(left_keys - right_keys),
        len(right_keys - left_keys),
    )


def render_cap_match_markdown(
    left: Sequence[HitRow],
    left_label: str,
    right: Sequence[HitRow] = (),
    right_label: str = "",
) -> str:
    """能力词典匹配报告：一份材料一张表；两份材料再加一张对照表。"""
    lines = [
        f"# 能力词典匹配 · {left_label}" + (f" × {right_label}" if right_label else ""),
        "",
        "> 由工具区的能力词典匹配 Tool（cap-match）生成"
        "｜纯规则：不联网、不读 .env、不调用大模型｜词表在 jd_agent/domain/lexicon.py",
        "",
        f"**{left_label}**：命中 {len(left)} 项能力",
        "",
        *_table(left),
        "",
    ]
    if right_label:
        lines += [
            f"**{right_label}**：命中 {len(right)} 项能力",
            "",
            *_table(right),
            "",
            "## 对照",
            "",
        ]
        lines += _compare(left, right, left_label, right_label)
    return "\n".join(lines).rstrip() + "\n"


class CapMatchTool(Tool):
    """能力词典匹配：一份材料出命中清单，两份材料出对照表（纯规则、不联网）。"""

    slug = "cap-match"
    title = "能力词典匹配"
    summary = "把材料（JD / 简历）逐行映射到能力词典的能力项；给两份材料再出一张对照表（纯规则、不联网）"
    usage = "python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md"

    def run(
        self,
        text: Optional[str] = None,
        path=None,
        other_text: Optional[str] = None,
        other_path=None,
        source_label: str = "",
        other_label: str = "",
        out_dir=None,
        stem: Optional[str] = None,
    ) -> ToolResult:
        """匹配能力项；输入有问题返回 ok=False，不抛异常、不打印。"""
        if bool(str(text or "").strip()) == bool(path):
            return ToolResult(ok=False, error="能力词典匹配要且只要给一份材料：text= 或 path=")
        if other_text is not None and other_path is not None:
            return ToolResult(ok=False, error="第二份材料同样只能给一个：other_text= 或 other_path=")

        body, label, error = _load(text, path, source_label)
        if error:
            return ToolResult(ok=False, error=error)
        if not body.strip():
            return ToolResult(ok=False, error="这份材料是空的，没有可匹配的内容")

        hits = capability_hits(body)
        rows = group_hits(hits)
        notes: List[str] = []
        if not rows:
            notes.append("这份材料一个能力项都没命中：检查一下是不是把正文粘漏了，或者换个写法")

        summary: List[Tuple[str, str]] = [
            ("材料", f"{label}（{len(body.splitlines())} 行）"),
            ("命中能力项", f"{len(rows)} 项"),
            ("命中位置", f"{sum(len(row[3]) for row in rows)} 处"),
        ]

        other_rows: List[HitRow] = []
        if other_text is not None or other_path is not None:
            other_body, other_label_resolved, other_error = _load(other_text, other_path, other_label)
            if other_error:
                return ToolResult(ok=False, error=other_error)
            if not other_body.strip():
                return ToolResult(ok=False, error="第二份材料是空的，没有可匹配的内容")
            other_label = other_label_resolved
            other_rows = group_hits(capability_hits(other_body))
            both, only_left, only_right = compare_counts(rows, other_rows)
            summary.append(
                (
                    "对照",
                    f"两份都命中 {both} 项；只有{label}命中 {only_left} 项；"
                    f"只有{other_label}命中 {only_right} 项",
                )
            )

        written: List[Path] = []
        if out_dir:
            directory = Path(out_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{stem or STEM}.md"
            target.write_text(
                render_cap_match_markdown(rows, label, other_rows, other_label), encoding="utf-8"
            )
            written.append(target)

        return ToolResult(ok=True, summary=summary, notes=notes, written=written)


CAP_MATCH_TOOL = CapMatchTool()
