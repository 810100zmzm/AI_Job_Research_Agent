"""把某个 Agent 的 LangGraph 状态图直接画在 Notebook 里。

用法（Jupyter / VS Code Notebook）：

    from jd_agent.agents.graph_view import show_graph
    from jd_agent.agents.polish_graph import compiled_graph

    show_graph(compiled_graph())

默认走 LangGraph 官方的 `draw_mermaid_png()`：先把图转成 mermaid 源码，
再交给 mermaid.ink 渲染成 PNG 字节流，最后用 `IPython.display.Image` 内联显示。
本地不用装 graphviz / pydot / pyppeteer。

没有网络时改用 `method="mermaid"`，不联网，直接把 mermaid 源码显示出来。

注意：图是 LangGraph 按真实节点 / 边生成的，节点名就是 Trace 里的 Action 名，
和 `compiled_graph().nodes` 完全一致，不存在「画得比代码好看」的问题。
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["METHODS", "graph_png_bytes", "show_graph"]

METHODS = ("png", "mermaid")


def _drawable(graph: Any) -> Any:
    """接受「已编译的图」或「get_graph() 的返回值」，统一拿到可画的对象。"""
    return graph.get_graph() if hasattr(graph, "get_graph") else graph


def graph_png_bytes(
    graph: Any,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    background_color: str = "white",
    padding: int = 10,
    frontmatter_config: Optional[dict] = None,
) -> bytes:
    """只取 PNG 字节流，不显示。需要联网（走 mermaid.ink）。"""
    return _drawable(graph).draw_mermaid_png(
        max_retries=max_retries,
        retry_delay=retry_delay,
        background_color=background_color,
        padding=padding,
        frontmatter_config=frontmatter_config,
    )


def show_graph(
    graph: Any,
    *,
    method: str = "png",
    output_file_path: Optional[str] = None,
    width: Optional[int] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    frontmatter_config: Optional[dict] = None,
) -> Optional[bytes]:
    """在 Notebook 里显示图，返回 PNG 字节流（method="png"）或 None。

    method:
      - "png"：默认。在线渲染成 PNG 后内联显示；给了 output_file_path 就顺手存一份。
      - "mermaid"：只显示 mermaid 源码，方便粘进支持 mermaid 的编辑器。
    """
    if method not in METHODS:
        raise ValueError(f"method 只能是 {METHODS} 里的一个，收到 {method!r}")

    drawable = _drawable(graph)

    if method == "mermaid":
        from IPython.display import Markdown, display

        display(Markdown("```mermaid\n" + drawable.draw_mermaid() + "\n```"))
        return None

    from IPython.display import Image, display

    data = graph_png_bytes(
        drawable,
        max_retries=max_retries,
        retry_delay=retry_delay,
        frontmatter_config=frontmatter_config,
    )
    display(Image(data, width=width))
    if output_file_path:
        with open(output_file_path, "wb") as fh:
            fh.write(data)
    return data
