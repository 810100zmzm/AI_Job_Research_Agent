"""配置：用 python-dotenv 从项目根 `.env` 读 API Key 与可选覆盖项。

`.env` 的解析交给 python-dotenv（`load_dotenv()` / `dotenv_values()`），这里只做三件事：
  * 只有 --llm / --vision / --check-llm 才会加载 .env，默认保持完全离线；
  * 已经存在的环境变量优先级更高（load_dotenv 默认 override=False，绝不覆盖它们）；
  * 记下「加载了哪些键 / 跳过了哪些键」，日志里只允许出现键名，绝不回显值。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Tuple

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError as exc:   # 依赖只有一个，缺了就直接说清楚怎么装
    raise ImportError(
        "缺少依赖 python-dotenv：先跑 pip install -r requirements.txt"
        "（或 pip install python-dotenv）"
    ) from exc

from .llm import (
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_QWEN_VL_MODEL,
    DEFAULT_TIMEOUT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# 认得的键名（加新键时改这里；日志里出现的也只有这些名字）
ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DASHSCOPE_BASE_URL",
    "QWEN_VL_MODEL",
    "LLM_TIMEOUT",
)


@dataclass
class EnvLoadResult:
    """一次 .env 加载的结果：只有键名，没有值。"""

    path: str = ""
    found: bool = False
    applied: List[str] = field(default_factory=list)   # 本次补进环境变量的键
    skipped: List[str] = field(default_factory=list)   # 环境变量里已有、未被覆盖的键
    error: str = ""

    def summary(self) -> str:
        if self.error:
            return self.error
        if not self.found:
            return f"没有找到 {self.path}（跳过，继续用现有环境变量）"
        parts: List[str] = []
        if self.applied:
            parts.append("已加载：" + "、".join(self.applied))
        if self.skipped:
            parts.append("环境变量里已有、未覆盖：" + "、".join(self.skipped))
        return "；".join(parts) or f"{self.path} 里没有可用的键"


def load_env(path=None) -> EnvLoadResult:
    """用 load_dotenv() 把 .env 里缺失的键补进环境变量；已存在的键一律不动。"""
    target = Path(path) if path else DEFAULT_ENV_FILE
    result = EnvLoadResult(path=str(target))
    if not target.is_file():
        return result
    result.found = True
    try:
        values: Dict[str, Optional[str]] = dotenv_values(target)
    except Exception as exc:                     # python-dotenv 解析失败也不该带崩主流程
        result.error = f"读取 {target} 失败：{exc}"
        return result
    result.skipped = [key for key in values if key in os.environ]
    try:
        load_dotenv(dotenv_path=target, override=False)
    except Exception as exc:
        result.error = f"加载 {target} 失败：{exc}"
        return result
    result.applied = [key for key in values if key not in result.skipped]
    return result


@dataclass(frozen=True)
class LLMSettings:
    """一次运行用到的全部大模型配置（只记键名，方便打印）。"""

    text_api_key: str = ""
    text_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    text_model: str = DEFAULT_DEEPSEEK_MODEL
    vision_api_key: str = ""
    vision_base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    vision_model: str = DEFAULT_QWEN_VL_MODEL
    timeout: float = DEFAULT_TIMEOUT
    env_file: str = ""
    env_keys: Tuple[str, ...] = ()

    @property
    def text_ready(self) -> bool:
        return bool(self.text_api_key.strip())

    @property
    def vision_ready(self) -> bool:
        return bool(self.vision_api_key.strip())

    def describe(self) -> List[str]:
        """给终端看的说明：只有键名、地址与模型名，没有值。"""
        text_key = "key=已配置 DEEPSEEK_API_KEY" if self.text_ready else "key=未配置 DEEPSEEK_API_KEY（该层跳过）"
        vision_key = (
            "key=已配置 DASHSCOPE_API_KEY / QWEN_API_KEY"
            if self.vision_ready
            else "key=未配置 DASHSCOPE_API_KEY / QWEN_API_KEY（该层跳过）"
        )
        return [
            f"文本层 DeepSeek：model={self.text_model}，base_url={self.text_base_url}，{text_key}",
            f"多模态层 Qwen-VL：model={self.vision_model}，base_url={self.vision_base_url}，{vision_key}",
            f"超时 LLM_TIMEOUT：{self.timeout:g}s",
        ]


def _value(environ: MutableMapping[str, str], key: str, default: str = "") -> str:
    raw = environ.get(key)
    if raw is None:
        return default
    raw = str(raw).strip()
    return raw or default


def _timeout(environ: MutableMapping[str, str]) -> float:
    raw = _value(environ, "LLM_TIMEOUT")
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def resolve_settings(
    environ: Optional[MutableMapping[str, str]] = None,
    *,
    env_file: str = "",
    env_keys: Tuple[str, ...] = (),
    text_model: str = "",
    vision_model: str = "",
    text_base_url: str = "",
    vision_base_url: str = "",
) -> LLMSettings:
    """从环境变量（已并入 .env 的内容）解析配置；命令行参数优先级最高。"""
    env: MutableMapping[str, str] = os.environ if environ is None else environ
    return LLMSettings(
        text_api_key=_value(env, "DEEPSEEK_API_KEY"),
        text_base_url=text_base_url or _value(env, "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        text_model=text_model or _value(env, "DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        vision_api_key=_value(env, "DASHSCOPE_API_KEY") or _value(env, "QWEN_API_KEY"),
        vision_base_url=vision_base_url or _value(env, "DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
        vision_model=vision_model or _value(env, "QWEN_VL_MODEL", DEFAULT_QWEN_VL_MODEL),
        timeout=_timeout(env),
        env_file=env_file,
        env_keys=tuple(env_keys),
    )
