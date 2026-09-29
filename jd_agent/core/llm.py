"""大模型调用层（可选）：用标准库 urllib 调 OpenAI 兼容接口，不需要任何 SDK。

这一层只有两个用途，判定权始终留在规则手里：
    DeepSeek  deepseek-chat —— 规则报告生成之后，补三块规则做不到的内容
                              （建议写法草稿 / 面试追问预演 / 报告润色）
    Qwen-VL   qwen-vl-max   —— 把项目描述里引用的图片读成文字事实

默认完全不联网：只有 --llm / --vision / --check-llm 才会创建客户端并发请求；
任何失败都抛 LLMError，由上层记成一块「调用失败」的内容，不影响规则结论与退出码。

生成层的三条调用纪律（红线，全部是常量，可直接检查）：
    单次调用超时 DEFAULT_CALL_TIMEOUT 秒；失败重试 DEFAULT_RETRIES 次；一次运行最多 DEFAULT_CALL_LIMIT 次调用。
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # 只用于类型标注，运行时不导入，避免与 settings.py 循环依赖
    from .settings import LLMSettings

# ---- 默认端点与模型（可用 .env / --text-model / --vision-model 覆盖）--------
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_VL_MODEL = "qwen-vl-max"
DEFAULT_TIMEOUT = 60.0

# v1.1 生成层（报告已生成之后才介入）的调用纪律
DEFAULT_CALL_TIMEOUT = 30.0   # 单次 LLM 调用超时（秒）：不阻塞主流程
DEFAULT_CALL_LIMIT = 10       # 单次运行最多调用几次（含重试）
DEFAULT_RETRIES = 1           # 失败重试次数

CHAT_PATH = "/chat/completions"
EMBEDDINGS_PATH = "/embeddings"
PING_MESSAGE = "ping：请只回复 pong"


class LLMError(RuntimeError):
    """一次调用失败（网络 / 额度 / 模型名 / 返回格式）。"""


class OpenAICompatClient:
    """OpenAI 兼容接口的最小客户端。

    DeepSeek 与 DashScope compatible-mode 都是同一套 /chat/completions 协议，
    因此文本层与多模态层共用这一个类，只是 base_url / model 不同。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = float(timeout or DEFAULT_TIMEOUT)

    @property
    def url(self) -> str:
        return f"{self.base_url}{CHAT_PATH}"

    def __repr__(self) -> str:  # 任何情况下都不回显 key
        return f"OpenAICompatClient(base_url={self.base_url!r}, model={self.model!r}, api_key=***)"

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """发一次请求并返回消息文本；任何失败都抛 LLMError。"""
        if not self.api_key:
            raise LLMError("缺少 API Key")
        payload: Dict[str, Any] = {"model": model or self.model, "messages": list(messages)}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        return _content(_post(request, self.timeout))

    def embed(
        self,
        texts: Sequence[str],
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """Call an OpenAI-compatible embeddings endpoint."""
        if not self.api_key:
            raise LLMError("缺少 API Key")
        payload: Dict[str, Any] = {"model": model or self.model, "input": list(texts)}
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        request = urllib.request.Request(
            f"{self.base_url}{EMBEDDINGS_PATH}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        return _embeddings(_post(request, self.timeout), expected=len(texts))


def _post(request: urllib.request.Request, timeout: float) -> str:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise LLMError(f"HTTP {exc.code}：{brief(detail)}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"网络不可达：{exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LLMError(f"请求超时（{timeout:g}s）") from exc


def _content(raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMError(f"返回不是合法 JSON：{brief(raw)}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise LLMError(f"接口报错：{brief(json.dumps(data['error'], ensure_ascii=False))}")
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise LLMError(f"返回里没有 choices：{brief(raw)}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):  # 少数兼容实现会返回分段内容
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = str(content or "").strip()
    if not text:
        raise LLMError("返回内容为空")
    return text


def _embeddings(raw: str, expected: int) -> List[List[float]]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMError(f"返回不是合法 JSON：{brief(raw)}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise LLMError(f"接口报错：{brief(json.dumps(data['error'], ensure_ascii=False))}")
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise LLMError(f"返回里没有 embeddings：{brief(raw)}")
    ordered = sorted(rows, key=lambda item: int(item.get("index", 0)))
    vectors = [item.get("embedding") for item in ordered]
    if expected and len(vectors) != expected:
        raise LLMError(f"嵌入数量不一致：期望 {expected}，返回 {len(vectors)}")
    result: List[List[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise LLMError("嵌入结果为空或格式错误")
        result.append([float(value) for value in vector])
    return result


def brief(text: str, limit: int = 200) -> str:
    """压成一行并截断，用于错误信息与日志。"""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def parse_json_reply(text: str) -> Any:
    """从模型回复里取出 JSON 对象（容忍 ``` 代码块与前后闲话）；取不出来返回 None。"""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        newline = raw.find("\n")
        if newline >= 0 and " " not in raw[:newline]:
            raw = raw[newline + 1 :]
    candidates = [raw]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(raw[start : end + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ---- 自检（--check-llm）-----------------------------------------------------

@dataclass
class CheckResult:
    """一次自检的结果：只带标签、模型名与错误摘要，不带任何密钥。"""

    label: str
    model: str
    ok: bool
    detail: str


def check_llm(
    settings: Optional["LLMSettings"],
    text_client: Optional[OpenAICompatClient] = None,
    vision_client: Optional[OpenAICompatClient] = None,
    check_vision: bool = False,
) -> List[CheckResult]:
    """给启用的层各发一条最小请求，自检 key / 网络 / 模型名。"""
    text_model = getattr(settings, "text_model", DEFAULT_DEEPSEEK_MODEL)
    vision_model = getattr(settings, "vision_model", DEFAULT_QWEN_VL_MODEL)
    results: List[CheckResult] = []

    text = text_client if text_client is not None else build_client(settings, "text")
    if text is None:
        results.append(CheckResult("文本层 DeepSeek", text_model, False, "未配置 DEEPSEEK_API_KEY"))
    else:
        results.append(probe(text, "文本层 DeepSeek", text_model))

    if check_vision:
        vision = vision_client if vision_client is not None else build_client(settings, "vision")
        if vision is None:
            results.append(
                CheckResult("多模态层 Qwen-VL", vision_model, False, "未配置 DASHSCOPE_API_KEY / QWEN_API_KEY")
            )
        else:
            results.append(probe(vision, "多模态层 Qwen-VL", vision_model))
    return results


def probe(client: OpenAICompatClient, label: str, model: str) -> CheckResult:
    try:
        reply = client.chat([{"role": "user", "content": PING_MESSAGE}], max_tokens=8, temperature=0)
    except LLMError as exc:
        return CheckResult(label, model, False, str(exc))
    except Exception as exc:  # 自检不该把程序打崩
        return CheckResult(label, model, False, f"{exc.__class__.__name__}: {exc}")
    return CheckResult(label, model, True, f"响应正常：{brief(reply, 60)}")


def build_client(
    settings: Optional["LLMSettings"], layer: str, timeout: Optional[float] = None
) -> Optional[OpenAICompatClient]:
    """按配置创建客户端；没配 key 就返回 None（缺 key 绝不发请求）。

    timeout 只有生成层会传（不阻塞：单次调用 30 秒），其它层沿用 settings.timeout。
    """
    if settings is None:
        return None
    seconds = settings.timeout if timeout is None else timeout
    if layer == "text":
        if not settings.text_ready:
            return None
        return OpenAICompatClient(
            settings.text_base_url, settings.text_api_key, settings.text_model, seconds
        )
    if not settings.vision_ready:
        return None
    return OpenAICompatClient(
        settings.vision_base_url, settings.vision_api_key, settings.vision_model, seconds
    )
