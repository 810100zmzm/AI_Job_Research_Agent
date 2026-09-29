"""Small process-local stores used by the FastAPI skeleton.

Uploads and task metadata are intentionally simple. Durable L2 records live
in the existing memory layer instead of being duplicated here.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from threading import RLock
from typing import Callable, Dict, List, Optional

from fastapi import UploadFile

from ..core.schema import TraceStep
from ..tools.file_parser import (
    KIND_IMAGE,
    ParsedFile,
    classify_file,
    parse_file_to_markdown,
)


@dataclass(frozen=True)
class UploadItem:
    file_id: str
    filename: str
    file_type: str
    original_path: Path
    markdown_path: Path
    markdown: str
    created_at: float



@dataclass(frozen=True)
class InputFileItem:
    path: str
    name: str
    label: str
    kind: str
    file_type: str
    size: int
    modified_at: float


class InputFileCatalog:
    """Expose a bounded view of the project's input directory."""

    ROOTS = {"jd": "jd", "resume": "profile"}

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def list(self, kind: str) -> List[InputFileItem]:
        normalized = str(kind or "").strip().lower()
        base = self._base(normalized)
        if not base.is_dir():
            return []
        items: List[InputFileItem] = []
        for path in base.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            file_type = classify_file(path)
            if not file_type:
                continue
            relative = path.relative_to(base).as_posix()
            stat = path.stat()
            items.append(
                InputFileItem(
                    path=relative,
                    name=path.name,
                    label=f"{self.ROOTS[normalized]}/{relative}",
                    kind=normalized,
                    file_type=file_type,
                    size=int(stat.st_size),
                    modified_at=float(stat.st_mtime),
                )
            )
        return sorted(items, key=lambda item: item.label.casefold())

    def resolve(self, kind: str, relative: str) -> Path:
        value = str(relative or "").strip().replace("\\", "/")
        parsed = PurePosixPath(value)
        if (
            not value
            or parsed.is_absolute()
            or parsed.drive
            or any(part in ("", ".", "..") for part in parsed.parts)
        ):
            raise ValueError("input 文件路径无效")
        base = self._base(kind).resolve()
        candidate = (base / Path(*parsed.parts)).resolve()
        if not candidate.is_relative_to(base):
            raise ValueError("input 文件路径越界")
        if not candidate.is_file() or not classify_file(candidate):
            raise ValueError("input 文件不存在或格式不支持")
        return candidate

    def _base(self, kind: str) -> Path:
        directory = self.ROOTS.get(str(kind or "").strip().lower())
        if directory is None:
            raise ValueError("只支持选择 jd 或 resume")
        return self.root / directory


class UploadStore:
    """Persist uploaded originals and their parsed Markdown copies."""

    def __init__(
        self,
        root: Path,
        image_transcriber: Optional[Callable[[Path], str]] = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_transcriber = image_transcriber
        self._lock = RLock()

    async def save(self, upload: UploadFile) -> UploadItem:
        raw = await upload.read()
        if not raw:
            raise ValueError("上传文件为空")
        return self._save_raw(raw, upload.filename)

    def save_path(self, path: Path) -> UploadItem:
        source = Path(path)
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise ValueError(f"读取 input 文件失败：{exc}") from exc
        if not raw:
            raise ValueError("input 文件为空")
        return self._save_raw(raw, source.name)

    def _save_raw(self, raw: bytes, filename: Optional[str]) -> UploadItem:
        filename = self._safe_name(filename)
        file_id = uuid.uuid4().hex
        folder = self.root / file_id
        folder.mkdir(parents=True, exist_ok=False)
        original_path = folder / filename
        original_path.write_bytes(raw)

        parsed: ParsedFile = self._parse(original_path)
        if parsed.empty:
            raise ValueError("文件没有可解析的文本内容")
        markdown_path = folder / f"{Path(filename).stem or 'content'}.md"
        markdown_path.write_text(parsed.markdown, encoding="utf-8")

        item = UploadItem(
            file_id=file_id,
            filename=filename,
            file_type=parsed.kind,
            original_path=original_path,
            markdown_path=markdown_path,
            markdown=parsed.markdown,
            created_at=time.time(),
        )
        self._write_metadata(item)
        return item

    def _parse(self, path: Path) -> ParsedFile:
        if classify_file(path) != KIND_IMAGE:
            return parse_file_to_markdown(path, strict=False)
        if self.image_transcriber is None:
            raise ValueError("截图上传需要配置 DASHSCOPE_API_KEY / QWEN_API_KEY")
        try:
            markdown = str(self.image_transcriber(path) or "").strip()
        except Exception as exc:
            raise ValueError(f"截图转录失败：{exc}") from exc
        if not markdown:
            raise ValueError("截图没有转录出可用内容")
        return ParsedFile(
            path=str(path),
            kind=KIND_IMAGE,
            markdown=markdown,
            notes=["图片由 Qwen-VL 逐字转写为 Markdown"],
        )

    def get(self, file_id: str) -> Optional[UploadItem]:
        if not file_id:
            return None
        folder = self.root / self._safe_id(file_id)
        metadata_path = folder / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

        original_path = folder / str(payload.get("filename") or "upload")
        markdown_name = str(payload.get("markdown_name") or "content.md")
        markdown_path = folder / markdown_name
        if not original_path.is_file() or not markdown_path.is_file():
            return None
        try:
            markdown = markdown_path.read_text(encoding="utf-8")
        except OSError:
            return None
        return UploadItem(
            file_id=str(payload.get("file_id") or file_id),
            filename=original_path.name,
            file_type=str(payload.get("file_type") or ""),
            original_path=original_path,
            markdown_path=markdown_path,
            markdown=markdown,
            created_at=float(payload.get("created_at") or 0.0),
        )

    def resolve_input(self, value: str) -> Optional[Path]:
        """Resolve a value that may be an upload id to its Markdown path."""
        item = self.get(str(value or "").strip())
        return item.markdown_path if item else None

    def _write_metadata(self, item: UploadItem) -> None:
        payload = {
            "file_id": item.file_id,
            "filename": item.filename,
            "file_type": item.file_type,
            "markdown_name": item.markdown_path.name,
            "created_at": item.created_at,
        }
        with self._lock:
            (item.markdown_path.parent / "metadata.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _safe_name(filename: Optional[str]) -> str:
        name = Path(str(filename or "upload")).name.strip() or "upload"
        name = re.sub(r"[^\w.\- \u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
        return name[:160] or "upload"

    @staticmethod
    def _safe_id(file_id: str) -> str:
        return "".join(ch for ch in str(file_id) if ch.isalnum() or ch in "-_")


@dataclass(frozen=True)
class TaskTraceEvent:
    index: int
    action: str
    observation: str
    state_update: str
    decision: str

    @classmethod
    def from_step(cls, step: TraceStep) -> "TaskTraceEvent":
        return cls(
            index=int(step.index),
            action=str(step.action),
            observation=str(step.observation),
            state_update=str(step.state_update),
            decision=str(step.decision),
        )

    def as_payload(self) -> Dict[str, str]:
        return {
            "index": str(self.index),
            "action": self.action,
            "observation": self.observation,
            "state_update": self.state_update,
            "decision": self.decision,
        }


@dataclass
class TaskState:
    task_id: str
    status: str = "queued"
    error: str = ""
    trace: Dict[int, TaskTraceEvent] = field(default_factory=dict)
    reports: Dict[str, str] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)
    session_id: str = "default"
    user_id: str = "default"
    jd_input: str = ""
    resume_input: str = ""
    answer: str = ""
    question: str = ""
    question_reason: str = ""
    trace_offset: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class TaskStore:
    """In-memory task state; reports remain durable under ``output/``."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskState] = {}
        self._lock = RLock()

    def create(self) -> TaskState:
        task_id = uuid.uuid4().hex
        task = TaskState(task_id=task_id)
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Optional[TaskState]:
        with self._lock:
            return self._tasks.get(task_id)

    def start(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = "running"
            task.trace_offset = max(task.trace, default=0)
            task.updated_at = time.time()

    def append_trace(self, task_id: str, step: TraceStep) -> None:
        event = TaskTraceEvent.from_step(step)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            index = task.trace_offset + event.index
            event = TaskTraceEvent(
                index=index,
                action=event.action,
                observation=event.observation,
                state_update=event.state_update,
                decision=event.decision,
            )
            task.trace[index] = event
            task.updated_at = time.time()

    def complete(self, task_id: str, reports: Dict[str, Path]) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = "completed"
            task.reports = {key: str(path) for key, path in reports.items()}
            task.error = ""
            task.updated_at = time.time()

    def set_inputs(self, task_id: str, **inputs: object) -> None:
        """Remember materialized inputs so later resume generation can reuse them."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.inputs.update({key: str(value) for key, value in inputs.items() if value})
            task.updated_at = time.time()

    def inputs(self, task_id: str) -> Dict[str, str]:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task.inputs) if task is not None else {}

    def set_run_context(
        self,
        task_id: str,
        *,
        session_id: str,
        user_id: str,
        jd_input: str,
        resume_input: str,
    ) -> None:
        """Keep the original request so an Ask can resume without re-uploading."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.session_id = str(session_id or "default")
            task.user_id = str(user_id or "default")
            task.jd_input = str(jd_input)
            task.resume_input = str(resume_input)
            task.updated_at = time.time()

    def run_context(self, task_id: str) -> Optional[Dict[str, str]]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return {
                "session_id": task.session_id,
                "user_id": task.user_id,
                "jd_input": task.jd_input,
                "resume_input": task.resume_input,
            }

    def wait_for_answer(self, task_id: str, question: str, question_reason: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = "waiting_for_answer"
            task.question = str(question or "")
            task.question_reason = str(question_reason or "")
            task.error = ""
            task.updated_at = time.time()

    def accept_answer(self, task_id: str, answer: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != "waiting_for_answer":
                return False
            task.status = "running"
            task.answer = str(answer)
            task.question = ""
            task.question_reason = ""
            task.updated_at = time.time()
            return True

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = "failed"
            task.error = str(error)
            task.updated_at = time.time()

    def events_after(self, task_id: str, index: int = 0) -> List[TaskTraceEvent]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return []
            return [task.trace[key] for key in sorted(task.trace) if key > index]

    def report_path(self, task_id: str, report_format: str) -> Optional[Path]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            value = task.reports.get(report_format)
        return Path(value) if value else None
