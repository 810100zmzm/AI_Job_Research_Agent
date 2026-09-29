"""Application services behind the FastAPI routes."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..agents.agent import run_agent
from ..agents.render import render_html, render_json, render_markdown
from ..agents.resume_polish import level_options, polish_resume, style_options
from ..core.llm import DEFAULT_CALL_TIMEOUT, build_client
from ..core.settings import DEFAULT_ENV_FILE, load_env, resolve_settings
from ..core.schema import DECISION_ASK
from ..memory import MemoryManager
from .stores import InputFileCatalog, InputFileItem, TaskStore, UploadItem, UploadStore

logger = logging.getLogger(__name__)


class AnalysisService:
    """Run the existing Agent loop and persist its report artifacts."""

    def __init__(
        self,
        tasks: TaskStore,
        uploads: UploadStore,
        memory: MemoryManager,
        data_dir: Path,
        output_dir: Path,
        input_dir: Path,
        resume_text_client_factory=None,
    ) -> None:
        self.tasks = tasks
        self.uploads = uploads
        self.memory = memory
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.input_files = InputFileCatalog(input_dir)
        self.resume_text_client_factory = resume_text_client_factory
        self.resume_output_dir = self.output_dir / "resumes"

    def report_path(self, task_id: str, report_format: str) -> Optional[Path]:
        """Return a generated report, including after a process restart."""
        tracked = self.tasks.report_path(task_id, report_format)
        if tracked is not None and tracked.is_file():
            return tracked
        filename = f"{Path(task_id).name}.{report_format}"
        candidate = self.output_dir / filename
        return candidate if candidate.is_file() else None

    def run(
        self,
        task_id: str,
        session_id: str,
        user_id: str,
        jd_input: str,
        resume_input: str,
        answer: str = "",
    ) -> None:
        self.tasks.start(task_id)
        try:
            task_dir = self.data_dir / "tmp" / "tasks" / task_id
            jd_path = self._materialize_input(jd_input, task_dir / "jd.md")
            resume_path = self._materialize_input(resume_input, task_dir / "resume.md")
            self.tasks.set_inputs(task_id, jd_path=jd_path, resume_path=resume_path)
            self.tasks.set_run_context(
                task_id,
                session_id=session_id,
                user_id=user_id,
                jd_input=jd_input,
                resume_input=resume_input,
            )
            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

            state = run_agent(
                jd_path,
                resume_path,
                answer=answer,
                memory=self.memory,
                session_id=session_id or "default",
                user_id=user_id or "default",
                trace_callback=lambda step: self.tasks.append_trace(task_id, step),
            )

            # Reconcile once in case a trace callback was skipped by an injected
            # runner or an early return path. TaskStore de-duplicates by index.
            for step in state.trace:
                self.tasks.append_trace(task_id, step)

            if state.decision == DECISION_ASK:
                self.tasks.wait_for_answer(task_id, state.question, state.question_reason)
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            reports: Dict[str, Path] = {
                "md": self.output_dir / f"{task_id}.md",
                "json": self.output_dir / f"{task_id}.json",
                "html": self.output_dir / f"{task_id}.html",
            }
            reports["md"].write_text(render_markdown(state, generated_at), encoding="utf-8")
            reports["json"].write_text(render_json(state, generated_at), encoding="utf-8")
            reports["html"].write_text(render_html(state, generated_at), encoding="utf-8")
            self.tasks.complete(task_id, reports)
        except Exception as exc:  # noqa: BLE001 - task boundary must record failures
            logger.exception("analysis task %s failed", task_id)
            self.tasks.fail(task_id, f"{exc.__class__.__name__}: {exc}")

    def answer(self, task_id: str, answer: str) -> None:
        context = self.tasks.run_context(task_id)
        if context is None:
            return
        self.run(
            task_id,
            context["session_id"],
            context["user_id"],
            context["jd_input"],
            context["resume_input"],
            answer=answer,
        )

    def _materialize_input(self, value: str, target: Path) -> Path:
        uploaded = self.uploads.resolve_input(value)
        if uploaded is not None:
            return uploaded
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(value), encoding="utf-8")
        return target

    def list_input_files(self, kind: str) -> List[InputFileItem]:
        return self.input_files.list(kind)

    def select_input_file(self, kind: str, path: str) -> UploadItem:
        source = self.input_files.resolve(kind, path)
        return self.uploads.save_path(source)

    def resume_options(self) -> Dict[str, object]:
        return {"levels": level_options(), "styles": style_options()}

    def polish_resume(
        self,
        *,
        level: str,
        style: str,
        task_id: str = "",
        resume_input: str = "",
        enable_polish: bool = True,
    ) -> Dict[str, object]:
        source, source_file = self._resume_source(task_id=task_id, resume_input=resume_input)
        text = source.read_text(encoding="utf-8")
        result = polish_resume(
            text,
            level=level,
            style=style,
            source_file=source_file,
            text_client=self._resume_text_client() if enable_polish else None,
            enable_polish=enable_polish,
        )
        resume_id = uuid.uuid4().hex
        self.resume_output_dir.mkdir(parents=True, exist_ok=True)
        (self.resume_output_dir / f"{resume_id}.md").write_text(result.markdown, encoding="utf-8")
        (self.resume_output_dir / f"{resume_id}.html").write_text(result.html, encoding="utf-8")
        level_meta = next((item for item in level_options() if item["key"] == result.level), {})
        style_meta = next((item for item in style_options() if item["key"] == result.style), {})
        return {
            "resume_id": resume_id,
            "task_id": task_id or "",
            "level": result.level,
            "level_name": level_meta.get("name", result.level),
            "style": result.style,
            "style_name": style_meta.get("name", result.style),
            "markdown": result.markdown,
            "html": result.html,
            "items": [item.__dict__ for item in result.items],
            "stats": result.stats.__dict__,
            "notice": result.notice,
        }

    def resume_path(self, resume_id: str, report_format: str) -> Optional[Path]:
        safe = "".join(ch for ch in str(resume_id or "") if ch.isalnum() or ch in "-_-")
        if not safe:
            return None
        candidate = self.resume_output_dir / f"{safe}.{report_format}"
        return candidate if candidate.is_file() else None

    def _resume_source(self, *, task_id: str, resume_input: str) -> tuple[Path, str]:
        if task_id:
            task_inputs = self.tasks.inputs(task_id)
            tracked = task_inputs.get("resume_path")
            if tracked and Path(tracked).is_file():
                path = Path(tracked)
                return path, path.name
            raise FileNotFoundError("分析任务没有可用的简历原文")
        if resume_input:
            uploaded = self.uploads.resolve_input(resume_input)
            if uploaded is not None:
                return uploaded, uploaded.name
            target = self.data_dir / "tmp" / "resume_inputs" / f"{uuid.uuid4().hex}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(resume_input), encoding="utf-8")
            return target, "resume.md"
        raise ValueError("请提供 task_id 或 resume_input")

    def _resume_text_client(self):
        if self.resume_text_client_factory is not None:
            return self.resume_text_client_factory()
        loaded = load_env(DEFAULT_ENV_FILE)
        if loaded.error:
            return None
        settings = resolve_settings(
            env_file=loaded.path,
            env_keys=tuple(loaded.applied) + tuple(loaded.skipped),
        )
        return build_client(settings, "text", timeout=DEFAULT_CALL_TIMEOUT)
