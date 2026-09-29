"""FastAPI application factory."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from .. import __version__
from ..core.llm import DEFAULT_CALL_TIMEOUT, build_client
from ..core.settings import (
    DEFAULT_ENV_FILE,
    PROJECT_ROOT,
    load_env,
    resolve_settings,
    resolve_storage_settings,
)
from ..memory import build_memory_manager
from ..services.jd_source import image_to_markdown
from .routers import build_router
from .services import AnalysisService
from .stores import TaskStore, UploadStore


class SPAStaticFiles(StaticFiles):
    """Serve built assets and fall back to index.html for Vue history routes."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            parts = Path(path).parts
            reserved = {"api", "docs", "openapi.json", "redoc", "health"}
            is_reserved = bool(parts and parts[0] in reserved)
            is_page = not parts or "." not in Path(path).name
            if exc.status_code == 404 and not is_reserved and is_page:
                return await super().get_response("index.html", scope)
            raise


def create_app(
    *,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    cors_origins: Sequence[str] = ("*",),
    image_transcriber: Optional[Callable[[Path], str]] = None,
    frontend_dir: Optional[Path] = None,
    resume_text_client_factory=None,
) -> FastAPI:
    """Create an app with injectable storage paths for tests and deployments."""
    storage = resolve_storage_settings()
    if data_dir is not None:
        storage = replace(storage, data_dir=Path(data_dir))
    storage.data_dir.mkdir(parents=True, exist_ok=True)

    uploads = UploadStore(
        storage.data_dir / "tmp" / "uploads",
        image_transcriber=image_transcriber or _transcribe_uploaded_image,
    )
    tasks = TaskStore()
    memory = build_memory_manager(
        storage.data_dir,
        short_term_ttl_seconds=storage.l1_ttl_seconds,
        short_term_max_items=storage.l1_max_items,
        long_term_backend=storage.long_term_backend,
        mongodb_uri=storage.mongodb_uri,
        mongodb_database=storage.mongodb_database,
    )
    resolved_output = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
    resolved_input = Path(input_dir) if input_dir is not None else PROJECT_ROOT / "input"
    service = AnalysisService(
        tasks=tasks,
        uploads=uploads,
        memory=memory,
        data_dir=storage.data_dir,
        output_dir=resolved_output,
        input_dir=resolved_input,
        resume_text_client_factory=resume_text_client_factory,
    )

    api = FastAPI(
        title="求职尽调助手 API",
        version=__version__,
        description="JD / 简历分析、Trace 流式输出与求职记录 L2 持久化。",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.include_router(build_router(service))

    @api.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    resolved_frontend = Path(frontend_dir) if frontend_dir else PROJECT_ROOT / "frontend" / "dist"
    if resolved_frontend.is_dir():
        api.mount(
            "/",
            SPAStaticFiles(directory=resolved_frontend, html=True),
            name="frontend",
        )

    return api


def _transcribe_uploaded_image(path: Path) -> str:
    """Transcribe an uploaded JD screenshot with the configured vision model."""
    loaded = load_env(DEFAULT_ENV_FILE)
    if loaded.error:
        raise ValueError(loaded.error)
    settings = resolve_settings(
        env_file=loaded.path,
        env_keys=tuple(loaded.applied) + tuple(loaded.skipped),
    )
    client = build_client(settings, "vision", timeout=DEFAULT_CALL_TIMEOUT)
    if client is None:
        raise ValueError("截图上传需要配置 DASHSCOPE_API_KEY / QWEN_API_KEY")
    return image_to_markdown(
        path,
        client,
        settings.vision_model,
        ref=path.name,
    )


app = create_app()
