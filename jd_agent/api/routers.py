"""HTTP routes for upload, analysis, SSE, reports, and L2 records."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, List, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from ..memory import KIND_PROFILE, L2_LONG_TERM, MemoryRecord
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnswerRequest,
    AnswerResponse,
    DeleteRecordResponse,
    InputFileItem,
    RecordCreate,
    RecordCreatedResponse,
    RecordItem,
    RecordUpdate,
    ResumeOptionsResponse,
    ResumePolishRequest,
    ResumePolishResponse,
    SelectInputFileRequest,
    UploadResponse,
)
from .services import AnalysisService
from .stores import TaskStore

REPORT_MEDIA_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "html": "text/html; charset=utf-8",
}


def build_router(service: AnalysisService) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["job-research"])

    def get_service() -> AnalysisService:
        return service

    @router.post("/upload", response_model=UploadResponse)
    async def upload_file(
        file: UploadFile = File(...),
        app_service: AnalysisService = Depends(get_service),
    ) -> UploadResponse:
        try:
            item = await app_service.uploads.save(file)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UploadResponse(
            file_id=item.file_id,
            filename=item.filename,
            md_content=item.markdown,
            file_type=item.file_type,
        )

    @router.get("/input-files", response_model=List[InputFileItem])
    def list_input_files(
        kind: Literal["jd", "resume"] = Query(...),
        app_service: AnalysisService = Depends(get_service),
    ) -> List[InputFileItem]:
        try:
            return [
                InputFileItem(**item.__dict__)
                for item in app_service.list_input_files(kind)
            ]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/input-files/select", response_model=UploadResponse)
    def select_input_file(
        payload: SelectInputFileRequest,
        app_service: AnalysisService = Depends(get_service),
    ) -> UploadResponse:
        try:
            item = app_service.select_input_file(payload.kind, payload.path)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UploadResponse(
            file_id=item.file_id,
            filename=item.filename,
            md_content=item.markdown,
            file_type=item.file_type,
        )

    @router.post("/analyze", response_model=AnalyzeResponse)
    def analyze(
        payload: AnalyzeRequest,
        background_tasks: BackgroundTasks,
        app_service: AnalysisService = Depends(get_service),
    ) -> AnalyzeResponse:
        task = app_service.tasks.create()
        background_tasks.add_task(
            app_service.run,
            task.task_id,
            payload.session_id,
            payload.user_id,
            payload.jd_input,
            payload.resume_input,
        )
        return AnalyzeResponse(task_id=task.task_id)

    @router.post("/task/{task_id}/answer", response_model=AnswerResponse)
    def answer_task(
        task_id: str,
        payload: AnswerRequest,
        background_tasks: BackgroundTasks,
        app_service: AnalysisService = Depends(get_service),
    ) -> AnswerResponse:
        task = app_service.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        answer = payload.answer.strip()
        if not answer:
            raise HTTPException(status_code=422, detail="回答不能为空")
        if not app_service.tasks.accept_answer(task_id, answer):
            raise HTTPException(status_code=409, detail="该任务当前不在等待回答")
        background_tasks.add_task(app_service.answer, task_id, answer)
        return AnswerResponse(task_id=task_id)

    @router.get("/chat")
    async def chat(
        request: Request,
        task_id: str = Query(...),
        app_service: AnalysisService = Depends(get_service),
    ) -> StreamingResponse:
        if app_service.tasks.get(task_id) is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return StreamingResponse(
            _stream_task(request, task_id, app_service.tasks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/report/{task_id}")
    def get_report(
        task_id: str,
        report_format: Literal["md", "json", "html"] = Query("md", alias="format"),
        app_service: AnalysisService = Depends(get_service),
    ) -> Response:
        path = app_service.report_path(task_id, report_format)
        if path is None:
            raise HTTPException(status_code=404, detail="报告尚未生成")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"读取报告失败：{exc}") from exc
        return Response(content=content, media_type=REPORT_MEDIA_TYPES[report_format])

    @router.get("/resume/options", response_model=ResumeOptionsResponse)
    def resume_options(
        app_service: AnalysisService = Depends(get_service),
    ) -> ResumeOptionsResponse:
        return ResumeOptionsResponse(**app_service.resume_options())

    @router.post("/resume/polish", response_model=ResumePolishResponse)
    def polish_resume(
        payload: ResumePolishRequest,
        app_service: AnalysisService = Depends(get_service),
    ) -> ResumePolishResponse:
        try:
            result = app_service.polish_resume(
                level=payload.level,
                style=payload.style,
                task_id=payload.task_id.strip(),
                resume_input=payload.resume_input.strip(),
                enable_polish=payload.enable_polish,
            )
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ResumePolishResponse(**result)

    @router.get("/resume/{resume_id}")
    def get_generated_resume(
        resume_id: str,
        report_format: Literal["md", "html"] = Query("md", alias="format"),
        app_service: AnalysisService = Depends(get_service),
    ) -> Response:
        path = app_service.resume_path(resume_id, report_format)
        if path is None:
            raise HTTPException(status_code=404, detail="简历尚未生成")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"读取简历失败：{exc}") from exc
        return Response(content=content, media_type=REPORT_MEDIA_TYPES[report_format])

    @router.post("/records", response_model=RecordCreatedResponse)
    def create_record(
        payload: RecordCreate,
        app_service: AnalysisService = Depends(get_service),
    ) -> RecordCreatedResponse:
        metadata = payload.model_dump()
        text = _record_text(payload)
        record = MemoryRecord(
            text=text,
            level=L2_LONG_TERM,
            kind=KIND_PROFILE,
            user_id="default",
            importance=0.8,
            metadata=metadata,
        )
        saved = app_service.memory.long_term.add(record)
        return RecordCreatedResponse(record_id=saved.memory_id)

    @router.get("/records", response_model=List[RecordItem])
    def list_records(
        status: str = Query(""),
        app_service: AnalysisService = Depends(get_service),
    ) -> List[RecordItem]:
        query_status = status.strip()
        records = [
            item
            for item in app_service.memory.long_term.all("default")
            if not query_status or str(item.metadata.get("status") or "") == query_status
        ]
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return [_record_item(item) for item in records]

    @router.put("/records/{record_id}", response_model=RecordItem)
    def update_record(
        record_id: str,
        payload: RecordUpdate,
        app_service: AnalysisService = Depends(get_service),
    ) -> RecordItem:
        record = next(
            (
                item
                for item in app_service.memory.long_term.all("default")
                if item.memory_id == record_id
            ),
            None,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="记录不存在")

        metadata = payload.model_dump()
        record.text = _record_text(payload)
        record.metadata = {**record.metadata, **metadata}
        saved = app_service.memory.long_term.add(record)
        return _record_item(saved)

    @router.delete("/records/{record_id}", response_model=DeleteRecordResponse)
    def delete_record(
        record_id: str,
        app_service: AnalysisService = Depends(get_service),
    ) -> DeleteRecordResponse:
        deleted = app_service.memory.long_term.delete([record_id])
        if not deleted:
            raise HTTPException(status_code=404, detail="记录不存在")
        return DeleteRecordResponse(success=True)

    return router


async def _stream_task(request: Request, task_id: str, tasks: TaskStore) -> AsyncIterator[str]:
    cursor = 0
    while True:
        if await request.is_disconnected():
            return

        for event in tasks.events_after(task_id, cursor):
            payload = json.dumps(event.as_payload(), ensure_ascii=False)
            yield f"data: {payload}\n\n"
            cursor = event.index

        task = tasks.get(task_id)
        if task is None:
            yield _sse_event("error", {"detail": "任务不存在"})
            return
        if task.status == "completed":
            yield _sse_event("done", {"task_id": task_id, "status": task.status})
            return
        if task.status == "failed":
            yield _sse_event("error", {"task_id": task_id, "detail": task.error})
            return
        if task.status == "waiting_for_answer":
            yield _sse_event(
                "ask",
                {
                    "task_id": task_id,
                    "question": task.question,
                    "question_reason": task.question_reason,
                },
            )
            return
        await asyncio.sleep(0.1)


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _record_item(record: MemoryRecord) -> RecordItem:
    metadata = record.metadata
    return RecordItem(
        record_id=record.memory_id,
        company=str(metadata.get("company") or ""),
        position=str(metadata.get("position") or ""),
        location=str(metadata.get("location") or ""),
        status=str(metadata.get("status") or ""),
        match_score=float(metadata.get("match_score") or 0.0),
        note=str(metadata.get("note") or ""),
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=dict(metadata),
    )


def _record_text(payload: RecordCreate) -> str:
    return " | ".join(
        part
        for part in (
            payload.company,
            payload.position,
            payload.location,
            payload.status,
            f"match_score={payload.match_score:g}",
            payload.note,
        )
        if part
    )
