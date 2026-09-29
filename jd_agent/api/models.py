"""Request and response models for the HTTP API."""
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    md_content: str
    file_type: str


class InputFileItem(BaseModel):
    path: str
    name: str
    label: str
    kind: Literal["jd", "resume"]
    file_type: str
    size: int = 0
    modified_at: float = 0.0


class SelectInputFileRequest(BaseModel):
    kind: Literal["jd", "resume"]
    path: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    jd_input: str = Field(min_length=1)
    resume_input: str = Field(min_length=1)
    session_id: str = "default"
    user_id: str = "default"


class AnalyzeResponse(BaseModel):
    task_id: str


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    task_id: str
    status: Literal["running"] = "running"


class ReportResponse(BaseModel):
    task_id: str
    format: Literal["md", "json", "html"]
    content: str


class RecordCreate(BaseModel):
    company: str = ""
    position: str = ""
    location: str = ""
    status: str = ""
    match_score: float = 0.0
    note: str = ""


class RecordUpdate(RecordCreate):
    """Full replacement payload for an existing application record."""


class RecordCreatedResponse(BaseModel):
    record_id: str


class RecordItem(BaseModel):
    record_id: str
    company: str = ""
    position: str = ""
    location: str = ""
    status: str = ""
    match_score: float = 0.0
    note: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, object] = Field(default_factory=dict)


class DeleteRecordResponse(BaseModel):
    success: bool = True


class ResumePolishRequest(BaseModel):
    task_id: str = ""
    resume_input: str = ""
    level: Literal["L1", "L2", "L3"] = "L2"
    style: Literal["classic", "structure", "accent"] = "classic"
    enable_polish: bool = True


class ResumePolishItem(BaseModel):
    ref: str
    original: str
    polished: str
    source: str = "llm-polish"
    engine: str = "original"
    status: str = "保留原文"
    reason: str = ""


class ResumePolishStats(BaseModel):
    total: int = 0
    dictionary: int = 0
    llm: int = 0
    original: int = 0
    blocked: int = 0
    calls: int = 0


class ResumePolishResponse(BaseModel):
    resume_id: str
    task_id: str = ""
    level: str
    level_name: str
    style: str
    style_name: str
    markdown: str
    html: str
    items: List[ResumePolishItem] = Field(default_factory=list)
    stats: ResumePolishStats = Field(default_factory=ResumePolishStats)
    notice: str = ""


class ResumeLevelOption(BaseModel):
    key: str
    name: str
    summary: str


class ResumeStyleOption(BaseModel):
    key: str
    name: str
    summary: str
    label: str


class ResumeOptionsResponse(BaseModel):
    levels: List[ResumeLevelOption] = Field(default_factory=list)
    styles: List[ResumeStyleOption] = Field(default_factory=list)

class ErrorResponse(BaseModel):
    detail: str


class TracePayload(BaseModel):
    action: str
    observation: str
    state_update: str
    decision: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    error: str = ""
    trace_count: int = 0
    reports: Dict[str, str] = Field(default_factory=dict)
    available_reports: List[str] = Field(default_factory=list)
