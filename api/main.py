"""Uvicorn entrypoint: ``uvicorn api.main:app --reload``."""

from jd_agent.api.app import app

__all__ = ["app"]
