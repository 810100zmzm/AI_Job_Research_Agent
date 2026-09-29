"""ASGI module alias: ``uvicorn jd_agent.api.main:app``."""
from .app import app, create_app

__all__ = ["app", "create_app"]
