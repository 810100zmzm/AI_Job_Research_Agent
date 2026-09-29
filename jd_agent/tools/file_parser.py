"""File-to-Markdown parser used by HTTP uploads and other integrations.

The CLI historically imported :func:`parse_file_markdown` from ``jd_files``.
This module gives those callers a stable filename without duplicating the
format-specific logic.
"""
from __future__ import annotations


from .jd_files import (
    HTML_SUFFIXES,
    IMAGE_SUFFIXES,
    JSON_SUFFIXES,
    KIND_HTML,
    KIND_IMAGE,
    KIND_JSON,
    KIND_TEXT,
    KIND_YAML,
    TEXT_SUFFIXES,
    YAML_SUFFIXES,
    ParsedFile,
    classify_file,
    parse_file_markdown as _parse_file_markdown,
)

SUPPORTED_SUFFIXES = (
    TEXT_SUFFIXES + HTML_SUFFIXES + JSON_SUFFIXES + YAML_SUFFIXES
)


def parse_file_to_markdown(path, strict: bool = False) -> ParsedFile:
    """Parse a supported file and return its Markdown representation.

    Upload content is user supplied, so the default is the less aggressive
    cleanup mode. Callers that want UI noise filtering can pass ``strict=True``.
    """
    return _parse_file_markdown(path, strict=strict)


def parse_file(path, strict: bool = False) -> ParsedFile:
    """Compatibility alias for callers that prefer the shorter name."""
    return parse_file_to_markdown(path, strict=strict)


def detect_file_type(path) -> str:
    """Return the parser kind for a path (``text``, ``html``, ...)."""
    return classify_file(path)


def is_supported_file(path) -> bool:
    """Whether ``path`` has a suffix handled by the parser."""
    return classify_file(path) in (KIND_TEXT, KIND_HTML, KIND_JSON, KIND_YAML)


__all__ = [
    "HTML_SUFFIXES",
    "IMAGE_SUFFIXES",
    "JSON_SUFFIXES",
    "KIND_HTML",
    "KIND_IMAGE",
    "KIND_JSON",
    "KIND_TEXT",
    "KIND_YAML",
    "ParsedFile",
    "SUPPORTED_SUFFIXES",
    "TEXT_SUFFIXES",
    "YAML_SUFFIXES",
    "classify_file",
    "detect_file_type",
    "is_supported_file",
    "parse_file",
    "parse_file_to_markdown",
]
