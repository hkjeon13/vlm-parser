from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParseOptions:
    render_dpi: int = 180
    trim: bool = True
    auto_slice: bool = True
    max_page_workers: int = 4


@dataclass(slots=True)
class VlmOptions:
    enabled: bool = False
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    max_concurrency: int = 4
    timeout_seconds: float = 60.0
    max_retries: int = 2
