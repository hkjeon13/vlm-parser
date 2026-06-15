from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any, Literal


def _to_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _to_json_value(v) for k, v in value.items()}
    return value


@dataclass(slots=True)
class ParserInfo:
    name: str = "vlm-parser"
    version: str = "0.1.0"


@dataclass(slots=True)
class SourceInfo:
    path: str
    filename: str
    file_size_bytes: int
    unit_count: int
    page_count: int
    document_type: str
    parser: ParserInfo = field(default_factory=ParserInfo)


@dataclass(slots=True)
class DocumentMetadata:
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    created_at: str | None = None
    modified_at: str | None = None


@dataclass(slots=True)
class DocumentResult:
    markdown: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


@dataclass(slots=True)
class StaticSpan:
    text: str
    bbox: list[float] = field(default_factory=list)
    font: str | None = None
    size: float | None = None
    flags: int | None = None
    color: str | None = None


@dataclass(slots=True)
class StaticLine:
    bbox: list[float] = field(default_factory=list)
    spans: list[StaticSpan] = field(default_factory=list)


@dataclass(slots=True)
class StaticBlock:
    id: str
    type: str
    bbox: list[float] = field(default_factory=list)
    text: str = ""
    lines: list[StaticLine] = field(default_factory=list)


@dataclass(slots=True)
class StaticImage:
    id: str
    bbox: list[float] = field(default_factory=list)
    xref: int | None = None
    width_px: int | None = None
    height_px: int | None = None
    colorspace: str | None = None
    ext: str | None = None
    path: str | None = None


@dataclass(slots=True)
class StaticUnitResult:
    text: str = ""
    blocks: list[StaticBlock] = field(default_factory=list)
    images: list[StaticImage] = field(default_factory=list)


@dataclass(slots=True)
class RenderImage:
    path: str
    width_px: int
    height_px: int
    dpi: int | None = None


@dataclass(slots=True)
class TrimmedRenderImage(RenderImage):
    bbox_in_original: list[int] = field(default_factory=list)
    applied: bool = False
    reason: str = "not_applied"


@dataclass(slots=True)
class RenderChunk:
    id: str
    index: int
    path: str
    bbox_in_original: list[int]
    bbox_in_trimmed: list[int]
    split_reason: str
    height_px: int


@dataclass(slots=True)
class RenderResult:
    original: RenderImage
    trimmed: TrimmedRenderImage
    chunks: list[RenderChunk] = field(default_factory=list)


@dataclass(slots=True)
class VlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(slots=True)
class VlmClientResponse:
    markdown: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    raw_usage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VlmChunkResult:
    chunk_id: str
    status: str
    markdown: str
    usage: VlmUsage = field(default_factory=VlmUsage)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VlmUnitResult:
    enabled: bool
    status: Literal["success", "partial", "failed", "skipped"]
    model: str | None
    chunks: list[VlmChunkResult] = field(default_factory=list)
    markdown: str = ""
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class PageMetrics:
    parse_seconds: float = 0.0


@dataclass(slots=True)
class ParseMetrics:
    total_seconds: float = 0.0
    page_count: int = 0
    average_seconds_per_page: float = 0.0
    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(slots=True)
class PageResult:
    unit_id: str
    unit_number: int
    page_number: int
    width_pt: float
    height_pt: float
    rotation: int
    static: StaticUnitResult
    markdown: str
    unit_type: str = "page"
    render: RenderResult | None = None
    vlm: VlmUnitResult | None = None
    metrics: PageMetrics = field(default_factory=PageMetrics)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParseResult:
    source: SourceInfo
    document: DocumentResult
    pages: list[PageResult]
    schema_version: str = "0.1"
    options: dict[str, Any] = field(default_factory=dict)
    metrics: ParseMetrics = field(default_factory=ParseMetrics)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return _to_json_value(self)

    def to_markdown(self) -> str:
        return self.document.markdown

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_markdown(self, path: str | Path) -> None:
        Path(path).write_text(self.to_markdown(), encoding="utf-8")
