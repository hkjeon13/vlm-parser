from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from vlm_parser.core.models import (
    DocumentResult,
    PageResult,
    ParseResult,
    ParserInfo,
    SourceInfo,
)
from vlm_parser.core.options import ParseOptions, VlmOptions
from vlm_parser.documents.base import DocumentAdapter
from vlm_parser.vlm.concurrency import GlobalVlmLimiter
from vlm_parser.vlm.rewriter import VlmRewriter


class Parser:
    def __init__(
        self,
        adapter: DocumentAdapter,
        options: ParseOptions,
        vlm: VlmOptions,
        vlm_client: Any = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ):
        self.adapter = adapter
        self.options = options
        self.vlm = vlm
        self.vlm_client = vlm_client
        self.progress_callback = progress_callback

    def parse(self, source: str | Path) -> ParseResult:
        source_path = Path(source)
        document = self.adapter.open(source_path)
        try:
            units = list(self.adapter.iter_units(document))
            metadata = self.adapter.get_metadata(document)
            pages: list[PageResult] = []
            total_units = len(units)
            if self.progress_callback is not None:
                self.progress_callback(0, total_units, f"Preparing {total_units} pages")

            for unit in units:
                static = self.adapter.extract_static(unit)
                page = unit.native
                render = self.adapter.render(unit)
                markdown = static.text.strip()
                vlm_result = None
                if self.vlm.enabled and self.vlm_client is not None and render is not None:
                    rewriter = VlmRewriter(
                        client=self.vlm_client,
                        limiter=GlobalVlmLimiter(self.vlm.max_concurrency),
                        model=self.vlm.model or "",
                    )
                    vlm_result = rewriter.rewrite_unit(
                        unit_id=unit.unit_id,
                        static=static,
                        chunks=render.chunks,
                    )
                    markdown = vlm_result.markdown
                pages.append(
                    PageResult(
                        unit_id=unit.unit_id,
                        unit_type=unit.unit_type,
                        unit_number=unit.unit_number,
                        page_number=unit.unit_number,
                        width_pt=float(page.rect.width),
                        height_pt=float(page.rect.height),
                        rotation=int(page.rotation),
                        static=static,
                        render=render,
                        vlm=vlm_result,
                        markdown=markdown,
                    )
                )
                if self.progress_callback is not None:
                    self.progress_callback(
                        len(pages),
                        total_units,
                        f"Parsed page {len(pages)} of {total_units}",
                    )

            document_markdown = "\n\n".join(page.markdown for page in pages if page.markdown)
            return ParseResult(
                source=SourceInfo(
                    path=str(source_path),
                    filename=source_path.name,
                    file_size_bytes=source_path.stat().st_size,
                    unit_count=len(units),
                    page_count=len(units),
                    document_type=self.adapter.document_type,
                    parser=ParserInfo(),
                ),
                options={
                    "static_engine": "pymupdf",
                    "render_dpi": self.options.render_dpi,
                    "trim_enabled": self.options.trim,
                    "auto_slice_enabled": self.options.auto_slice,
                    "vlm_enabled": self.vlm.enabled,
                    "vlm_model": self.vlm.model,
                },
                document=DocumentResult(markdown=document_markdown, metadata=metadata),
                pages=pages,
            )
        finally:
            self.adapter.close(document)
