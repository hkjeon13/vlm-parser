from __future__ import annotations

from pathlib import Path
import tempfile

import fitz
from PIL import Image

from vlm_parser.core.models import (
    DocumentMetadata,
    RenderChunk,
    RenderImage,
    RenderResult,
    StaticUnitResult,
    TrimmedRenderImage,
)
from vlm_parser.documents.base import DocumentHandle, DocumentUnit
from vlm_parser.documents.pdf.static_extractor import extract_page_static
from vlm_parser.image.chunker import ChunkOptions, split_horizontal_chunks
from vlm_parser.image.preprocess import TrimOptions, trim_uniform_margins


class PdfDocumentAdapter:
    document_type = "pdf"
    unit_type = "page"

    def __init__(self, render_dpi: int = 180, trim: bool = True, auto_slice: bool = True):
        self.render_dpi = render_dpi
        self.trim = trim
        self.auto_slice = auto_slice
        self.asset_dir = Path(tempfile.mkdtemp(prefix="vlm-parser-"))

    def open(self, source: str | Path) -> DocumentHandle:
        return DocumentHandle(source=str(source), native=fitz.open(source))

    def close(self, document: DocumentHandle) -> None:
        document.native.close()

    def get_metadata(self, document: DocumentHandle) -> DocumentMetadata:
        metadata = document.native.metadata or {}
        return DocumentMetadata(
            title=metadata.get("title") or None,
            author=metadata.get("author") or None,
            subject=metadata.get("subject") or None,
            keywords=metadata.get("keywords") or None,
            created_at=metadata.get("creationDate") or None,
            modified_at=metadata.get("modDate") or None,
        )

    def iter_units(self, document: DocumentHandle) -> list[DocumentUnit]:
        return [
            DocumentUnit(
                unit_id=f"p{index + 1}",
                unit_type="page",
                unit_number=index + 1,
                native=document.native[index],
            )
            for index in range(document.native.page_count)
        ]

    def extract_static(self, unit: DocumentUnit) -> StaticUnitResult:
        return extract_page_static(unit.native, unit.unit_number)

    def render(self, unit: DocumentUnit) -> RenderResult | None:
        page = unit.native
        pixmap = page.get_pixmap(dpi=self.render_dpi)
        original_path = self.asset_dir / f"page-{unit.unit_number:03d}.png"
        pixmap.save(original_path)

        image = Image.open(original_path).convert("RGB")
        original = RenderImage(
            path=str(original_path),
            width_px=image.width,
            height_px=image.height,
            dpi=self.render_dpi,
        )

        if self.trim:
            trim_result = trim_uniform_margins(image, TrimOptions())
        else:
            trim_result = trim_uniform_margins(image, TrimOptions(edge_content_guard_px=0))
            trim_result = type(trim_result)(image, (0, 0, image.width, image.height), False, "disabled")

        trimmed_path = self.asset_dir / f"page-{unit.unit_number:03d}-trimmed.png"
        trim_result.image.save(trimmed_path)
        trimmed = TrimmedRenderImage(
            path=str(trimmed_path),
            width_px=trim_result.image.width,
            height_px=trim_result.image.height,
            dpi=self.render_dpi,
            bbox_in_original=list(trim_result.bbox_in_original),
            applied=trim_result.applied,
            reason=trim_result.reason,
        )

        if self.auto_slice:
            chunk_boxes = split_horizontal_chunks(trim_result.image, ChunkOptions())
        else:
            chunk_boxes = split_horizontal_chunks(
                trim_result.image,
                ChunkOptions(max_chunk_height_px=max(trim_result.image.height, 1)),
            )

        chunks: list[RenderChunk] = []
        trim_x0, trim_y0, _, _ = trim_result.bbox_in_original
        for chunk_box in chunk_boxes:
            x0, y0, x1, y1 = chunk_box.bbox_in_trimmed
            chunk_path = self.asset_dir / f"page-{unit.unit_number:03d}-chunk-{chunk_box.index + 1:03d}.png"
            trim_result.image.crop(chunk_box.bbox_in_trimmed).save(chunk_path)
            chunks.append(
                RenderChunk(
                    id=f"p{unit.unit_number}-c{chunk_box.index + 1}",
                    index=chunk_box.index,
                    path=str(chunk_path),
                    bbox_in_original=[trim_x0 + x0, trim_y0 + y0, trim_x0 + x1, trim_y0 + y1],
                    bbox_in_trimmed=[x0, y0, x1, y1],
                    split_reason=chunk_box.split_reason,
                    height_px=y1 - y0,
                )
            )

        return RenderResult(original=original, trimmed=trimmed, chunks=chunks)
