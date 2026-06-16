from __future__ import annotations

import threading
import time

from vlm_parser.core.models import (
    DocumentMetadata,
    RenderChunk,
    RenderImage,
    RenderResult,
    StaticUnitResult,
    TrimmedRenderImage,
)
from vlm_parser.core.options import ParseOptions, VlmOptions
from vlm_parser.core.parser import Parser
from vlm_parser.documents.base import DocumentHandle, DocumentUnit


class FakePage:
    rect = type("Rect", (), {"width": 100, "height": 200})()
    rotation = 0


class FakeAdapter:
    document_type = "pdf"
    unit_type = "page"

    def open(self, source):
        return DocumentHandle(source=str(source), native=object())

    def close(self, document):
        return None

    def get_metadata(self, document):
        return DocumentMetadata()

    def iter_units(self, document):
        return [
            DocumentUnit(unit_id="p1", unit_type="page", unit_number=1, native=FakePage()),
            DocumentUnit(unit_id="p2", unit_type="page", unit_number=2, native=FakePage()),
        ]

    def extract_static(self, unit):
        return StaticUnitResult(text=f"page {unit.unit_number}")

    def render(self, unit):
        image = RenderImage(path=f"page-{unit.unit_number}.png", width_px=100, height_px=100)
        chunk = RenderChunk(
            id=f"p{unit.unit_number}-c1",
            index=0,
            path=f"page-{unit.unit_number}-chunk.png",
            bbox_in_original=[0, 0, 100, 100],
            bbox_in_trimmed=[0, 0, 100, 100],
            split_reason="end",
            height_px=100,
        )
        return RenderResult(
            original=image,
            trimmed=TrimmedRenderImage(path=image.path, width_px=100, height_px=100),
            chunks=[chunk],
        )


class RecordingSlowVlmClient:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def rewrite_chunk(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.15)
            return f"rewritten {request.static.text}"
        finally:
            with self.lock:
                self.active -= 1


def test_parser_rewrites_multiple_pages_in_parallel(tmp_path):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.7")
    client = RecordingSlowVlmClient()
    parser = Parser(
        adapter=FakeAdapter(),
        options=ParseOptions(max_page_workers=2),
        vlm=VlmOptions(enabled=True, model="fake-model", max_concurrency=2),
        vlm_client=client,
    )

    result = parser.parse(source)

    assert client.max_active == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.to_markdown() == "rewritten page 1\n\nrewritten page 2"
