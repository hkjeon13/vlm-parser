from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm_parser.core.models import ParseResult
from vlm_parser.core.options import ParseOptions, VlmOptions
from vlm_parser.core.parser import Parser
from vlm_parser.documents.pdf import PdfDocumentAdapter


@dataclass(slots=True)
class PdfParser:
    options: ParseOptions = field(default_factory=ParseOptions)
    vlm: VlmOptions = field(default_factory=VlmOptions)
    vlm_client: Any = None

    def parse(self, source: str | Path) -> ParseResult:
        return Parser(
            adapter=PdfDocumentAdapter(
                render_dpi=self.options.render_dpi,
                trim=self.options.trim,
                auto_slice=self.options.auto_slice,
            ),
            options=self.options,
            vlm=self.vlm,
            vlm_client=self.vlm_client,
        ).parse(source)
