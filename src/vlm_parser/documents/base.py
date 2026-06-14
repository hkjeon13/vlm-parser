from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from vlm_parser.core.models import DocumentMetadata, RenderResult, StaticUnitResult


@dataclass(slots=True)
class DocumentHandle:
    source: str
    native: Any


@dataclass(slots=True)
class DocumentUnit:
    unit_id: str
    unit_type: str
    unit_number: int
    native: Any


class DocumentAdapter(Protocol):
    document_type: str
    unit_type: str

    def open(self, source: str) -> DocumentHandle:
        ...

    def close(self, document: DocumentHandle) -> None:
        ...

    def get_metadata(self, document: DocumentHandle) -> DocumentMetadata:
        ...

    def iter_units(self, document: DocumentHandle) -> Iterable[DocumentUnit]:
        ...

    def extract_static(self, unit: DocumentUnit) -> StaticUnitResult:
        ...

    def render(self, unit: DocumentUnit) -> RenderResult | None:
        ...
