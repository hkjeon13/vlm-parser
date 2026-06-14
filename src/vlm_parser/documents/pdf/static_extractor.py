from __future__ import annotations

from typing import Any

from vlm_parser.core.models import StaticBlock, StaticLine, StaticSpan, StaticUnitResult


def _color_to_hex(color: int | None) -> str | None:
    if color is None:
        return None
    return f"#{color:06x}"


def extract_page_static(page: Any, page_number: int) -> StaticUnitResult:
    text = page.get_text("text")
    raw = page.get_text("dict")
    blocks: list[StaticBlock] = []

    for block_index, block in enumerate(raw.get("blocks", []), start=1):
        block_type = "text" if block.get("type") == 0 else "image"
        lines: list[StaticLine] = []
        block_text_parts: list[str] = []

        for line in block.get("lines", []):
            spans: list[StaticSpan] = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                block_text_parts.append(span_text)
                spans.append(
                    StaticSpan(
                        text=span_text,
                        bbox=[float(v) for v in span.get("bbox", [])],
                        font=span.get("font"),
                        size=span.get("size"),
                        flags=span.get("flags"),
                        color=_color_to_hex(span.get("color")),
                    )
                )
            lines.append(
                StaticLine(
                    bbox=[float(v) for v in line.get("bbox", [])],
                    spans=spans,
                )
            )

        blocks.append(
            StaticBlock(
                id=f"p{page_number}-b{block_index}",
                type=block_type,
                bbox=[float(v) for v in block.get("bbox", [])],
                text="".join(block_text_parts).strip(),
                lines=lines,
            )
        )

    return StaticUnitResult(text=text, blocks=blocks, images=[])
