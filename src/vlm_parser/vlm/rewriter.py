from __future__ import annotations

from dataclasses import dataclass

from vlm_parser.core.models import (
    RenderChunk,
    StaticUnitResult,
    VlmClientResponse,
    VlmChunkResult,
    VlmUsage,
    VlmUnitResult,
)
from vlm_parser.vlm.client import VlmClient
from vlm_parser.vlm.concurrency import GlobalVlmLimiter


@dataclass(frozen=True, slots=True)
class VlmChunkRequest:
    unit_id: str
    chunk: RenderChunk
    static: StaticUnitResult
    previous_markdown: str
    model: str


@dataclass(slots=True)
class VlmRewriter:
    client: VlmClient
    limiter: GlobalVlmLimiter
    model: str

    def rewrite_unit(
        self,
        unit_id: str,
        static: StaticUnitResult,
        chunks: list[RenderChunk],
    ) -> VlmUnitResult:
        import time

        started_at = time.perf_counter()
        previous_markdown = ""
        results: list[VlmChunkResult] = []

        for chunk in chunks:
            request = VlmChunkRequest(
                unit_id=unit_id,
                chunk=chunk,
                static=static,
                previous_markdown=previous_markdown,
                model=self.model,
            )
            with self.limiter:
                response = self.client.rewrite_chunk(request)
            if isinstance(response, VlmClientResponse):
                markdown = response.markdown
                usage = VlmUsage(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                )
            else:
                markdown = response
                usage = VlmUsage()
            results.append(
                VlmChunkResult(
                    chunk_id=chunk.id,
                    status="success",
                    markdown=markdown,
                    usage=usage,
                )
            )
            previous_markdown = markdown

        page_markdown = "\n\n".join(result.markdown for result in results if result.markdown)
        return VlmUnitResult(
            enabled=True,
            status="success",
            model=self.model,
            chunks=results,
            markdown=page_markdown,
            elapsed_seconds=time.perf_counter() - started_at,
        )
