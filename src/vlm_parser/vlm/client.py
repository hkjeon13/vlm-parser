from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from vlm_parser.core.models import VlmClientResponse


class VlmClient(Protocol):
    def rewrite_chunk(self, request: object) -> str | VlmClientResponse:
        """Return Markdown for one chunk request."""


@dataclass(slots=True)
class OpenAICompatibleVlmClient:
    base_url: str
    api_key: str
    model: str
    http_client: object | None = None
    timeout_seconds: float = 60.0
    reasoning_effort: str = "auto"

    def rewrite_chunk(self, request: object) -> VlmClientResponse:
        client = self.http_client or httpx.Client()
        response = client.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self._payload(request),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return VlmClientResponse(
            markdown=data["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            reasoning_tokens=_reasoning_tokens_from_usage(usage),
            raw_usage=usage,
        )

    def _payload(self, request: object) -> dict:
        chunk = request.chunk
        image_bytes = Path(chunk.path).read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "Rewrite the visible document chunk as Markdown. "
            "Use the static text as reference, preserve tables and captions, "
            "and do not invent content.\n\n"
            f"Unit: {request.unit_id}\n"
            f"Chunk: {chunk.id}\n"
            f"Previous Markdown:\n{request.previous_markdown}\n\n"
            f"Static text:\n{request.static.text}"
        )
        payload = {
            "model": request.model or self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded}",
                            },
                        },
                    ],
                }
            ],
        }
        reasoning = _reasoning_payload(self.reasoning_effort)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        return payload


def _reasoning_payload(reasoning_effort: str) -> dict[str, str] | None:
    if reasoning_effort == "auto":
        return None
    if reasoning_effort == "off":
        return {"effort": "none"}
    if reasoning_effort in {"low", "medium", "high"}:
        return {"effort": reasoning_effort}
    return None


def _reasoning_tokens_from_usage(usage: dict) -> int:
    details = usage.get("completion_tokens_details") or {}
    if isinstance(details, dict):
        return int(
            details.get("reasoning_tokens")
            or details.get("internal_reasoning_tokens")
            or 0
        )
    return int(usage.get("reasoning_tokens") or usage.get("internal_reasoning_tokens") or 0)
