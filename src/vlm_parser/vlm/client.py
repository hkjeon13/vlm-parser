from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx


class VlmClient(Protocol):
    def rewrite_chunk(self, request: object) -> str:
        """Return Markdown for one chunk request."""


@dataclass(slots=True)
class OpenAICompatibleVlmClient:
    base_url: str
    api_key: str
    model: str
    http_client: object | None = None
    timeout_seconds: float = 60.0

    def rewrite_chunk(self, request: object) -> str:
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
        return data["choices"][0]["message"]["content"]

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
        return {
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
