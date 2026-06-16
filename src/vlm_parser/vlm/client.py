from __future__ import annotations

import base64
import json
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
    max_json_retries: int = 2

    def rewrite_chunk(self, request: object) -> VlmClientResponse:
        client = self.http_client or httpx.Client()
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        raw_usages = []
        retry_error = ""
        for attempt in range(self.max_json_retries + 1):
            response = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=self._payload(request, retry_error=retry_error),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            raw_usages.append(usage)
            _add_usage_totals(usage_totals, usage)
            content = data["choices"][0]["message"]["content"]
            try:
                markdown = _markdown_from_json_content(content)
            except ValueError as exc:
                retry_error = str(exc)
                if attempt < self.max_json_retries:
                    continue
                raise ValueError(
                    "VLM response must be a valid JSON object with a string text field"
                ) from exc
            return VlmClientResponse(
                markdown=markdown,
                prompt_tokens=usage_totals["prompt_tokens"],
                completion_tokens=usage_totals["completion_tokens"],
                total_tokens=usage_totals["total_tokens"],
                reasoning_tokens=usage_totals["reasoning_tokens"],
                raw_usage={"attempts": raw_usages},
            )
        raise RuntimeError("unreachable")

    def _payload(self, request: object, *, retry_error: str = "") -> dict:
        chunk = request.chunk
        image_bytes = Path(chunk.path).read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "Rewrite the visible document chunk as Markdown. "
            "Use the static text as reference, preserve tables and captions, "
            "and do not invent content. "
            'Return only valid JSON in exactly this shape: {"text": "..."}; '
            "put the rewritten Markdown string in text and do not include any prose outside JSON.\n\n"
            f"Unit: {request.unit_id}\n"
            f"Chunk: {chunk.id}\n"
            f"Previous Markdown:\n{request.previous_markdown}\n\n"
            f"Static text:\n{request.static.text}"
        )
        if retry_error:
            prompt += (
                "\n\nPrevious response was invalid JSON. "
                f"Validation error: {retry_error}. "
                'Retry with only {"text": "..."} and no other text.'
            )
        payload = {
            "model": request.model or self.model,
            "response_format": _json_response_format(),
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


def _json_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vlm_chunk_rewrite",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Rewritten Markdown for the visible document chunk.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }


def _markdown_from_json_content(content: str) -> str:
    try:
        data = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError("response content is not valid JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("response JSON must contain a string text field")
    return data["text"].strip()


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().lower() in {"```json", "```"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _add_usage_totals(totals: dict[str, int], usage: dict) -> None:
    totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    totals["reasoning_tokens"] += _reasoning_tokens_from_usage(usage)
