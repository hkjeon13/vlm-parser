from vlm_parser.core.models import RenderChunk, StaticUnitResult, VlmClientResponse
from vlm_parser.vlm.client import OpenAICompatibleVlmClient
from vlm_parser.vlm.concurrency import GlobalVlmLimiter
from vlm_parser.vlm.rewriter import VlmChunkRequest, VlmRewriter


class RecordingClient:
    def __init__(self):
        self.requests: list[VlmChunkRequest] = []

    def rewrite_chunk(self, request: VlmChunkRequest) -> str:
        self.requests.append(request)
        return f"markdown-{request.chunk.id}"


def test_vlm_rewriter_processes_chunks_sequentially_with_previous_context():
    client = RecordingClient()
    rewriter = VlmRewriter(client=client, limiter=GlobalVlmLimiter(max_concurrency=1), model="model-a")
    chunks = [
        RenderChunk("c1", 0, "c1.png", [0, 0, 100, 50], [0, 0, 100, 50], "start", 50),
        RenderChunk("c2", 1, "c2.png", [0, 50, 100, 100], [0, 50, 100, 100], "end", 50),
    ]

    result = rewriter.rewrite_unit(
        unit_id="p1",
        static=StaticUnitResult(text="raw text"),
        chunks=chunks,
    )

    assert [request.chunk.id for request in client.requests] == ["c1", "c2"]
    assert client.requests[0].previous_markdown == ""
    assert client.requests[1].previous_markdown == "markdown-c1"
    assert result.status == "success"
    assert result.markdown == "markdown-c1\n\nmarkdown-c2"


class UsageRecordingClient:
    def rewrite_chunk(self, request: VlmChunkRequest) -> VlmClientResponse:
        return VlmClientResponse(
            markdown=f"markdown-{request.chunk.id}",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reasoning_tokens=2,
        )


def test_vlm_rewriter_preserves_usage_from_client_response():
    rewriter = VlmRewriter(
        client=UsageRecordingClient(),
        limiter=GlobalVlmLimiter(max_concurrency=1),
        model="model-a",
    )
    chunks = [
        RenderChunk("c1", 0, "c1.png", [0, 0, 100, 50], [0, 0, 100, 50], "end", 50),
    ]

    result = rewriter.rewrite_unit(
        unit_id="p1",
        static=StaticUnitResult(text="raw text"),
        chunks=chunks,
    )

    assert result.chunks[0].usage.prompt_tokens == 10
    assert result.chunks[0].usage.completion_tokens == 5
    assert result.chunks[0].usage.total_tokens == 15
    assert result.chunks[0].usage.reasoning_tokens == 2


class FakeHttpClient:
    def __init__(self, responses=None):
        self.request = None
        self.requests = []
        self.responses = list(responses or [FakeResponse()])

    def post(self, url, headers, json, timeout):
        self.request = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        self.requests.append(self.request)
        return self.responses.pop(0)


class FakeResponse:
    def __init__(self, content='{"text": "rewritten markdown"}'):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": self.content}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 125,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }


def test_openai_compatible_client_posts_chat_completion_payload(tmp_path):
    image_path = tmp_path / "chunk.png"
    image_path.write_bytes(b"fake-image")
    http_client = FakeHttpClient()
    client = OpenAICompatibleVlmClient(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="vision-model",
        http_client=http_client,
        timeout_seconds=10,
    )
    request = VlmChunkRequest(
        unit_id="p1",
        chunk=RenderChunk("c1", 0, str(image_path), [0, 0, 10, 10], [0, 0, 10, 10], "end", 10),
        static=StaticUnitResult(text="raw text"),
        previous_markdown="previous",
        model="vision-model",
    )

    response = client.rewrite_chunk(request)

    assert response.markdown == "rewritten markdown"
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20
    assert response.total_tokens == 125
    assert response.reasoning_tokens == 5
    assert http_client.request["url"] == "https://api.example.com/v1/chat/completions"
    assert http_client.request["headers"]["Authorization"] == "Bearer key"
    assert http_client.request["json"]["model"] == "vision-model"
    assert http_client.request["json"]["messages"][0]["content"][1]["type"] == "image_url"
    assert http_client.request["json"]["response_format"]["type"] == "json_schema"
    assert http_client.request["json"]["response_format"]["json_schema"]["schema"]["required"] == ["text"]
    assert "Return only valid JSON" in http_client.request["json"]["messages"][0]["content"][0]["text"]
    assert "reasoning" not in http_client.request["json"]


def test_openai_compatible_client_retries_until_json_text_response(tmp_path):
    image_path = tmp_path / "chunk.png"
    image_path.write_bytes(b"fake-image")
    http_client = FakeHttpClient(
        responses=[
            FakeResponse("Here is the rewritten document:\n\n# Bad"),
            FakeResponse('{"text": "# Clean"}'),
        ]
    )
    client = OpenAICompatibleVlmClient(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="vision-model",
        http_client=http_client,
    )
    request = VlmChunkRequest(
        unit_id="p1",
        chunk=RenderChunk("c1", 0, str(image_path), [0, 0, 10, 10], [0, 0, 10, 10], "end", 10),
        static=StaticUnitResult(text="raw text"),
        previous_markdown="previous",
        model="vision-model",
    )

    response = client.rewrite_chunk(request)

    assert response.markdown == "# Clean"
    assert len(http_client.requests) == 2
    assert "Previous response was invalid JSON" in http_client.requests[1]["json"]["messages"][0]["content"][0]["text"]


def test_openai_compatible_client_rejects_json_without_text(tmp_path):
    image_path = tmp_path / "chunk.png"
    image_path.write_bytes(b"fake-image")
    http_client = FakeHttpClient(
        responses=[
            FakeResponse('{"markdown": "# Bad"}'),
            FakeResponse('{"markdown": "# Still bad"}'),
            FakeResponse('{"markdown": "# Nope"}'),
        ]
    )
    client = OpenAICompatibleVlmClient(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="vision-model",
        http_client=http_client,
    )
    request = VlmChunkRequest(
        unit_id="p1",
        chunk=RenderChunk("c1", 0, str(image_path), [0, 0, 10, 10], [0, 0, 10, 10], "end", 10),
        static=StaticUnitResult(text="raw text"),
        previous_markdown="previous",
        model="vision-model",
    )

    try:
        client.rewrite_chunk(request)
    except ValueError as exc:
        assert "valid JSON object with a string text field" in str(exc)
    else:
        raise AssertionError("Expected invalid JSON response to fail")


def test_openai_compatible_client_adds_reasoning_effort_when_configured(tmp_path):
    image_path = tmp_path / "chunk.png"
    image_path.write_bytes(b"fake-image")
    http_client = FakeHttpClient()
    client = OpenAICompatibleVlmClient(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="vision-model",
        http_client=http_client,
        reasoning_effort="low",
    )
    request = VlmChunkRequest(
        unit_id="p1",
        chunk=RenderChunk("c1", 0, str(image_path), [0, 0, 10, 10], [0, 0, 10, 10], "end", 10),
        static=StaticUnitResult(text="raw text"),
        previous_markdown="previous",
        model="vision-model",
    )

    client.rewrite_chunk(request)

    assert http_client.request["json"]["reasoning"] == {"effort": "low"}


def test_openai_compatible_client_maps_reasoning_off_to_none(tmp_path):
    image_path = tmp_path / "chunk.png"
    image_path.write_bytes(b"fake-image")
    http_client = FakeHttpClient()
    client = OpenAICompatibleVlmClient(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="vision-model",
        http_client=http_client,
        reasoning_effort="off",
    )
    request = VlmChunkRequest(
        unit_id="p1",
        chunk=RenderChunk("c1", 0, str(image_path), [0, 0, 10, 10], [0, 0, 10, 10], "end", 10),
        static=StaticUnitResult(text="raw text"),
        previous_markdown="previous",
        model="vision-model",
    )

    client.rewrite_chunk(request)

    assert http_client.request["json"]["reasoning"] == {"effort": "none"}
