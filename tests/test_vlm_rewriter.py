from vlm_parser.core.models import RenderChunk, StaticUnitResult
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


class FakeHttpClient:
    def __init__(self):
        self.request = None

    def post(self, url, headers, json, timeout):
        self.request = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        return FakeResponse()


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "rewritten markdown"}}]}


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

    markdown = client.rewrite_chunk(request)

    assert markdown == "rewritten markdown"
    assert http_client.request["url"] == "https://api.example.com/v1/chat/completions"
    assert http_client.request["headers"]["Authorization"] == "Bearer key"
    assert http_client.request["json"]["model"] == "vision-model"
    assert http_client.request["json"]["messages"][0]["content"][1]["type"] == "image_url"
