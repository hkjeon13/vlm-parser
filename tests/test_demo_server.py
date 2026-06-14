from pathlib import Path

from demo.server import (
    DemoConfig,
    JobOptions,
    JobStore,
    UploadedFile,
    build_vlm_client,
    load_demo_config,
    normalize_model_base_url,
    process_job,
    render_page,
)


def test_load_demo_config_reads_model_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MODEL_API_KEY=secret-key",
                "MODEL_NAME=qwen/qwen3.7-plus",
                "MODEL_BASE_URL=https://openrouter.ai/api/v1/chat/completions",
            ]
        ),
        encoding="utf-8",
    )

    config = load_demo_config(env_file)

    assert config.api_key == "secret-key"
    assert config.model == "qwen/qwen3.7-plus"
    assert config.base_url == "https://openrouter.ai/api/v1/chat/completions"


def test_normalize_model_base_url_accepts_chat_completions_endpoint():
    normalized = normalize_model_base_url(
        "https://openrouter.ai/api/v1/chat/completions"
    )

    assert normalized == "https://openrouter.ai/api/v1"


def test_build_vlm_client_returns_none_when_config_is_incomplete():
    config = DemoConfig(api_key="", model="qwen/qwen3.7-plus", base_url="")

    assert build_vlm_client(config) is None


def test_job_store_creates_queued_job_with_uploaded_pdf(tmp_path: Path):
    store = JobStore(tmp_path)

    job = store.create(
        UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"),
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True),
    )

    assert job.status == "queued"
    assert job.filename == "sample.pdf"
    assert job.source_path.read_bytes() == b"%PDF-1.7"
    assert store.get(job.id) == job
    assert store.list()[0].id == job.id
    assert store.to_summary(job)["status"] == "queued"


def test_process_job_stores_json_and_markdown_results(tmp_path: Path):
    store = JobStore(tmp_path)
    job = store.create(
        UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"),
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True),
    )

    class FakeResult:
        def to_json(self):
            return {"document": {"markdown": "# Parsed"}}

        def to_markdown(self):
            return "# Parsed"

    class FakeParser:
        def parse(self, source):
            assert Path(source) == job.source_path
            return FakeResult()

    def parser_factory(*, use_vlm, render_dpi, trim, auto_slice, config):
        assert use_vlm is False
        assert render_dpi == 180
        assert trim is True
        assert auto_slice is True
        return FakeParser()

    process_job(
        job.id,
        store=store,
        config=DemoConfig(),
        parser_factory=parser_factory,
    )

    completed = store.get(job.id)
    assert completed is not None
    assert completed.status == "done"
    assert completed.result_json == {"document": {"markdown": "# Parsed"}}
    assert completed.markdown == "# Parsed"


def test_render_page_uses_three_pane_review_layout():
    html = render_page(config=DemoConfig(model="qwen/qwen3.7-plus"))

    assert 'class="app-shell"' in html
    assert "left-rail" in html
    assert 'class="pdf-stage"' in html
    assert "result-panel" in html
    assert "미리보기" in html
    assert "HTML" in html
    assert "JSON" in html
