from pathlib import Path
import shutil
import subprocess

from demo.server import (
    DemoConfig,
    JobOptions,
    JobStore,
    OpenRouterModel,
    OpenRouterPricing,
    UploadedFile,
    api_index_payload,
    build_vlm_client,
    calculate_openrouter_cost,
    content_disposition_header,
    fetch_openrouter_models,
    is_openrouter_base_url,
    source_pdf_link,
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


def test_is_openrouter_base_url_only_accepts_openrouter_api_root():
    assert is_openrouter_base_url("https://openrouter.ai/api/v1") is True
    assert is_openrouter_base_url("https://openrouter.ai/api/v1/chat/completions") is True
    assert is_openrouter_base_url("https://api.example.com/v1") is False


class FakeModelsHttpClient:
    def __init__(self):
        self.request = None

    def get(self, url, timeout):
        self.request = {"url": url, "timeout": timeout}
        return FakeModelsResponse()


class FakeModelsResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": [
                {
                    "id": "text/model",
                    "name": "Text Model",
                    "context_length": 4096,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": ["temperature"],
                },
                {
                    "id": "vision/model",
                    "name": "Vision Model",
                    "context_length": 8192,
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000002",
                        "request": "0.0001",
                        "image": "0.0003",
                        "internal_reasoning": "0.000004",
                    },
                    "supported_parameters": ["reasoning", "include_reasoning"],
                },
            ]
        }


def test_fetch_openrouter_models_filters_to_vision_text_models():
    http_client = FakeModelsHttpClient()

    models = fetch_openrouter_models(http_client=http_client)

    assert http_client.request["url"] == "https://openrouter.ai/api/v1/models?output_modalities=text"
    assert [model.id for model in models] == ["vision/model"]
    assert models[0].supports_reasoning is True
    assert models[0].pricing.image == 0.0003


def test_calculate_openrouter_cost_uses_usage_and_pricing():
    result_json = {
        "pages": [
            {
                "vlm": {
                    "chunks": [
                        {
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 50,
                                "reasoning_tokens": 10,
                            }
                        },
                        {
                            "usage": {
                                "prompt_tokens": 25,
                                "completion_tokens": 5,
                                "reasoning_tokens": 0,
                            }
                        },
                    ]
                }
            }
        ]
    }
    model = OpenRouterModel(
        id="vision/model",
        name="Vision Model",
        context_length=8192,
        pricing=OpenRouterPricing(
            prompt=0.01,
            completion=0.02,
            request=0.5,
            image=0.25,
            internal_reasoning=0.03,
        ),
    )

    metrics = calculate_openrouter_cost(result_json, model)

    assert metrics["request_count"] == 2
    assert metrics["prompt_tokens"] == 125
    assert metrics["completion_tokens"] == 55
    assert metrics["reasoning_tokens"] == 10
    assert metrics["total_cost_usd"] == 4.15


def test_build_vlm_client_returns_none_when_config_is_incomplete():
    config = DemoConfig(api_key="", model="qwen/qwen3.7-plus", base_url="")

    assert build_vlm_client(config) is None


def test_job_store_creates_uploaded_job_with_uploaded_pdf(tmp_path: Path):
    store = JobStore(tmp_path)

    file = store.create_file(UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"))
    job = store.create_job(
        file.id,
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True, model=""),
    )

    assert job.status == "uploaded"
    assert job.filename == "sample.pdf"
    assert job.source_path.read_bytes() == b"%PDF-1.7"
    assert store.get(job.id) == job
    assert store.list()[0].id == job.id
    assert store.to_summary(job)["status"] == "uploaded"
    assert store.to_summary(job)["model"] == ""
    assert store.to_summary(job)["links"]["source_pdf"].endswith("/sample.pdf")
    assert store.to_file_summary(file)["latest_job"]["id"] == job.id


def test_job_store_allows_multiple_parse_jobs_for_one_uploaded_file(tmp_path: Path):
    store = JobStore(tmp_path)
    file = store.create_file(UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"))

    static_job = store.create_job(
        file.id,
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True, model=""),
    )
    vlm_job = store.create_job(
        file.id,
        JobOptions(use_vlm=True, render_dpi=180, trim=True, auto_slice=True, model="vision/model"),
    )

    assert static_job.id != vlm_job.id
    assert static_job.source_path == vlm_job.source_path == file.source_path
    assert [job.id for job in store.list_jobs(file.id)] == [vlm_job.id, static_job.id]
    assert store.to_file_summary(file)["job_count"] == 2


def test_job_store_deletes_uploaded_file_and_its_jobs(tmp_path: Path):
    store = JobStore(tmp_path)
    file = store.create_file(UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"))
    job = store.create_job(
        file.id,
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True, model=""),
    )

    assert store.delete_file(file.id) is True

    assert store.get_file(file.id) is None
    assert store.get(job.id) is None
    assert not file.source_path.exists()


def test_job_store_can_mark_uploaded_job_queued(tmp_path: Path):
    store = JobStore(tmp_path)
    file = store.create_file(UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"))
    job = store.create_job(
        file.id,
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True, model=""),
    )

    queued = store.mark_queued(job.id)

    assert queued is not None
    assert queued.status == "queued"


def test_job_store_tracks_parse_progress(tmp_path: Path):
    store = JobStore(tmp_path)
    job = store.create(
        UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"),
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True, model=""),
    )

    updated = store.update_progress(job.id, current=2, total=5, label="Parsed page 2 of 5")

    assert updated is not None
    assert updated.progress_current == 2
    assert updated.progress_total == 5
    assert updated.progress_percent == 40
    summary = store.to_summary(updated)
    assert summary["progress"]["current"] == 2
    assert summary["progress"]["total"] == 5
    assert summary["progress"]["percent"] == 40
    assert summary["progress"]["label"] == "Parsed page 2 of 5"


def test_process_job_stores_json_and_markdown_results(tmp_path: Path):
    store = JobStore(tmp_path)
    file = store.create_file(UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"))
    job = store.create_job(
        file.id,
        JobOptions(use_vlm=False, render_dpi=180, trim=True, auto_slice=True, model=""),
    )

    class FakeResult:
        def to_json(self):
            return {"document": {"markdown": "# Parsed"}}

        def to_markdown(self):
            return "# Parsed"

    class FakeParser:
        def __init__(self, progress_callback):
            self.progress_callback = progress_callback

        def parse(self, source):
            assert Path(source) == job.source_path
            self.progress_callback(1, 2, "Parsed page 1 of 2")
            return FakeResult()

    def parser_factory(*, use_vlm, render_dpi, trim, auto_slice, config, progress_callback=None):
        assert use_vlm is False
        assert render_dpi == 180
        assert trim is True
        assert auto_slice is True
        assert config.model == ""
        assert progress_callback is not None
        return FakeParser(progress_callback)

    process_job(
        job.id,
        store=store,
        config=DemoConfig(),
        parser_factory=parser_factory,
    )

    completed = store.get(job.id)
    assert completed is not None
    assert completed.status == "done"
    assert completed.progress_percent == 100
    assert completed.result_json == {"document": {"markdown": "# Parsed"}}
    assert completed.markdown == "# Parsed"


def test_render_page_uses_document_workspace_layout():
    html = render_page(config=DemoConfig(model="qwen/qwen3.7-plus"))

    assert 'class="app-shell"' in html
    assert "left-rail" in html
    assert 'class="pdf-stage"' in html
    assert "result-panel" in html
    assert "rail-detail" in html
    assert "detail-panel" not in html
    assert "MD" in html
    assert "미리보기" not in html
    assert "JSON" in html
    assert "<h2>파일</h2>" in html
    assert "<h2>선택 파일</h2>" in html
    assert "파일을 드래그하여 업로드" in html
    assert "parse-step" not in html
    assert "pdf-controls" not in html


def test_render_page_includes_openrouter_model_controls():
    html = render_page(
        config=DemoConfig(
            api_key="key",
            model="vision/model",
            base_url="https://openrouter.ai/api/v1",
        ),
        openrouter_models=[
            OpenRouterModel(
                id="vision/model",
                name="Vision Model",
                context_length=8192,
                pricing=OpenRouterPricing(prompt=0.000001, completion=0.000002),
                supports_reasoning=True,
            )
        ],
    )

    assert 'name="model"' in html
    assert '<option value="vision/model" selected>' in html
    assert 'id="model-field"' in html
    assert 'id="model-input"' not in html
    assert "Model ID" not in html
    assert "Vision Model" in html
    assert "supports reasoning" in html


def test_render_page_supports_collapsible_and_resizable_workspace():
    html = render_page(config=DemoConfig())

    assert "--rail-width: 310px" in html
    assert "--detail-width" not in html
    assert "--content-width: calc((100% - var(--rail-width) - 6px) / 2)" in html
    assert "--pdf-width: var(--content-width)" in html
    assert "--result-width: var(--content-width)" in html
    assert 'id="sidebar-toggle"' in html
    assert 'aria-label="Files 사이드바 접기"' in html
    assert 'class="workspace-resizer"' in html
    assert 'data-resizer="pdf-result"' in html
    assert "workspace.classList.toggle('rail-collapsed')" in html
    assert "fitWorkspaceContentWidths()" in html
    assert "setPointerCapture(event.pointerId)" in html
    assert "grid-template-columns: var(--rail-width) var(--pdf-width) 6px var(--result-width);" in html
    assert ".workspace.rail-collapsed" in html


def test_render_page_links_upload_button_to_file_input():
    html = render_page(config=DemoConfig())

    assert 'id="upload-trigger"' in html
    assert 'id="pdf-input"' in html
    assert 'name="pdf"' in html
    assert 'id="selected-file-name"' in html
    assert "fileInput.click()" in html
    assert "uploadSelectedFile()" in html
    assert "parseSelectedFile()" in html
    assert "fileInput.addEventListener('change', async" in html


def test_render_page_places_download_actions_in_tab_bar():
    html = render_page(config=DemoConfig())

    assert 'id="tab-download-links"' in html
    assert "tabDownloadLinks" in html
    assert 'title="Markdown 다운로드"' in html
    assert 'title="JSON 다운로드"' in html
    assert "job.links.markdown" in html
    assert "job.links.json" in html
    assert "file.links.jobs" in html
    assert "selectedFile.links.parse" in html


def test_render_page_shows_parse_progress():
    html = render_page(config=DemoConfig())

    assert "function progressBar(job) {" in html
    assert 'class="progress-track"' in html
    assert 'class="progress-fill"' in html
    assert "const progress = job.progress || {};" in html
    assert "progressBar(job)" in html


def test_render_page_includes_mobile_friendly_layout_rules():
    html = render_page(config=DemoConfig())

    assert "@media (max-width: 780px)" in html
    assert ".app-shell { min-width: 0; min-height: 100vh; height: auto; overflow: visible; display: block; }" in html
    assert ".topbar {" in html
    assert "grid-template-areas: \"title\" \"upload\";" in html
    assert ".brand { grid-area: title; min-width: 0; }" in html
    assert ".upload-bar { grid-area: upload; width: 100%; grid-template-columns: auto minmax(0, 1fr); }" in html
    assert ".workspace { display: flex; flex-direction: column; height: auto; }" in html
    assert ".left-rail { max-height: 220px; border-width: 0 0 1px 0; }" in html
    assert ".result-tabs { overflow-x: auto; flex-wrap: nowrap; padding: 0 12px; }" in html


def test_render_page_keeps_pdf_iframe_stable_during_polling():
    html = render_page(config=DemoConfig())

    assert "let renderedPdfFileId = null;" in html
    assert "function renderPdfPreview(file) {" in html
    assert "if (renderedPdfFileId === file.id) {" in html
    assert "renderPdfPreview(file);" in html


def test_render_page_uses_wide_pdf_preview_spacing():
    html = render_page(config=DemoConfig())

    assert "padding: 0;" in html
    assert "width: 100%;" in html


def test_render_page_opens_pdf_preview_in_full_width_mode():
    html = render_page(config=DemoConfig())

    assert "function pdfPreviewUrl(file) {" in html
    assert "view=FitH" in html
    assert "zoom=page-width" in html
    assert "navpanes=0" in html
    assert 'src="${pdfPreviewUrl(file)}"' in html


def test_render_page_displays_results_by_page():
    html = render_page(config=DemoConfig())

    assert "function pageSeparatedMarkdown()" in html
    assert "selectedJson.pages" in html
    assert "page.page_number" in html
    assert "## Page ${pageNumber}" in html


def test_render_page_keeps_result_scroll_stable_during_polling():
    html = render_page(config=DemoConfig())

    assert "let renderedResultKey = null;" in html
    assert "function resultRenderKey(job) {" in html
    assert "if (renderedResultKey === resultRenderKey(job)) {" in html
    assert "renderedResultKey = resultRenderKey(selectedJob);" in html


def test_render_page_keeps_download_buttons_stable_during_polling():
    html = render_page(config=DemoConfig())

    stable_check = html.index("if (renderedResultKey === resultRenderKey(job)) {")
    download_update = html.index("tabDownloadLinks.innerHTML = `")

    assert stable_check < download_update


def test_render_page_embeds_valid_javascript(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        return
    html = render_page(config=DemoConfig())
    script_start = html.index("<script>") + len("<script>")
    script_end = html.index("</script>", script_start)
    script_path = tmp_path / "demo-page.js"
    script_path.write_text(html[script_start:script_end], encoding="utf-8")

    subprocess.run([node, "--check", str(script_path)], check=True)


def test_source_pdf_link_uses_original_filename_for_pdf_viewer_title():
    link = source_pdf_link("job123", "2026년_그룹_정보보호_수준진단.pdf")

    assert link == (
        "/api/jobs/job123/source/"
        "2026%EB%85%84_%EA%B7%B8%EB%A3%B9_"
        "%EC%A0%95%EB%B3%B4%EB%B3%B4%ED%98%B8_"
        "%EC%88%98%EC%A4%80%EC%A7%84%EB%8B%A8.pdf"
    )


def test_api_index_payload_lists_job_endpoints():
    payload = api_index_payload()

    assert payload["name"] == "vlm-parser demo api"
    assert payload["endpoints"]["files"] == "/api/files"
    assert payload["endpoints"]["file_detail"] == "/api/files/{file_id}"
    assert payload["endpoints"]["file_parse"] == "/api/files/{file_id}/parse"
    assert payload["endpoints"]["jobs"] == "/api/jobs"
    assert payload["endpoints"]["job_detail"] == "/api/jobs/{job_id}"


def test_content_disposition_header_supports_korean_filenames():
    header = content_disposition_header(
        "inline",
        "2026년_그룹_정보보호_수준진단.pdf",
    )

    header.encode("latin-1")
    assert header.startswith('inline; filename="download.pdf"')
    assert "filename*=UTF-8''" in header
