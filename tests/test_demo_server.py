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
    effective_openrouter_reasoning_effort,
    fetch_openrouter_models,
    is_openrouter_base_url,
    source_pdf_link,
    load_demo_config,
    normalize_model_base_url,
    process_job,
    render_page,
    render_pdf_preview_png,
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


def test_effective_openrouter_reasoning_effort_requires_supported_model(monkeypatch):
    config = DemoConfig(api_key="key", model="vision/model", base_url="https://openrouter.ai/api/v1")

    monkeypatch.setattr(
        "demo.server.safe_fetch_openrouter_models",
        lambda _config: [OpenRouterModel(id="vision/model", name="Vision Model", supports_reasoning=True)],
    )

    assert effective_openrouter_reasoning_effort(config, "high") == "high"


def test_effective_openrouter_reasoning_effort_ignores_unsupported_model(monkeypatch):
    config = DemoConfig(api_key="key", model="vision/model", base_url="https://openrouter.ai/api/v1")

    monkeypatch.setattr(
        "demo.server.safe_fetch_openrouter_models",
        lambda _config: [OpenRouterModel(id="vision/model", name="Vision Model", supports_reasoning=False)],
    )

    assert effective_openrouter_reasoning_effort(config, "high") == "auto"


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
    assert store.to_file_summary(file)["links"]["preview_png"].endswith("/preview.png")
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


def test_job_store_reports_real_storage_usage(tmp_path: Path):
    store = JobStore(tmp_path, storage_limit_bytes=10)
    store.create_file(UploadedFile(filename="first.pdf", content=b"1234"))
    store.create_file(UploadedFile(filename="second.pdf", content=b"123"))

    usage = store.storage_usage()

    assert usage == {
        "used_bytes": 7,
        "limit_bytes": 10,
        "percent": 70,
    }


def test_render_pdf_preview_png_renders_first_page(tmp_path: Path):
    import fitz

    source = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=160, height=120)
    page.insert_text((24, 48), "Preview")
    doc.save(source)
    doc.close()

    preview = render_pdf_preview_png(source)

    assert preview.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(preview) > 1000


def test_job_store_restores_files_and_completed_jobs_from_disk(tmp_path: Path):
    store = JobStore(tmp_path)
    file = store.create_file(UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"))
    job = store.create_job(
        file.id,
        JobOptions(
            use_vlm=True,
            render_dpi=220,
            trim=True,
            auto_slice=True,
            max_page_workers=6,
            reasoning_effort="low",
            model="vision/model",
        ),
    )
    assert job is not None
    store.update_progress(job.id, current=3, total=3, label="Complete")
    store.mark_done(job.id, {"document": {"markdown": "# Parsed"}}, "# Parsed")

    restored = JobStore(tmp_path)

    restored_files = restored.list_files()
    assert [restored_file.id for restored_file in restored_files] == [file.id]
    restored_file = restored_files[0]
    assert restored_file.filename == "sample.pdf"
    assert restored_file.source_path.read_bytes() == b"%PDF-1.7"

    restored_jobs = restored.list_jobs(file.id)
    assert [restored_job.id for restored_job in restored_jobs] == [job.id]
    restored_job = restored_jobs[0]
    assert restored_job.status == "done"
    assert restored_job.markdown == "# Parsed"
    assert restored_job.result_json == {"document": {"markdown": "# Parsed"}}
    assert restored_job.options.use_vlm is True
    assert restored_job.options.max_page_workers == 6
    assert restored_job.options.reasoning_effort == "low"
    assert restored.to_file_summary(restored_file)["latest_job"]["id"] == job.id


def test_process_job_stores_json_and_markdown_results(tmp_path: Path):
    store = JobStore(tmp_path)
    file = store.create_file(UploadedFile(filename="sample.pdf", content=b"%PDF-1.7"))
    job = store.create_job(
        file.id,
        JobOptions(
            use_vlm=False,
            render_dpi=180,
            trim=True,
            auto_slice=True,
            max_page_workers=3,
            reasoning_effort="high",
            model="",
        ),
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

    def parser_factory(
        *,
        use_vlm,
        render_dpi,
        trim,
        auto_slice,
        max_page_workers,
        reasoning_effort,
        config,
        progress_callback=None,
    ):
        assert use_vlm is False
        assert render_dpi == 180
        assert trim is True
        assert auto_slice is True
        assert max_page_workers == 3
        assert reasoning_effort == "high"
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
    assert "<h2>Files</h2>" in html
    assert "<h2>파일</h2>" not in html
    assert "<h2>선택 파일</h2>" in html
    assert "파일을 드래그하여 업로드" not in html
    assert "upload-dropzone" not in html
    assert 'id="rail-upload-trigger"' in html
    assert "account-pill" not in html
    assert "Use VLM</label>" in html
    assert "Use VLM rewrite" not in html
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
    assert "supports reasoning" not in html
    assert "Vision Model (vision/model) 🧠" in html


def test_render_page_supports_collapsible_and_resizable_workspace():
    html = render_page(config=DemoConfig())

    assert "--rail-width: 310px" in html
    assert "--detail-width" not in html
    assert "--content-width: calc((100% - var(--rail-width) - 6px) / 2)" in html
    assert "--pdf-width: var(--content-width)" in html
    assert "--result-width: var(--content-width)" in html
    assert 'id="sidebar-toggle"' in html
    assert 'aria-label="Files 사이드바 접기"' in html
    assert "border-bottom: 1px solid var(--line);" not in html.split(".jobs header {", 1)[1].split("}", 1)[0]
    assert 'class="workspace-resizer"' in html
    assert 'data-resizer="pdf-result"' in html
    assert "workspace.classList.toggle('rail-collapsed')" in html
    assert "fitWorkspaceContentWidths()" in html
    assert "setPointerCapture(event.pointerId)" in html
    assert "grid-template-columns: var(--rail-width) var(--pdf-width) 6px var(--result-width);" in html
    assert ".workspace.rail-collapsed" in html
    assert ".workspace.rail-collapsed .rail-search" in html
    assert ".workspace.rail-collapsed .rail-detail" in html


def test_render_page_links_upload_button_to_file_input():
    html = render_page(config=DemoConfig())

    assert 'id="rail-upload-trigger"' in html
    assert 'aria-label="파일 업로드"' in html
    assert 'id="upload-trigger"' not in html
    assert 'id="job-count"' not in html
    assert 'aria-hidden="true">...</span>' in html
    assert '>+</button>' not in html
    assert 'id="pdf-input"' in html
    assert 'name="pdf"' in html
    assert 'name="render_dpi" type="hidden" value="180"' in html
    assert 'name="max_page_workers" type="number" min="1" max="16" value="4"' in html
    assert 'name="reasoning_effort"' in html
    assert 'form="upload-form" name="max_page_workers"' in html
    assert 'form="upload-form" name="reasoning_effort"' in html
    assert '<option value="auto" selected>Think auto</option>' in html
    assert '<option value="off">Think off</option>' in html
    assert '<option value="low">Think low</option>' in html
    assert '<option value="medium">Think medium</option>' in html
    assert '<option value="high">Think high</option>' in html
    assert 'class="reasoning-control"' in html
    assert 'class="page-workers-control"' in html
    assert html.index('class="vlm-settings-dialog"') < html.index('class="reasoning-control"')
    assert html.index('class="vlm-settings-dialog"') < html.index('class="page-workers-control"')
    assert 'id="vlm-settings-modal"' in html
    assert 'data-upload-action="open-vlm-settings"' in html
    assert 'VLM 설정' in html
    assert "toggleVlmSettingsModal(true)" in html
    assert "toggleVlmSettingsModal(false)" in html
    assert "Render DPI" not in html
    assert 'id="selected-file-name"' not in html
    assert "selectedFileName" not in html
    assert 'id="upload-menu"' in html
    assert 'data-upload-action="select-file"' in html
    assert "toggleUploadMenu()" in html
    assert "uploadSelectedFile()" in html
    assert "parseSelectedFile()" in html
    assert "event.stopPropagation();" in html
    assert "fileInput.addEventListener('change', async" in html
    upload_form = html.split('<form id="upload-form"', 1)[1].split('</form>', 1)[0]
    assert 'class="reasoning-control"' not in upload_form
    assert 'class="page-workers-control"' not in upload_form
    assert 'id="model-select"' not in upload_form
    assert 'id="model-input"' not in upload_form
    assert 'Use VLM</label>' in upload_form
    assert '<button type="submit">실행</button>' in upload_form


def test_render_page_uses_mobile_safe_pdf_preview_and_menus():
    html = render_page(config=DemoConfig())

    assert api_index_payload()["endpoints"]["file_preview_png"] == "/api/files/{file_id}/preview.png"
    assert "file.links.preview_png" in html
    assert "isMobileViewport()" in html
    assert '<img class="pdf-preview-image"' in html
    assert ".job-row.menu-open { overflow: visible; }" in html
    assert "@media (max-width: 780px)" in html
    assert ".file-menu { position: fixed;" in html


def test_render_page_centers_empty_states_and_upload_ellipsis():
    html = render_page(config=DemoConfig())

    assert ".empty-state {" in html
    assert "text-align: center;" in html
    assert "place-items: center;" in html
    assert "max-width: 100%;" in html
    assert "overflow-wrap: anywhere;" in html
    assert "letter-spacing: 0;" in html
    assert 'aria-hidden="true">...</span>' in html


def test_render_page_moves_file_status_to_pdf_preview_footer():
    html = render_page(config=DemoConfig())

    assert 'id="pdf-meta-footer"' in html
    assert "renderPdfPreviewMeta(file)" in html
    assert "selectedFileStatusText()" in html
    assert "상태 / 페이지" not in html
    assert "업로드 / 모델" not in html


def test_render_page_fetches_real_storage_usage():
    html = render_page(config=DemoConfig())

    assert 'id="storage-meter"' in html
    assert 'class="storage-summary"' in html
    assert "fetch('/api/storage')" in html
    assert "formatBytes(storage.used_bytes)" in html
    assert "1.24 GB / 10 GB" not in html


def test_render_page_places_download_actions_in_tab_bar():
    html = render_page(config=DemoConfig())

    assert 'id="tab-download-links"' in html
    assert 'id="tab-active-download"' in html
    assert 'id="tab-md-download"' not in html
    assert 'id="tab-json-download"' not in html
    assert 'class="tab-download-icon"' in html
    assert "updateTabDownloadLinks(job);" in html
    assert "activeTab === 'json' ? job.links.json : job.links.markdown" in html
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

    assert "min-width: 1120px;" in html
    assert "grid-template-columns: minmax(250px, 320px) minmax(0, 1fr);" in html
    assert "padding: 10px 18px;" in html
    assert "width: 100%;" in html
    assert "flex: 1 1 0;" in html
    assert "max-width: 640px;" in html
    assert "text-overflow: ellipsis;" in html
    assert "@media (max-width: 780px)" in html
    assert ".app-shell { min-width: 0; min-height: 100vh; height: auto; overflow: visible; display: block; }" in html
    assert ".topbar {" in html
    assert "grid-template-areas: \"title\" \"upload\";" in html
    assert ".brand { grid-area: title; min-width: 0; }" in html
    assert ".upload-bar { grid-area: upload; width: 100%; flex-wrap: wrap; }" in html
    assert "grid-template-columns: auto auto;" in html
    assert "flex: 1 1 100%;" in html
    assert ".upload-bar .options label.check {" in html
    assert ".pdf-stage { min-height: 380px; border-right: 0; border-bottom: 1px solid var(--line); }" in html
    assert ".pdf-empty { width: min(260px, calc(100% - 32px)); padding: 20px; overflow-wrap: anywhere; }" in html
    assert ".pdf-shell { width: 100%; height: min(50vh, 420px); min-height: 340px; }" in html
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
    assert "padding: 10px 8px;" in html
    assert "padding: 22px 18px;" not in html
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

    download_update = html.index("updateTabDownloadLinks(job);")
    stable_check = html.index("if (renderedResultKey === resultRenderKey(job)) {")

    assert download_update < stable_check
    render_file_start = html.index("async function renderFile(file) {")
    render_job_start = html.index("async function renderJob(job) {")
    render_file_body = html[render_file_start:render_job_start]

    assert "tabDownloadLinks.innerHTML = '';" not in render_file_body


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
    assert payload["endpoints"]["storage"] == "/api/storage"
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
