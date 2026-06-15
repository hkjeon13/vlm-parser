from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from uuid import uuid4

import httpx


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_parser import ParseOptions, PdfParser, VlmOptions  # noqa: E402
from vlm_parser.vlm.client import OpenAICompatibleVlmClient  # noqa: E402


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True, slots=True)
class DemoConfig:
    api_key: str = ""
    model: str = ""
    base_url: str = ""


@dataclass(frozen=True, slots=True)
class OpenRouterPricing:
    prompt: float = 0.0
    completion: float = 0.0
    request: float = 0.0
    image: float = 0.0
    internal_reasoning: float = 0.0


@dataclass(frozen=True, slots=True)
class OpenRouterModel:
    id: str
    name: str
    context_length: int = 0
    pricing: OpenRouterPricing = field(default_factory=OpenRouterPricing)
    supports_reasoning: bool = False
    supports_include_reasoning: bool = False


@dataclass(frozen=True, slots=True)
class UploadedFile:
    filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class JobOptions:
    use_vlm: bool
    render_dpi: int
    trim: bool
    auto_slice: bool
    model: str = ""


@dataclass(slots=True)
class WorkspaceFile:
    id: str
    filename: str
    source_path: Path
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(slots=True)
class ParseJob:
    id: str
    file_id: str
    filename: str
    source_path: Path
    options: JobOptions
    status: str = "uploaded"
    created_at: float = 0.0
    updated_at: float = 0.0
    error: str = ""
    result_json: dict | None = None
    markdown: str = ""
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: int = 0
    progress_label: str = ""


class JobStore:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, WorkspaceFile] = {}
        self._jobs: dict[str, ParseJob] = {}
        self._file_jobs: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def create_file(self, uploaded: UploadedFile) -> WorkspaceFile:
        file_id = uuid4().hex
        file_dir = self.root_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)
        source_path = file_dir / "source.pdf"
        source_path.write_bytes(uploaded.content)
        now = time.time()
        file = WorkspaceFile(
            id=file_id,
            filename=uploaded.filename or "uploaded.pdf",
            source_path=source_path,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._files[file_id] = file
            self._file_jobs[file_id] = []
        return file

    def create_job(self, file_id: str, options: JobOptions) -> ParseJob | None:
        with self._lock:
            file = self._files.get(file_id)
            if file is None:
                return None
            job_id = uuid4().hex
            now = time.time()
            job = ParseJob(
                id=job_id,
                file_id=file.id,
                filename=file.filename,
                source_path=file.source_path,
                options=options,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            self._file_jobs.setdefault(file_id, []).append(job_id)
            file.updated_at = now
            return job

    def create(self, uploaded: UploadedFile, options: JobOptions) -> ParseJob:
        file = self.create_file(uploaded)
        job = self.create_job(file.id, options)
        if job is None:
            raise RuntimeError("Failed to create parse job")
        return job

    def get_file(self, file_id: str) -> WorkspaceFile | None:
        with self._lock:
            return self._files.get(file_id)

    def get(self, job_id: str) -> ParseJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_files(self) -> list[WorkspaceFile]:
        with self._lock:
            return sorted(
                self._files.values(),
                key=lambda file: file.updated_at,
                reverse=True,
            )

    def list_jobs(self, file_id: str | None = None) -> list[ParseJob]:
        with self._lock:
            if file_id is None:
                jobs = list(self._jobs.values())
            else:
                jobs = [
                    self._jobs[job_id]
                    for job_id in self._file_jobs.get(file_id, [])
                    if job_id in self._jobs
                ]
            return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def list(self) -> list[ParseJob]:
        return self.list_jobs()

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            file = self._files.pop(file_id, None)
            if file is None:
                return False
            for job_id in self._file_jobs.pop(file_id, []):
                self._jobs.pop(job_id, None)
        shutil.rmtree(self.root_dir / file_id, ignore_errors=True)
        return True

    def mark_running(self, job_id: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "running"
            if not job.progress_label:
                job.progress_label = "Starting parse"
            self._touch(job)
            return job

    def mark_queued(self, job_id: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "queued"
            job.error = ""
            job.progress_current = 0
            job.progress_total = 0
            job.progress_percent = 0
            job.progress_label = "Queued"
            self._touch(job)
            return job

    def update_progress(self, job_id: str, *, current: int, total: int, label: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            safe_total = max(total, 0)
            safe_current = max(0, min(current, safe_total)) if safe_total else max(0, current)
            job.progress_current = safe_current
            job.progress_total = safe_total
            job.progress_percent = int(round((safe_current / safe_total) * 100)) if safe_total else 0
            job.progress_label = label
            self._touch(job)
            return job

    def mark_done(self, job_id: str, result_json: dict, markdown: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "done"
            job.result_json = result_json
            job.markdown = markdown
            job.error = ""
            job.progress_current = job.progress_total or job.progress_current
            job.progress_percent = 100
            job.progress_label = "Complete"
            self._touch(job)
            return job

    def mark_failed(self, job_id: str, error: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "failed"
            job.error = error
            job.progress_label = "Failed"
            self._touch(job)
            return job

    def _set_status(self, job_id: str, status: str, *, clear_error: bool = False) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = status
            if clear_error:
                job.error = ""
            self._touch(job)
            return job

    def _touch(self, job: ParseJob) -> None:
        now = time.time()
        job.updated_at = now
        file = self._files.get(job.file_id)
        if file is not None:
            file.updated_at = now

    def _latest_job(self, file_id: str) -> ParseJob | None:
        jobs = self.list_jobs(file_id)
        return jobs[0] if jobs else None

    def to_file_summary(self, file: WorkspaceFile) -> dict:
        jobs = self.list_jobs(file.id)
        latest_job = jobs[0] if jobs else None
        return {
            "id": file.id,
            "filename": file.filename,
            "created_at": file.created_at,
            "updated_at": file.updated_at,
            "job_count": len(jobs),
            "latest_job": self.to_summary(latest_job) if latest_job else None,
            "links": {
                "self": f"/api/files/{file.id}",
                "jobs": f"/api/files/{file.id}/jobs",
                "parse": f"/api/files/{file.id}/parse",
                "source_pdf": file_source_pdf_link(file.id, file.filename),
            },
        }

    def to_summary(self, job: ParseJob) -> dict:
        metrics = job.result_json.get("metrics", {}) if job.result_json else {}
        return {
            "id": job.id,
            "file_id": job.file_id,
            "filename": job.filename,
            "status": job.status,
            "error": job.error,
            "use_vlm": job.options.use_vlm,
            "model": job.options.model,
            "render_dpi": job.options.render_dpi,
            "trim": job.options.trim,
            "auto_slice": job.options.auto_slice,
            "metrics": {
                "total_seconds": metrics.get("total_seconds"),
                "average_seconds_per_page": metrics.get("average_seconds_per_page"),
                "cost_usd": metrics.get("cost_usd"),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "reasoning_tokens": metrics.get("reasoning_tokens"),
            },
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "has_result": job.result_json is not None,
            "progress": {
                "current": job.progress_current,
                "total": job.progress_total,
                "percent": job.progress_percent,
                "label": job.progress_label,
            },
            "links": {
                "self": f"/api/jobs/{job.id}",
                "source_pdf": source_pdf_link(job.id, job.filename),
                "json": f"/api/jobs/{job.id}/result.json",
                "markdown": f"/api/jobs/{job.id}/result.md",
            },
        }


JOB_STORE = JobStore(Path(tempfile.gettempdir()) / "vlm-parser-demo-jobs")


def load_demo_config(env_path: str | Path = ".env") -> DemoConfig:
    values = dict(os.environ)
    path = Path(env_path)
    if path.exists():
        values.update(_read_env_file(path))
    return DemoConfig(
        api_key=values.get("MODEL_API_KEY", ""),
        model=values.get("MODEL_NAME", ""),
        base_url=values.get("MODEL_BASE_URL", ""),
    )


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def normalize_model_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    suffix = "/chat/completions"
    if cleaned.endswith(suffix):
        return cleaned[: -len(suffix)]
    return cleaned


def is_openrouter_base_url(base_url: str) -> bool:
    return normalize_model_base_url(base_url) == OPENROUTER_BASE_URL


def _float_price(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _openrouter_pricing(data: dict) -> OpenRouterPricing:
    pricing = data.get("pricing") or {}
    return OpenRouterPricing(
        prompt=_float_price(pricing.get("prompt")),
        completion=_float_price(pricing.get("completion")),
        request=_float_price(pricing.get("request")),
        image=_float_price(pricing.get("image")),
        internal_reasoning=_float_price(pricing.get("internal_reasoning")),
    )


def _openrouter_model_from_payload(data: dict) -> OpenRouterModel:
    supported = set(data.get("supported_parameters") or [])
    return OpenRouterModel(
        id=str(data.get("id") or ""),
        name=str(data.get("name") or data.get("id") or ""),
        context_length=int(data.get("context_length") or 0),
        pricing=_openrouter_pricing(data),
        supports_reasoning="reasoning" in supported,
        supports_include_reasoning="include_reasoning" in supported,
    )


def fetch_openrouter_models(http_client: object | None = None) -> list[OpenRouterModel]:
    client = http_client or httpx.Client()
    response = client.get(
        f"{OPENROUTER_BASE_URL}/models?output_modalities=text",
        timeout=10,
    )
    response.raise_for_status()
    models: list[OpenRouterModel] = []
    for item in response.json().get("data", []):
        architecture = item.get("architecture") or {}
        input_modalities = set(architecture.get("input_modalities") or [])
        output_modalities = set(architecture.get("output_modalities") or [])
        if "image" not in input_modalities or "text" not in output_modalities:
            continue
        model = _openrouter_model_from_payload(item)
        if model.id:
            models.append(model)
    return sorted(models, key=lambda model: model.name.lower())


def safe_fetch_openrouter_models(config: DemoConfig) -> list[OpenRouterModel]:
    if not is_openrouter_base_url(config.base_url):
        return []
    try:
        return fetch_openrouter_models()
    except Exception:
        return []


def find_openrouter_model(models: list[OpenRouterModel], model_id: str) -> OpenRouterModel | None:
    for model in models:
        if model.id == model_id:
            return model
    return None


def calculate_openrouter_cost(result_json: dict, model: OpenRouterModel) -> dict:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    reasoning_tokens = 0
    request_count = 0
    for page in result_json.get("pages") or []:
        vlm = page.get("vlm") or {}
        for chunk in vlm.get("chunks") or []:
            usage = chunk.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
            request_count += 1

    pricing = model.pricing
    total_cost = (
        prompt_tokens * pricing.prompt
        + completion_tokens * pricing.completion
        + reasoning_tokens * pricing.internal_reasoning
        + request_count * pricing.request
        + request_count * pricing.image
    )
    return {
        "provider": "openrouter",
        "model": model.id,
        "request_count": request_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_cost_usd": round(total_cost, 10),
    }


def add_openrouter_cost_metrics(result_json: dict, config: DemoConfig) -> None:
    if not is_openrouter_base_url(config.base_url) or not config.model:
        return
    models = safe_fetch_openrouter_models(config)
    model = find_openrouter_model(models, config.model)
    if model is None:
        return
    cost = calculate_openrouter_cost(result_json, model)
    metrics = result_json.setdefault("metrics", {})
    metrics["cost_usd"] = cost["total_cost_usd"]
    metrics["openrouter"] = cost


def build_vlm_client(config: DemoConfig) -> OpenAICompatibleVlmClient | None:
    if not config.api_key or not config.model or not config.base_url:
        return None
    return OpenAICompatibleVlmClient(
        base_url=normalize_model_base_url(config.base_url),
        api_key=config.api_key,
        model=config.model,
        timeout_seconds=120,
    )


def make_parser(
    *,
    use_vlm: bool,
    render_dpi: int,
    trim: bool,
    auto_slice: bool,
    config: DemoConfig,
    progress_callback=None,
) -> PdfParser:
    vlm_client = build_vlm_client(config) if use_vlm else None
    return PdfParser(
        options=ParseOptions(
            render_dpi=render_dpi,
            trim=trim,
            auto_slice=auto_slice,
        ),
        vlm=VlmOptions(
            enabled=use_vlm and vlm_client is not None,
            model=config.model or None,
            base_url=normalize_model_base_url(config.base_url) if config.base_url else None,
            api_key=config.api_key or None,
            timeout_seconds=120,
        ),
        vlm_client=vlm_client,
        progress_callback=progress_callback,
    )


def process_job(
    job_id: str,
    *,
    store: JobStore,
    config: DemoConfig,
    parser_factory=make_parser,
) -> None:
    job = store.mark_running(job_id)
    if job is None:
        return
    try:
        def report_progress(current: int, total: int, label: str) -> None:
            store.update_progress(job.id, current=current, total=total, label=label)

        effective_config = DemoConfig(
            api_key=config.api_key,
            model=job.options.model or config.model,
            base_url=config.base_url,
        )
        parser = parser_factory(
            use_vlm=job.options.use_vlm,
            render_dpi=job.options.render_dpi,
            trim=job.options.trim,
            auto_slice=job.options.auto_slice,
            config=effective_config,
            progress_callback=report_progress,
        )
        result = parser.parse(job.source_path)
        result_json = result.to_json()
        add_openrouter_cost_metrics(result_json, effective_config)
        store.mark_done(job.id, result_json, result.to_markdown())
    except Exception as exc:
        traceback.print_exc()
        store.mark_failed(job.id, f"{type(exc).__name__}: {exc}")


def start_job(job_id: str, *, store: JobStore = JOB_STORE, config: DemoConfig) -> None:
    thread = threading.Thread(
        target=process_job,
        kwargs={"job_id": job_id, "store": store, "config": config},
        daemon=True,
    )
    thread.start()


def api_index_payload() -> dict:
    return {
        "name": "vlm-parser demo api",
        "status": "ok",
        "endpoints": {
            "files": "/api/files",
            "file_detail": "/api/files/{file_id}",
            "file_jobs": "/api/files/{file_id}/jobs",
            "file_parse": "/api/files/{file_id}/parse",
            "jobs": "/api/jobs",
            "job_detail": "/api/jobs/{job_id}",
            "source_pdf": "/api/jobs/{job_id}/source.pdf",
            "result_json": "/api/jobs/{job_id}/result.json",
            "result_markdown": "/api/jobs/{job_id}/result.md",
        },
    }


def content_disposition_header(disposition: str, filename: str) -> str:
    suffix = Path(filename).suffix or ".bin"
    ascii_name = f"download{suffix}"
    encoded_name = quote(Path(filename).name, safe="")
    return f'{disposition}; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded_name}'


def source_pdf_link(job_id: str, filename: str) -> str:
    display_name = Path(filename).name or "uploaded.pdf"
    if not Path(display_name).suffix:
        display_name = f"{display_name}.pdf"
    return f"/api/jobs/{job_id}/source/{quote(display_name, safe='')}"


def file_source_pdf_link(file_id: str, filename: str) -> str:
    display_name = Path(filename).name or "uploaded.pdf"
    if not Path(display_name).suffix:
        display_name = f"{display_name}.pdf"
    return f"/api/files/{file_id}/source/{quote(display_name, safe='')}"


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "vlm-parser-demo/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._handle_get_api(path)
            return
        if path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        config = load_demo_config(ROOT_DIR / ".env")
        self._send_html(
            render_page(
                config=config,
                openrouter_models=safe_fetch_openrouter_models(config),
            )
        )

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/jobs", "/api/files"}:
            self._handle_create_file()
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "files"] and parts[3] == "parse":
            self._handle_parse_file(parts[2])
            return
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "parse":
            self._handle_parse_job(parts[2])
            return
        if path != "/parse":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            file, error = self._enqueue_file()
            config = load_demo_config(ROOT_DIR / ".env")
            if error:
                self._send_html(render_page(config=config, error=error))
            else:
                self._send_html(render_page(config=config, notice=f"{file.filename} 파일을 등록했습니다."))
        except Exception as exc:
            traceback.print_exc()
            self._send_html(render_page(error=f"{type(exc).__name__}: {exc}"))

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "files"]:
            if JOB_STORE.delete_file(parts[2]):
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_create_file(self) -> None:
        try:
            file, error = self._enqueue_file()
            if error:
                self._send_json({"error": error}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                {"file": JOB_STORE.to_file_summary(file)},
                status=HTTPStatus.ACCEPTED,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_parse_file(self, file_id: str) -> None:
        try:
            file = JOB_STORE.get_file(file_id)
            if file is None:
                self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
                return

            fields, _uploaded = self._parse_multipart()
            options = _job_options_from_fields(fields)
            config = load_demo_config(ROOT_DIR / ".env")
            effective_config = DemoConfig(
                api_key=config.api_key,
                model=options.model or config.model,
                base_url=config.base_url,
            )
            if options.use_vlm and build_vlm_client(effective_config) is None:
                self._send_json(
                    {"error": ".env에 MODEL_API_KEY, MODEL_NAME, MODEL_BASE_URL을 모두 설정해야 VLM을 사용할 수 있습니다."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            job = JOB_STORE.create_job(file.id, options)
            if job is None:
                self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
                return
            queued = JOB_STORE.mark_queued(job.id)
            if queued is None:
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            start_job(queued.id, config=config)
            self._send_json(
                {
                    "file": JOB_STORE.to_file_summary(file),
                    "job": JOB_STORE.to_summary(queued),
                },
                status=HTTPStatus.ACCEPTED,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_parse_job(self, job_id: str) -> None:
        try:
            job = JOB_STORE.get(job_id)
            if job is None:
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if job.status in {"queued", "running"}:
                self._send_json(
                    {"job": JOB_STORE.to_summary(job)},
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if job.status == "done":
                self._send_json({"job": JOB_STORE.to_summary(job)})
                return

            config = load_demo_config(ROOT_DIR / ".env")
            effective_config = DemoConfig(
                api_key=config.api_key,
                model=job.options.model or config.model,
                base_url=config.base_url,
            )
            if job.options.use_vlm and build_vlm_client(effective_config) is None:
                self._send_json(
                    {"error": ".env에 MODEL_API_KEY, MODEL_NAME, MODEL_BASE_URL을 모두 설정해야 VLM을 사용할 수 있습니다."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            queued = JOB_STORE.mark_queued(job.id)
            if queued is None:
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            start_job(queued.id, config=config)
            self._send_json(
                {"job": JOB_STORE.to_summary(queued)},
                status=HTTPStatus.ACCEPTED,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _enqueue_file(self) -> tuple[WorkspaceFile | None, str]:
        fields, uploaded = self._parse_multipart()
        if uploaded is None or not uploaded.content:
            return None, "PDF 파일을 선택해 주세요."

        return JOB_STORE.create_file(uploaded), ""

    def _handle_get_api(self, path: str) -> None:
        if path in {"/api", "/api/"}:
            self._send_json(api_index_payload())
            return

        if path == "/api/jobs":
            self._send_json({"jobs": [JOB_STORE.to_summary(job) for job in JOB_STORE.list()]})
            return
        if path == "/api/files":
            self._send_json({"files": [JOB_STORE.to_file_summary(file) for file in JOB_STORE.list_files()]})
            return

        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[:2] == ["api", "files"]:
            self._handle_get_file_api(parts)
            return

        if len(parts) < 3 or parts[:2] != ["api", "jobs"]:
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        job = JOB_STORE.get(parts[2])
        if job is None:
            self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
            return

        if len(parts) == 3:
            self._send_json({"job": JOB_STORE.to_summary(job)})
            return

        if (len(parts) == 4 and parts[3] == "source.pdf") or (
            len(parts) == 5 and parts[3] == "source"
        ):
            self._send_file(
                job.source_path.read_bytes(),
                content_type="application/pdf",
                filename=job.filename,
                inline=True,
            )
            return

        if job.status != "done" or job.result_json is None:
            self._send_json({"error": "Job is not complete"}, status=HTTPStatus.CONFLICT)
            return

        if len(parts) == 4 and parts[3] == "result.json":
            self._send_file(
                json.dumps(job.result_json, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json; charset=utf-8",
                filename=f"{Path(job.filename).stem or job.id}.json",
            )
            return

        if len(parts) == 4 and parts[3] == "result.md":
            self._send_file(
                job.markdown.encode("utf-8"),
                content_type="text/markdown; charset=utf-8",
                filename=f"{Path(job.filename).stem or job.id}.md",
            )
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_get_file_api(self, parts: list[str]) -> None:
        file = JOB_STORE.get_file(parts[2])
        if file is None:
            self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
            return

        if len(parts) == 3:
            self._send_json({"file": JOB_STORE.to_file_summary(file)})
            return

        if len(parts) == 4 and parts[3] == "jobs":
            self._send_json({"jobs": [JOB_STORE.to_summary(job) for job in JOB_STORE.list_jobs(file.id)]})
            return

        if (len(parts) == 4 and parts[3] == "source.pdf") or (
            len(parts) == 5 and parts[3] == "source"
        ):
            self._send_file(
                file.source_path.read_bytes(),
                content_type="application/pdf",
                filename=file.filename,
                inline=True,
            )
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _parse_multipart(self) -> tuple[dict[str, str], UploadedFile | None]:
        length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        if "multipart/form-data" not in content_type:
            parsed = parse_qs(body.decode("utf-8"))
            return {key: values[-1] for key, values in parsed.items()}, None
        raw_message = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8") + body
        message = BytesParser(policy=policy.default).parsebytes(raw_message)

        fields: dict[str, str] = {}
        uploaded: UploadedFile | None = None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            content = part.get_payload(decode=True) or b""
            if filename:
                uploaded = UploadedFile(filename=filename, content=content)
            elif name:
                fields[name] = content.decode(part.get_content_charset() or "utf-8")
        return fields, uploaded

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(
        self,
        content: bytes,
        *,
        content_type: str,
        filename: str,
        inline: bool = False,
    ) -> None:
        disposition = "inline" if inline else "attachment"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Disposition",
            content_disposition_header(disposition, filename),
        )
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def _int_field(value: str, *, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _job_options_from_fields(fields: dict[str, str]) -> JobOptions:
    return JobOptions(
        use_vlm=fields.get("use_vlm", "off") == "on",
        render_dpi=_int_field(fields.get("render_dpi", "180"), default=180),
        trim=fields.get("trim", "off") == "on",
        auto_slice=fields.get("auto_slice", "off") == "on",
        model=fields.get("model", "").strip(),
    )


def render_page(
    *,
    config: DemoConfig | None = None,
    markdown: str = "",
    json_text: str = "",
    filename: str = "",
    error: str = "",
    notice: str = "",
    used_vlm: bool = False,
    openrouter_models: list[OpenRouterModel] | None = None,
) -> str:
    config = config or DemoConfig()
    openrouter_models = openrouter_models or []
    has_vlm_config = bool(config.api_key and config.model and config.base_url)
    escaped_markdown = html.escape(markdown)
    escaped_json = html.escape(json_text)
    escaped_error = html.escape(error)
    escaped_notice = html.escape(notice)
    escaped_filename = html.escape(filename)
    model_label = html.escape(config.model or "not configured")
    model_value = html.escape(config.model)
    vlm_status = "ready" if has_vlm_config else "missing .env values"
    is_openrouter = is_openrouter_base_url(config.base_url)
    model_options = "\n".join(
        (
            f'<option value="{html.escape(model.id)}"'
            f'{" selected" if model.id == config.model else ""}>'
            f'{html.escape(model.name)} ({html.escape(model.id)})'
            f'{" · supports reasoning" if model.supports_reasoning else ""}'
            f'</option>'
        )
        for model in openrouter_models
    )
    model_select = (
        f"""
        <label class="model-control">
          <span>Model</span>
          <select id="model-select">
            <option value="{model_value}">Configured default ({model_value or "none"})</option>
            {model_options}
          </select>
        </label>
        """
        if is_openrouter and openrouter_models
        else ""
    )
    model_field = f'<input id="model-field" name="model" type="hidden" value="{model_value}">'
    model_manual = (
        f"""
        <label class="model-control">
          <span>Model ID</span>
          <input id="model-input" type="text" value="{model_value}" placeholder="provider/model-id">
        </label>
    """
        if not (is_openrouter and openrouter_models)
        else ""
    )
    result_section = ""
    if markdown or json_text:
        result_section = f"""
        <section class="results">
          <div class="result-meta">
            <strong>{escaped_filename}</strong>
            <span>VLM: {"on" if used_vlm else "off"}</span>
          </div>
          <div class="panes">
            <article>
              <h2>Markdown</h2>
              <pre>{escaped_markdown}</pre>
            </article>
            <article>
              <h2>JSON</h2>
              <pre>{escaped_json}</pre>
            </article>
          </div>
        </section>
        """

    error_section = (
        f'<div class="error" role="alert">{escaped_error}</div>' if error else ""
    )
    notice_section = (
        f'<div class="notice" role="status">{escaped_notice}</div>' if notice else ""
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vlm-parser demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #22242a;
      --muted: #68707c;
      --line: #d9ddd5;
      --accent: #0f766e;
      --accent-strong: #0b4f4a;
      --error: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }}
    .app-shell {{
      position: relative;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      height: 100vh;
      min-width: 1280px;
      overflow: hidden;
    }}
    .topbar {{
      display: grid;
      grid-template-columns: minmax(260px, 330px) minmax(0, 1fr) 42px;
      gap: 20px;
      align-items: center;
      margin: 0;
      min-height: 78px;
      padding: 12px 26px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    .brand {{
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr);
      gap: 12px;
      align-items: center;
      min-width: 0;
    }}
    .brand-mark {{
      display: grid;
      place-items: center;
      width: 38px;
      height: 38px;
      border: 2px solid var(--accent);
      border-radius: 7px;
      color: var(--accent);
      font-size: 22px;
      font-weight: 900;
      line-height: 1;
    }}
    .brand-copy {{
      display: grid;
      gap: 4px;
      min-width: 0;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 20px;
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 17px; line-height: 1.05; }}
    h2 {{ font-size: 16px; }}
    .status {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      color: var(--accent);
      font-size: 12px;
      font-weight: 750;
      line-height: 1.2;
    }}
    .status strong {{
      display: none;
    }}
    .status span {{
      white-space: nowrap;
    }}
    .status::after {{
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: var(--accent);
    }}
    form {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .upload-bar {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      margin: 0;
      border: 0;
      border-radius: 0;
      padding: 0;
      background: transparent;
      gap: 8px 12px;
      min-width: 0;
      justify-self: stretch;
      height: 100%;
    }}
    #pdf-input {{
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }}
    .upload-bar .options {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: nowrap;
      gap: 12px;
      min-width: 0;
      flex: 1 1 auto;
    }}
    .upload-bar .options > * {{
      min-width: 0;
    }}
    .upload-bar .field-options {{
      display: none;
    }}
    .upload-bar .check {{
      display: none;
    }}
    .upload-bar .options label.check {{
      display: flex;
      min-height: 36px;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }}
    .upload-bar .options label:not(.check) {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      line-height: 1.1;
      white-space: nowrap;
    }}
    .upload-bar .options label:not(.check):has(select) {{
      flex: 1 1 420px;
      max-width: 720px;
    }}
    .upload-bar .options label:not(.check):has(input[type="text"]) {{
      flex: 0 1 280px;
    }}
    .upload-bar select,
    .upload-bar input[type="text"] {{
      flex: 1 1 auto;
      width: 100%;
      min-width: 0;
      height: 36px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      padding: 4px 10px;
    }}
    .upload-bar button[type="submit"] {{
      min-height: 36px;
      min-width: 68px;
      flex: 0 0 auto;
      justify-self: end;
      border-radius: 7px;
      background: var(--accent);
      color: #fff;
      padding: 0 14px;
    }}
    .account-pill {{
      display: grid;
      place-items: center;
      width: 38px;
      height: 38px;
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      font-weight: 850;
    }}
    label {{ display: grid; gap: 7px; font-weight: 650; }}
    input[type="file"], input[type="number"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .options {{
      display: grid;
      gap: 10px;
      align-content: start;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 9px;
      font-weight: 600;
      min-height: 34px;
    }}
    .check input {{ width: 18px; height: 18px; }}
    button {{
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
      font-weight: 750;
      min-height: 44px;
      padding: 10px 14px;
    }}
    button:hover {{ background: var(--accent-strong); }}
    .error {{
      border: 1px solid #f1a29b;
      border-radius: 8px;
      background: #fff5f3;
      color: var(--error);
      padding: 12px 14px;
      margin-bottom: 16px;
      overflow-wrap: anywhere;
    }}
    .notice {{
      border: 1px solid #a6d8ce;
      border-radius: 8px;
      background: #effaf7;
      color: var(--accent-strong);
      padding: 12px 14px;
      margin-bottom: 16px;
      overflow-wrap: anywhere;
    }}
    .workspace {{
      --rail-width: 310px;
      --content-width: calc((100% - var(--rail-width) - 6px) / 2);
      --pdf-width: var(--content-width);
      --result-width: var(--content-width);
      display: grid;
      grid-template-columns: var(--rail-width) var(--pdf-width) 6px var(--result-width);
      gap: 0;
      align-items: stretch;
      min-height: 0;
      height: 100%;
    }}
    .workspace.rail-collapsed {{
      --rail-width: 44px;
    }}
    .workspace.rail-collapsed .left-rail h2,
    .workspace.rail-collapsed .left-rail .badge,
    .workspace.rail-collapsed .job-list {{
      display: none;
    }}
    .workspace.rail-collapsed .jobs header {{
      justify-content: center;
      padding: 12px 6px;
    }}
    .jobs {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .left-rail {{
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto auto auto;
      border-width: 0 1px 0 0;
      border-radius: 0;
      background: #fbfcfd;
      min-height: 0;
    }}
    .jobs header {{
      position: relative;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .rail-title {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .sidebar-toggle {{
      display: inline-grid;
      place-items: center;
      width: 28px;
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--muted);
      padding: 0;
      font-size: 16px;
      line-height: 1;
    }}
    .sidebar-toggle:hover {{ background: #eef2f7; color: var(--ink); }}
    .workspace-resizer {{
      min-width: 6px;
      cursor: col-resize;
      background: #e2e8f0;
      border-right: 1px solid #cbd5e1;
      border-left: 1px solid #cbd5e1;
    }}
    .workspace-resizer:hover,
    .workspace-resizer.active {{
      background: var(--accent);
      border-color: var(--accent);
    }}
    .job-list {{
      display: grid;
      align-content: start;
      max-height: none;
      height: auto;
      overflow: auto;
      padding: 12px;
      gap: 8px;
    }}
    .rail-search {{
      margin: 12px;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 8px 12px;
      font: inherit;
      font-size: 13px;
    }}
    .storage-meter {{
      display: grid;
      gap: 8px;
      padding: 0 16px 18px;
      color: var(--muted);
      font-size: 12px;
    }}
    .storage-meter span {{
      color: var(--ink);
      font-weight: 750;
    }}
    .storage-bar {{
      height: 7px;
      border-radius: 999px;
      background: #e5e7eb;
      overflow: hidden;
    }}
    .storage-bar div {{
      width: 16%;
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
    }}
    .job-row {{
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 32px;
      gap: 8px;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      color: var(--ink);
      text-align: left;
      min-height: 0;
      padding: 11px 10px 11px 14px;
    }}
    .job-row:hover {{ background: #fff; }}
    .job-row.active {{ background: #eefaf7; border-color: #95d5c8; box-shadow: inset 3px 0 0 var(--accent); }}
    .file-row-main {{
      display: grid;
      min-width: 0;
      gap: 4px;
    }}
    .job-row strong, .file-name {{
      display: block;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 750;
    }}
    .file-menu-button {{
      display: inline-grid;
      place-items: center;
      width: 32px;
      min-height: 32px;
      align-self: start;
      border-radius: 6px;
      background: transparent;
      color: var(--muted);
      padding: 0;
      font-size: 20px;
      line-height: 1;
    }}
    .file-menu-button:hover {{ background: #e2e8f0; color: var(--ink); }}
    .file-menu {{
      position: absolute;
      top: 42px;
      right: 8px;
      z-index: 10;
      display: grid;
      min-width: 120px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 10px 26px rgba(15,23,42,0.16);
      padding: 6px;
    }}
    .file-menu[hidden] {{ display: none; }}
    .file-menu button {{
      min-height: 32px;
      border-radius: 6px;
      background: transparent;
      color: var(--ink);
      padding: 6px 9px;
      text-align: left;
      font-size: 13px;
    }}
    .file-menu button:hover {{ background: #f1f5f9; }}
    .file-menu button.danger {{ color: var(--error); }}
    .job-row span {{ color: var(--muted); font-size: 13px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 24px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--muted);
      background: #fbfbfa;
      font-size: 12px;
      font-weight: 750;
    }}
    .badge.done {{ color: #0f6b3d; background: #eef9f1; border-color: #b8dfc2; }}
    .badge.failed {{ color: var(--error); background: #fff5f3; border-color: #f1a29b; }}
    .badge.running, .badge.queued {{ color: #73510a; background: #fff8e8; border-color: #edd28a; }}
    .progress-block {{
      display: grid;
      gap: 6px;
      width: 100%;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }}
    .progress-track {{
      width: 100%;
      height: 8px;
      border-radius: 99px;
      overflow: hidden;
      background: #dbe3ef;
    }}
    .progress-fill {{
      height: 100%;
      width: var(--progress, 0%);
      min-width: 0;
      border-radius: inherit;
      background: var(--accent);
      transition: width 0.2s ease;
    }}
    .progress-text {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    .pdf-stage {{
      min-width: 0;
      min-height: 0;
      background: #f3f6fa;
      border-right: 1px solid var(--line);
      padding: 22px 18px;
    }}
    .pdf-canvas {{
      width: 100%;
      height: 100%;
      min-height: 0;
      overflow: hidden;
      padding: 0;
      display: grid;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .pdf-empty {{
      align-self: center;
      justify-self: center;
      width: min(520px, 80%);
      border: 1px dashed #9aa8ba;
      border-radius: 10px;
      background: rgba(255,255,255,0.75);
      color: var(--muted);
      padding: 28px;
      text-align: center;
      font-weight: 700;
    }}
    .pdf-shell {{
      width: 100%;
      height: 100%;
      min-height: 0;
      background: #fff;
    }}
    .preview {{
      min-width: 0;
      display: grid;
      gap: 12px;
    }}
    .result-panel {{
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      gap: 0;
      min-height: 0;
      background: #fff;
    }}
    .preview-toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      padding: 20px 18px 12px;
    }}
    .result-tabs {{
      display: flex;
      align-items: flex-end;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
    }}
    .result-tabs button {{
      min-width: 96px;
      min-height: 38px;
      border: 1px solid transparent;
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      background: #f8fafc;
      color: #475569;
      font-weight: 750;
    }}
    .result-tabs button.active {{
      background: #fff;
      border-color: var(--line);
      color: var(--accent);
      transform: translateY(1px);
    }}
    .rail-detail {{
      display: grid;
      gap: 8px;
      border-top: 1px solid var(--line);
      padding: 10px 12px;
      background: #fff;
    }}
    .rail-detail h2 {{
      margin: 0;
      font-size: 13px;
    }}
    .detail-card {{
      display: grid;
      gap: 7px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px;
    }}
    .detail-row {{
      display: grid;
      gap: 2px;
      color: var(--muted);
      font-size: 12px;
    }}
    .detail-row strong {{
      color: var(--ink);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .tab-download-links {{
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 8px;
      padding-bottom: 6px;
    }}
    .tab-download-links a {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 9px;
      color: var(--accent-strong);
      background: var(--panel);
      text-decoration: none;
      font-size: 12px;
      font-weight: 800;
    }}
    .tab-download-links a:hover {{ border-color: var(--accent); }}
    .result-body {{
      min-height: 0;
      overflow: auto;
      padding: 18px;
    }}
    .result-footer {{
      display: flex;
      justify-content: flex-end;
      padding: 8px 18px 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .download-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .download-links a {{
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      color: var(--accent-strong);
      background: var(--panel);
      text-decoration: none;
      font-weight: 750;
    }}
    .review-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }}
    .pdf-frame {{
      width: 100%;
      height: 100%;
      min-height: 0;
      border: 0;
      background: #fff;
    }}
    .empty-state {{
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfbfa;
      color: var(--muted);
      padding: 24px;
    }}
    .results {{
      display: grid;
      gap: 12px;
    }}
    .result-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
    }}
    .panes {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }}
    article {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    article h2 {{
      border-bottom: 1px solid var(--line);
      padding: 12px 14px;
      background: #fbfbfa;
    }}
    pre {{
      min-height: 360px;
      max-height: 70vh;
      margin: 0;
      padding: 14px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .page-result {{
      display: grid;
      gap: 10px;
      padding: 14px 0 18px;
      border-top: 1px solid var(--line);
    }}
    .page-result:first-of-type {{ border-top: 0; padding-top: 0; }}
    .page-result h3 {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    @media (max-width: 780px) {{
      header, form, .panes, .review-grid {{ grid-template-columns: 1fr; }}
      header {{ display: grid; }}
      .status {{ min-width: 0; }}
      body {{ overflow: auto; }}
      .app-shell {{ min-width: 0; min-height: 100vh; height: auto; overflow: visible; display: block; }}
      .topbar {{
        position: sticky;
        top: 0;
        z-index: 5;
        grid-template-columns: minmax(0, 1fr);
        grid-template-areas: "title" "upload";
        gap: 10px;
        min-height: 0;
        padding: 10px 12px;
      }}
      .brand {{ grid-area: title; min-width: 0; }}
      .account-pill {{ display: none; }}
      .upload-bar {{ grid-area: upload; width: 100%; flex-wrap: wrap; }}
      .upload-bar .options {{
        flex-wrap: wrap;
        justify-content: stretch;
        align-items: end;
      }}
      .upload-bar .options label:not(.check):has(select),
      .upload-bar .options label:not(.check):has(input[type="text"]) {{
        flex: 1 1 220px;
        max-width: none;
      }}
      .upload-bar .options label.check:nth-of-type(4) {{ min-width: 0; white-space: normal; }}
      .upload-bar button[type="submit"] {{ flex: 0 0 auto; }}
      .workspace {{ display: flex; flex-direction: column; height: auto; }}
      .workspace.rail-collapsed .job-list {{ display: grid; }}
      .workspace.rail-collapsed .left-rail h2,
      .workspace.rail-collapsed .left-rail .badge {{ display: inline-flex; }}
      .workspace-resizer {{ display: none; }}
      .left-rail {{ max-height: 220px; border-width: 0 0 1px 0; }}
      .storage-meter, .rail-detail {{ display: none; }}
      .jobs header {{ padding: 10px 12px; }}
      .job-list {{ height: auto; max-height: 164px; }}
      .pdf-stage {{ min-height: 520px; border-right: 0; border-bottom: 1px solid var(--line); }}
      .pdf-canvas {{ padding: 8px; }}
      .pdf-shell {{ width: 100%; height: min(62vh, 520px); min-height: 420px; }}
      .result-panel {{ min-height: 560px; }}
      .preview-toolbar {{ padding: 14px 12px 10px; }}
      .result-tabs {{ overflow-x: auto; flex-wrap: nowrap; padding: 0 12px; }}
      .result-tabs button {{ min-width: 84px; flex: 0 0 auto; }}
      .tab-download-links {{ margin-left: 8px; flex: 0 0 auto; }}
      .result-body {{ min-height: 420px; padding: 12px; }}
      pre {{ min-height: 260px; max-height: none; font-size: 12px; }}
      .result-footer {{ justify-content: flex-start; overflow-wrap: anywhere; }}
    }}
    @media (max-width: 430px) {{
      h1 {{ font-size: 15px; }}
      .topbar {{ grid-template-columns: 1fr; }}
      .upload-bar button[type="submit"] {{ padding: 0 10px; }}
      .tab-download-links a {{ padding: 4px 7px; }}
    }}
  </style>
</head>
<body>
  <main class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">▣</div>
        <div class="brand-copy">
          <h1>VLM Parser Demo</h1>
          <div class="status">
            <strong>{model_label}</strong>
            <span>VLM config: {vlm_status}</span>
          </div>
        </div>
      </div>
      <form id="upload-form" class="upload-bar" action="/api/files" method="post" enctype="multipart/form-data">
        <input id="pdf-input" name="pdf" type="file" accept="application/pdf,.pdf">
        <input name="render_dpi" type="hidden" value="180">
        <input name="trim" type="hidden" value="on">
        <input name="auto_slice" type="hidden" value="on">
        <div class="options">
          {model_select}
          {model_manual}
          {model_field}
          <label class="check"><input name="use_vlm" type="checkbox"> Use VLM rewrite</label>
          <button type="submit">실행</button>
        </div>
      </form>
      <div class="account-pill" aria-label="계정">A</div>
    </header>
    {error_section}
    {notice_section}
    <section class="workspace">
      <aside class="jobs left-rail">
        <header>
          <div class="rail-title">
            <button id="sidebar-toggle" class="sidebar-toggle" type="button" aria-label="Files 사이드바 접기" title="Files 사이드바 접기">‹</button>
            <h2>파일</h2>
          </div>
          <button id="rail-upload-trigger" class="file-menu-button" type="button" title="파일 업로드" aria-label="파일 업로드">⋯</button>
        </header>
        <input class="rail-search" type="search" placeholder="파일명 검색" aria-label="파일명 검색">
        <div id="job-list" class="job-list">
          <div class="empty-state">No files yet.</div>
        </div>
        <section class="rail-detail">
          <h2>선택 파일</h2>
          <div class="detail-card">
            <div class="detail-row">
              <span>파일명</span>
              <strong id="detail-filename">선택된 파일 없음</strong>
            </div>
            <div class="detail-row">
              <span>상태 / 페이지</span>
              <strong><span id="detail-status">-</span> · <span id="detail-pages">-</span></strong>
            </div>
            <div class="detail-row">
              <span>업로드 / 모델</span>
              <strong><span id="detail-created">-</span> · <span id="detail-model">-</span></strong>
            </div>
          </div>
        </section>
        <div class="storage-meter">
          <span>저장소 사용량</span>
          <div>1.24 GB / 10 GB</div>
          <div class="storage-bar" aria-hidden="true"><div></div></div>
        </div>
      </aside>
      <section class="pdf-stage">
        <div id="pdf-canvas" class="pdf-canvas">
          <div class="pdf-empty">PDF를 업로드하면 이 영역에서 원문을 확인할 수 있습니다.</div>
        </div>
      </section>
      <div class="workspace-resizer" data-resizer="pdf-result" role="separator" aria-label="PDF와 결과 패널 너비 조정" aria-orientation="vertical" tabindex="0"></div>
      <section class="preview result-panel">
        <div class="preview-toolbar">
          <strong id="selected-title">Select a completed job</strong>
          <div id="download-links" class="download-links"></div>
        </div>
        <div class="result-tabs" role="tablist">
          <button id="tab-html" class="active" type="button" data-tab="html">MD</button>
          <button id="tab-json" type="button" data-tab="json">JSON</button>
          <div id="tab-download-links" class="tab-download-links"></div>
        </div>
        <div id="preview-body" class="result-body">
          <div class="empty-state">Upload a PDF to start parsing asynchronously.</div>
        </div>
        <div class="result-footer">Job ID: <span id="job-id-label">-</span></div>
      </section>
    </section>
    {result_section}
  </main>
  <script>
    const form = document.getElementById('upload-form');
    const modelSelect = document.getElementById('model-select');
    const modelInput = document.getElementById('model-input');
    const modelField = document.getElementById('model-field');
    const fileInput = document.getElementById('pdf-input');
    const railUploadTrigger = document.getElementById('rail-upload-trigger');
    const workspace = document.querySelector('.workspace');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const workspaceResizers = Array.from(document.querySelectorAll('[data-resizer]'));
    const jobList = document.getElementById('job-list');
    const jobCount = document.getElementById('job-count');
    const selectedTitle = document.getElementById('selected-title');
    const downloadLinks = document.getElementById('download-links');
    const tabDownloadLinks = document.getElementById('tab-download-links');
    const previewBody = document.getElementById('preview-body');
    const pdfCanvas = document.getElementById('pdf-canvas');
    const jobIdLabel = document.getElementById('job-id-label');
    const detailFilename = document.getElementById('detail-filename');
    const detailStatus = document.getElementById('detail-status');
    const detailPages = document.getElementById('detail-pages');
    const detailCreated = document.getElementById('detail-created');
    const detailModel = document.getElementById('detail-model');
    const tabButtons = Array.from(document.querySelectorAll('[data-tab]'));
    let selectedFileId = null;
    let selectedJobId = null;
    let selectedFile = null;
    let selectedJob = null;
    let selectedMarkdown = '';
    let selectedJson = null;
    let renderedPdfFileId = null;
    let renderedResultKey = null;
    let openMenuFileId = null;
    let activeResize = null;
    let activeTab = 'html';

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }}[char]));
    }}

    function statusLabel(job) {{
      if (!job) {{
        return '<span class="badge">uploaded</span>';
      }}
      return `<span class="badge ${{escapeHtml(job.status)}}">${{escapeHtml(job.status)}}</span>`;
    }}

    function syncModelField() {{
      if (!modelField) {{
        return;
      }}
      const selected = modelSelect?.value || '';
      const manual = modelInput?.value || '';
      modelField.value = selected || manual;
    }}

    function formatSeconds(value) {{
      const number = Number(value);
      if (!Number.isFinite(number)) {{
        return '-';
      }}
      return `${{number.toFixed(2)}}s`;
    }}

    function formatCost(value) {{
      const number = Number(value);
      if (!Number.isFinite(number)) {{
        return '-';
      }}
      return `$${{number.toFixed(6)}}`;
    }}

    function formatDateTime(value) {{
      const number = Number(value);
      if (!Number.isFinite(number) || number <= 0) {{
        return '-';
      }}
      return new Date(number * 1000).toLocaleString('ko-KR', {{
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      }});
    }}

    function compactMetrics(job) {{
      const metrics = job?.metrics || {{}};
      const parts = [];
      if (job?.model) {{
        parts.push(job.model);
      }}
      if (metrics.cost_usd !== null && metrics.cost_usd !== undefined) {{
        parts.push(formatCost(metrics.cost_usd));
      }}
      if (metrics.total_seconds !== null && metrics.total_seconds !== undefined) {{
        parts.push(formatSeconds(metrics.total_seconds));
      }}
      if (metrics.average_seconds_per_page !== null && metrics.average_seconds_per_page !== undefined) {{
        parts.push(`${{formatSeconds(metrics.average_seconds_per_page)}}/page`);
      }}
      return parts.join(' · ');
    }}

    function progressBar(job) {{
      const progress = job.progress || {{}};
      const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
      const total = Number(progress.total || 0);
      const current = Number(progress.current || 0);
      const label = progress.label || (job.status === 'done' ? 'Complete' : 'Waiting');
      const countText = total ? `${{current}} / ${{total}}` : '';
      return `
        <div class="progress-block" aria-label="Parse progress">
          <div class="progress-track" aria-hidden="true">
            <div class="progress-fill" style="--progress: ${{percent}}%"></div>
          </div>
          <div class="progress-text">
            <span>${{escapeHtml(label)}}</span>
            <span>${{escapeHtml(countText || `${{percent}}%`)}}</span>
          </div>
        </div>
      `;
    }}

    function fileMeta(file) {{
      const job = file.latest_job;
      if (!job) {{
        return `${{statusLabel(null)}} ${{escapeHtml(file.job_count)}} runs`;
      }}
      const metrics = compactMetrics(job);
      return `${{statusLabel(job)}} ${{escapeHtml(file.job_count)}} runs · VLM: ${{job.use_vlm ? 'on' : 'off'}} · DPI ${{job.render_dpi}}${{metrics ? ` · ${{escapeHtml(metrics)}}` : ''}}`;
    }}

    function updateDetailPanel(file, job) {{
      if (!detailFilename) {{
        return;
      }}
      const pages = selectedJson?.pages?.length || job?.progress?.total || '-';
      detailFilename.textContent = file?.filename || '선택된 파일 없음';
      detailStatus.textContent = job?.status || (file ? 'uploaded' : '-');
      detailPages.textContent = pages;
      detailCreated.textContent = formatDateTime(file?.created_at);
      detailModel.textContent = job?.model || '-';
    }}

    function pdfPreviewUrl(file) {{
      return `${{file.links.source_pdf}}#view=FitH&zoom=page-width&navpanes=0`;
    }}

    function renderPdfPreview(file) {{
      if (renderedPdfFileId === file.id) {{
        return;
      }}
      renderedPdfFileId = file.id;
      pdfCanvas.innerHTML = `
        <div class="pdf-shell">
          <iframe class="pdf-frame" src="${{pdfPreviewUrl(file)}}" title="PDF preview"></iframe>
        </div>
      `;
    }}

    function pageSeparatedMarkdown() {{
      const pages = Array.isArray(selectedJson?.pages) ? selectedJson.pages : [];
      if (!pages.length) {{
        return selectedMarkdown;
      }}
      return pages.map((page, index) => {{
        const pageNumber = page.page_number ?? page.unit_number ?? index + 1;
        const markdown = page.markdown || page.static?.text || '';
        return `## Page ${{pageNumber}}\\n\\n${{markdown}}`.trim();
      }}).join('\\n\\n---\\n\\n');
    }}

    function resultRenderKey(job) {{
      return `${{job.id}}:${{job.status}}:${{job.updated_at}}:${{activeTab}}`;
    }}

    function clamp(value, min, max) {{
      return Math.min(Math.max(value, min), max);
    }}

    function setWorkspaceWidths(pdfWidth, resultWidth) {{
      workspace.style.setProperty('--pdf-width', `${{Math.round(pdfWidth)}}px`);
      workspace.style.setProperty('--result-width', `${{Math.round(resultWidth)}}px`);
    }}

    function workspaceRailWidth() {{
      return workspace.classList.contains('rail-collapsed')
        ? 44
        : workspace.querySelector('.left-rail')?.getBoundingClientRect().width || 0;
    }}

    function workspaceContentAvailable() {{
      const workspaceRect = workspace.getBoundingClientRect();
      return Math.max(680, workspaceRect.width - workspaceRailWidth() - 6);
    }}

    function fitWorkspaceContentWidths() {{
      if (!workspace) {{
        return;
      }}
      const pdfWidth = document.querySelector('.pdf-stage')?.getBoundingClientRect().width || 1;
      const resultWidth = document.querySelector('.result-panel')?.getBoundingClientRect().width || 1;
      const ratio = clamp(pdfWidth / Math.max(pdfWidth + resultWidth, 1), 0.3, 0.7);
      const available = workspaceContentAvailable();
      setWorkspaceWidths(available * ratio, available * (1 - ratio));
    }}

    function resizePdfResult(clientX) {{
      if (!workspace) {{
        return;
      }}
      const workspaceRect = workspace.getBoundingClientRect();
      const available = workspaceContentAvailable();
      const relativeX = clientX - workspaceRect.left - workspaceRailWidth();
      const pdfWidth = clamp(relativeX, 320, available - 360);
      setWorkspaceWidths(pdfWidth, available - pdfWidth);
    }}

    async function refreshFiles() {{
      const response = await fetch('/api/files');
      const data = await response.json();
      const files = data.files || [];
      if (jobCount) {{
        jobCount.textContent = String(files.length);
      }}
      if (!files.length) {{
        selectedFileId = null;
        selectedJobId = null;
        selectedFile = null;
        selectedJob = null;
        jobList.innerHTML = '<div class="empty-state">No files yet.</div>';
        selectedTitle.textContent = 'Select a file';
        jobIdLabel.textContent = '-';
        updateDetailPanel(null, null);
        return;
      }}
      if (!selectedFileId || !files.some((file) => file.id === selectedFileId)) {{
        selectedFileId = files[0].id;
        selectedJobId = files[0].latest_job?.id || null;
      }}
      jobList.innerHTML = files.map((file) => `
        <div class="job-row ${{file.id === selectedFileId ? 'active' : ''}}" data-file-id="${{escapeHtml(file.id)}}" role="button" tabindex="0">
          <div class="file-row-main">
            <strong class="file-name" title="${{escapeHtml(file.filename)}}">${{escapeHtml(file.filename)}}</strong>
            <span>${{fileMeta(file)}}</span>
            ${{['queued', 'running'].includes(file.latest_job?.status) ? progressBar(file.latest_job) : ''}}
          </div>
          <button class="file-menu-button" type="button" data-file-menu-id="${{escapeHtml(file.id)}}" title="파일 메뉴" aria-label="파일 메뉴">⋯</button>
          <div class="file-menu" data-file-menu="${{escapeHtml(file.id)}}" ${{file.id === openMenuFileId ? '' : 'hidden'}}>
            <button type="button" data-file-action="details">상세정보</button>
            <button class="danger" type="button" data-file-action="delete">삭제</button>
          </div>
        </div>
      `).join('');
      const selected = files.find((file) => file.id === selectedFileId) || files[0];
      if (selected) {{
        selectedFileId = selected.id;
        await renderFile(selected);
      }}
    }}

    async function renderFile(file) {{
      selectedFile = file;
      selectedTitle.textContent = file.filename;
      downloadLinks.innerHTML = '';
      tabDownloadLinks.innerHTML = '';
      renderPdfPreview(file);
      const jobsResponse = await fetch(file.links.jobs);
      const jobsData = await jobsResponse.json();
      const jobs = jobsData.jobs || [];
      const job = jobs.find((item) => item.id === selectedJobId) || jobs[0] || file.latest_job;
      if (!job) {{
        selectedJob = null;
        selectedJobId = null;
        renderedResultKey = null;
        jobIdLabel.textContent = '-';
        previewBody.className = 'result-body';
        previewBody.innerHTML = '<div class="empty-state">업로드 완료. 실행을 누르면 이 파일의 새 파싱을 시작합니다.</div>';
        updateDetailPanel(file, null);
        return;
      }}
      selectedJobId = job.id;
      updateDetailPanel(file, job);
      await renderJob(job);
    }}

    async function renderJob(job) {{
      selectedJob = job;
      jobIdLabel.textContent = job.id;
      updateDetailPanel(selectedFile, job);
      if (job.status === 'failed') {{
        renderedResultKey = null;
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="error">${{escapeHtml(job.error || 'Parsing failed.')}}</div>`;
        return;
      }}
      if (job.status !== 'done') {{
        renderedResultKey = null;
        tabDownloadLinks.innerHTML = '';
        previewBody.className = 'result-body';
        previewBody.innerHTML = `
          <div class="empty-state">
            <div>Status: ${{escapeHtml(job.status)}}</div>
            ${{progressBar(job)}}
          </div>
        `;
        return;
      }}
      if (renderedResultKey === resultRenderKey(job)) {{
        return;
      }}
      tabDownloadLinks.innerHTML = `
        <a href="${{job.links.markdown}}" title="Markdown 다운로드" aria-label="Markdown 다운로드">
          <span aria-hidden="true">↓</span><span>MD</span>
        </a>
        <a href="${{job.links.json}}" title="JSON 다운로드" aria-label="JSON 다운로드">
          <span aria-hidden="true">↓</span><span>JSON</span>
        </a>
      `;
      const markdownResponse = await fetch(job.links.markdown);
      selectedMarkdown = await markdownResponse.text();
      const jsonResponse = await fetch(job.links.json);
      selectedJson = await jsonResponse.json();
      updateDetailPanel(selectedFile, job);
      renderResultTab();
    }}

    function renderResultTab() {{
      tabButtons.forEach((button) => {{
        button.classList.toggle('active', button.dataset.tab === activeTab);
      }});
      previewBody.className = 'result-body';
      if (!selectedJob || selectedJob.status !== 'done') {{
        return;
      }}
      renderedResultKey = resultRenderKey(selectedJob);
      if (activeTab === 'json') {{
        previewBody.innerHTML = `<pre>${{escapeHtml(JSON.stringify(selectedJson, null, 2))}}</pre>`;
        return;
      }}
      if (activeTab === 'html') {{
        previewBody.innerHTML = `<pre>${{escapeHtml(pageSeparatedMarkdown())}}</pre>`;
        return;
      }}
    }}

    async function uploadSelectedFile() {{
      if (!fileInput.files.length) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = '<div class="error">PDF 파일을 먼저 선택해 주세요.</div>';
        fileInput.click();
        return;
      }}
      uploadTrigger.disabled = true;
      uploadTrigger.textContent = '업로드 중...';
      try {{
        syncModelField();
        const response = await fetch('/api/files', {{
          method: 'POST',
          body: new FormData(form)
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || 'Upload failed');
        }}
        selectedFileId = data.file.id;
        selectedJobId = null;
        form.reset();
        selectedFileName.textContent = 'PDF, PNG, JPG (최대 100MB)';
        form.querySelector('input[name="trim"]').checked = true;
        form.querySelector('input[name="auto_slice"]').checked = true;
        syncModelField();
        await refreshFiles();
      }} catch (error) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="error">${{escapeHtml(error.message)}}</div>`;
      }} finally {{
        uploadTrigger.disabled = false;
        uploadTrigger.textContent = '업로드';
      }}
    }}

    async function parseSelectedFile() {{
      const button = form.querySelector('button[type="submit"]');
      if (!selectedFileId || !selectedFile) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = '<div class="error">먼저 PDF를 업로드해 주세요.</div>';
        fileInput.click();
        return;
      }}
      button.disabled = true;
      button.textContent = '실행 중...';
      try {{
        syncModelField();
        const response = await fetch(selectedFile.links.parse, {{
          method: 'POST',
          body: new FormData(form)
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || 'Parse failed');
        }}
        selectedJobId = data.job.id;
        selectedFileId = data.file.id;
        await refreshFiles();
      }} catch (error) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="error">${{escapeHtml(error.message)}}</div>`;
      }} finally {{
        button.disabled = false;
        button.textContent = '실행';
      }}
    }}

    async function showFileDetails(fileId) {{
      const file = selectedFile && selectedFile.id === fileId ? selectedFile : null;
      if (!file) {{
        return;
      }}
      const jobsResponse = await fetch(file.links.jobs);
      const jobsData = await jobsResponse.json();
      const jobs = jobsData.jobs || [];
      renderedResultKey = null;
      previewBody.className = 'result-body';
      previewBody.innerHTML = `
        <div class="empty-state">
          <strong>${{escapeHtml(file.filename)}}</strong><br>
          Parse jobs: ${{escapeHtml(jobs.length)}}<br>
          Latest: ${{escapeHtml(jobs[0]?.status || 'none')}}
        </div>
      `;
    }}

    async function deleteFile(fileId) {{
      const file = selectedFile && selectedFile.id === fileId ? selectedFile : null;
      const label = file?.filename || '이 파일';
      if (!window.confirm(`${{label}}을 삭제할까요? 관련 파싱 결과도 같이 삭제됩니다.`)) {{
        return;
      }}
      const response = await fetch(`/api/files/${{encodeURIComponent(fileId)}}`, {{ method: 'DELETE' }});
      const data = await response.json();
      if (!response.ok) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="error">${{escapeHtml(data.error || 'Delete failed')}}</div>`;
        return;
      }}
      if (selectedFileId === fileId) {{
        selectedFileId = null;
        selectedJobId = null;
        selectedFile = null;
        selectedJob = null;
        renderedPdfFileId = null;
      }}
      openMenuFileId = null;
      await refreshFiles();
    }}

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      await parseSelectedFile();
    }});

    sidebarToggle.addEventListener('click', () => {{
      const collapsed = workspace.classList.toggle('rail-collapsed');
      sidebarToggle.textContent = collapsed ? '›' : '‹';
      sidebarToggle.setAttribute('aria-label', collapsed ? 'Files 사이드바 펼치기' : 'Files 사이드바 접기');
      sidebarToggle.setAttribute('title', collapsed ? 'Files 사이드바 펼치기' : 'Files 사이드바 접기');
      fitWorkspaceContentWidths();
    }});

    workspaceResizers.forEach((resizer) => {{
      resizer.addEventListener('pointerdown', (event) => {{
        activeResize = resizer.dataset.resizer;
        resizer.classList.add('active');
        resizer.setPointerCapture(event.pointerId);
        resizePdfResult(event.clientX);
      }});
      resizer.addEventListener('pointermove', (event) => {{
        if (activeResize !== resizer.dataset.resizer) {{
          return;
        }}
        resizePdfResult(event.clientX);
      }});
      resizer.addEventListener('pointerup', (event) => {{
        activeResize = null;
        resizer.classList.remove('active');
        resizer.releasePointerCapture(event.pointerId);
      }});
      resizer.addEventListener('keydown', (event) => {{
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) {{
          return;
        }}
        event.preventDefault();
        const rect = workspace.getBoundingClientRect();
        const pdfRect = document.querySelector('.pdf-stage').getBoundingClientRect();
        const direction = event.key === 'ArrowLeft' ? -32 : 32;
        resizePdfResult(pdfRect.right + direction);
      }});
    }});

    railUploadTrigger.addEventListener('click', () => {{
      fileInput.click();
    }});

    modelSelect?.addEventListener('change', syncModelField);
    modelInput?.addEventListener('input', syncModelField);

    fileInput.addEventListener('change', async () => {{
      if (fileInput.files.length) {{
        await uploadSelectedFile();
      }}
    }});

    jobList.addEventListener('click', async (event) => {{
      const menuButton = event.target.closest('[data-file-menu-id]');
      if (menuButton) {{
        openMenuFileId = openMenuFileId === menuButton.dataset.fileMenuId ? null : menuButton.dataset.fileMenuId;
        await refreshFiles();
        return;
      }}
      const action = event.target.closest('[data-file-action]');
      if (action) {{
        const row = event.target.closest('[data-file-id]');
        if (!row) {{
          return;
        }}
        selectedFileId = row.dataset.fileId;
        openMenuFileId = null;
        await refreshFiles();
        if (action.dataset.fileAction === 'details') {{
          await showFileDetails(row.dataset.fileId);
        }} else if (action.dataset.fileAction === 'delete') {{
          await deleteFile(row.dataset.fileId);
        }}
        return;
      }}
      const row = event.target.closest('[data-file-id]');
      if (!row) {{
        return;
      }}
      selectedFileId = row.dataset.fileId;
      selectedJobId = null;
      openMenuFileId = null;
      await refreshFiles();
    }});

    tabButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        activeTab = button.dataset.tab;
        renderResultTab();
      }});
    }});

    syncModelField();
    refreshFiles();
    setInterval(refreshFiles, 2000);

  </script>
</body>
</html>"""


def run(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"vlm-parser demo listening on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vlm-parser demo server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
