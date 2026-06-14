from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_parser import ParseOptions, PdfParser, VlmOptions  # noqa: E402
from vlm_parser.vlm.client import OpenAICompatibleVlmClient  # noqa: E402


@dataclass(frozen=True, slots=True)
class DemoConfig:
    api_key: str = ""
    model: str = ""
    base_url: str = ""


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


@dataclass(slots=True)
class ParseJob:
    id: str
    filename: str
    source_path: Path
    options: JobOptions
    status: str = "uploaded"
    created_at: float = 0.0
    updated_at: float = 0.0
    error: str = ""
    result_json: dict | None = None
    markdown: str = ""


class JobStore:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, ParseJob] = {}
        self._lock = threading.Lock()

    def create(self, uploaded: UploadedFile, options: JobOptions) -> ParseJob:
        job_id = uuid4().hex
        job_dir = self.root_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        source_path = job_dir / "source.pdf"
        source_path.write_bytes(uploaded.content)
        now = time.time()
        job = ParseJob(
            id=job_id,
            filename=uploaded.filename or "uploaded.pdf",
            source_path=source_path,
            options=options,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> ParseJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[ParseJob]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )

    def mark_running(self, job_id: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "running"
            job.updated_at = time.time()
            return job

    def mark_queued(self, job_id: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "queued"
            job.error = ""
            job.updated_at = time.time()
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
            job.updated_at = time.time()
            return job

    def mark_failed(self, job_id: str, error: str) -> ParseJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "failed"
            job.error = error
            job.updated_at = time.time()
            return job

    def to_summary(self, job: ParseJob) -> dict:
        return {
            "id": job.id,
            "filename": job.filename,
            "status": job.status,
            "error": job.error,
            "use_vlm": job.options.use_vlm,
            "render_dpi": job.options.render_dpi,
            "trim": job.options.trim,
            "auto_slice": job.options.auto_slice,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "has_result": job.result_json is not None,
            "links": {
                "self": f"/api/jobs/{job.id}",
                "parse": f"/api/jobs/{job.id}/parse",
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
        parser = parser_factory(
            use_vlm=job.options.use_vlm,
            render_dpi=job.options.render_dpi,
            trim=job.options.trim,
            auto_slice=job.options.auto_slice,
            config=config,
        )
        result = parser.parse(job.source_path)
        store.mark_done(job.id, result.to_json(), result.to_markdown())
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
            "jobs": "/api/jobs",
            "job_detail": "/api/jobs/{job_id}",
            "parse": "/api/jobs/{job_id}/parse",
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
        self._send_html(render_page(config=config))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/jobs":
            self._handle_create_job()
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "parse":
            self._handle_parse_job(parts[2])
            return
        if path != "/parse":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            job, error = self._enqueue_job()
            config = load_demo_config(ROOT_DIR / ".env")
            if error:
                self._send_html(render_page(config=config, error=error))
            else:
                self._send_html(render_page(config=config, notice=f"{job.filename} 작업을 등록했습니다."))
        except Exception as exc:
            traceback.print_exc()
            self._send_html(render_page(error=f"{type(exc).__name__}: {exc}"))

    def _handle_create_job(self) -> None:
        try:
            job, error = self._enqueue_job()
            if error:
                self._send_json({"error": error}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                {"job": JOB_STORE.to_summary(job)},
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
            if job.options.use_vlm and build_vlm_client(config) is None:
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

    def _enqueue_job(self) -> tuple[ParseJob | None, str]:
        fields, uploaded = self._parse_multipart()
        if uploaded is None or not uploaded.content:
            return None, "PDF 파일을 선택해 주세요."

        config = load_demo_config(ROOT_DIR / ".env")
        use_vlm = fields.get("use_vlm", "off") == "on"
        if use_vlm and build_vlm_client(config) is None:
            return None, ".env에 MODEL_API_KEY, MODEL_NAME, MODEL_BASE_URL을 모두 설정해야 VLM을 사용할 수 있습니다."

        job = JOB_STORE.create(
            uploaded,
            JobOptions(
                use_vlm=use_vlm,
                render_dpi=_int_field(fields.get("render_dpi", "180"), default=180),
                trim=fields.get("trim", "off") == "on",
                auto_slice=fields.get("auto_slice", "off") == "on",
            ),
        )
        return job, ""

    def _handle_get_api(self, path: str) -> None:
        if path in {"/api", "/api/"}:
            self._send_json(api_index_payload())
            return

        if path == "/api/jobs":
            self._send_json({"jobs": [JOB_STORE.to_summary(job) for job in JOB_STORE.list()]})
            return

        parts = path.strip("/").split("/")
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

    def _parse_multipart(self) -> tuple[dict[str, str], UploadedFile | None]:
        length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
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


def render_page(
    *,
    config: DemoConfig | None = None,
    markdown: str = "",
    json_text: str = "",
    filename: str = "",
    error: str = "",
    notice: str = "",
    used_vlm: bool = False,
) -> str:
    config = config or DemoConfig()
    has_vlm_config = bool(config.api_key and config.model and config.base_url)
    escaped_markdown = html.escape(markdown)
    escaped_json = html.escape(json_text)
    escaped_error = html.escape(error)
    escaped_notice = html.escape(notice)
    escaped_filename = html.escape(filename)
    model_label = html.escape(config.model or "not configured")
    vlm_status = "ready" if has_vlm_config else "missing .env values"
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
      grid-template-rows: 60px minmax(0, 1fr);
      height: 100vh;
      min-width: 1080px;
      overflow: hidden;
    }}
    .topbar {{
      align-items: center;
      margin: 0;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }}
    .topbar .status {{
      display: none;
    }}
    .parse-step {{
      position: absolute;
      left: 50%;
      top: 13px;
      transform: translateX(-50%);
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      border: 2px solid var(--accent);
      border-radius: 999px;
      background: #fff;
      color: var(--accent-strong);
      padding: 4px 12px 4px 6px;
      font-weight: 800;
      box-shadow: 0 0 0 4px rgba(91, 92, 246, 0.12);
    }}
    .parse-step span {{
      display: grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: var(--accent);
      color: #fff;
      font-size: 12px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 20px;
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 16px; line-height: 1.05; }}
    h2 {{ font-size: 16px; }}
    .status {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 260px;
      color: var(--muted);
    }}
    .status strong {{ color: var(--ink); display: block; }}
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
      position: absolute;
      top: 10px;
      right: 24px;
      z-index: 2;
      display: flex;
      align-items: center;
      grid-template-columns: none;
      margin: 0;
      border: 0;
      border-radius: 0;
      padding: 0;
      background: transparent;
      gap: 8px;
    }}
    .upload-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      min-height: 34px;
      border: 2px solid var(--accent);
      border-radius: 7px;
      background: #fff;
      color: var(--accent);
      padding: 0 12px;
      font-weight: 800;
    }}
    #pdf-input {{
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }}
    .selected-file-name {{
      max-width: 180px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .upload-bar .options {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .upload-bar .options > label:first-child {{
      display: none;
    }}
    .upload-bar .field-options {{
      display: none;
    }}
    .upload-bar .check {{
      display: none;
    }}
    .upload-bar .options label.check:nth-of-type(4) {{
      display: flex;
      min-height: 34px;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }}
    .upload-bar button[type="submit"] {{
      min-height: 34px;
      border-radius: 7px;
      background: var(--accent);
      color: #fff;
      padding: 0 14px;
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
      display: grid;
      grid-template-columns: 304px minmax(420px, 1fr) minmax(520px, 42vw);
      gap: 0;
      align-items: stretch;
      min-height: 0;
      height: 100%;
    }}
    .jobs {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .left-rail {{
      border-width: 0 1px 0 0;
      border-radius: 0;
      background: #f8fafc;
      min-height: 0;
    }}
    .jobs header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .job-list {{
      display: grid;
      align-content: start;
      max-height: none;
      height: 100%;
      overflow: auto;
      padding: 12px;
      gap: 8px;
    }}
    .job-row {{
      display: grid;
      gap: 4px;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      color: var(--ink);
      text-align: left;
      min-height: 0;
      padding: 11px 14px;
    }}
    .job-row:hover {{ background: #fff; }}
    .job-row.active {{ background: #eef2ff; border-color: #8b8cff; box-shadow: inset 3px 0 0 #5b5cf6; }}
    .job-row strong {{
      display: block;
      overflow-wrap: anywhere;
      font-weight: 750;
    }}
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
    .pdf-stage {{
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(0, 1fr) 48px;
      background: #cfd7e3;
      border-right: 1px solid var(--line);
    }}
    .pdf-canvas {{
      min-height: 0;
      overflow: auto;
      padding: 10px 12px 16px;
      display: grid;
      place-items: start center;
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
      width: min(840px, calc(100% - 8px));
      height: calc(100vh - 170px);
      min-height: 560px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(15,23,42,0.16);
    }}
    .pdf-controls {{
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr) 120px;
      align-items: center;
      gap: 16px;
      padding: 8px 18px;
      background: #e8edf4;
      border-top: 1px solid #c6d0df;
    }}
    .zoom-track {{
      height: 8px;
      border-radius: 99px;
      background: linear-gradient(90deg, #111827 0 52%, #b7c1cf 52% 100%);
    }}
    .page-chip {{
      justify-self: end;
      color: #475569;
      font-weight: 700;
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
      color: var(--ink);
      transform: translateY(1px);
    }}
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
    @media (max-width: 780px) {{
      header, form, .panes, .workspace, .review-grid {{ grid-template-columns: 1fr; }}
      header {{ display: grid; }}
      .status {{ min-width: 0; }}
      body {{ overflow: auto; }}
      .app-shell {{ min-width: 0; height: auto; }}
      .pdf-shell {{ width: 100%; height: 560px; }}
    }}
  </style>
</head>
<body>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <h1>vlm-parser demo</h1>
      </div>
      <div class="parse-step"><span>1</span> Parse</div>
      <div class="status">
        <strong>{model_label}</strong>
        <span>VLM config: {vlm_status}</span>
      </div>
    </header>
    {error_section}
    {notice_section}
    <form id="upload-form" class="upload-bar" action="/api/jobs" method="post" enctype="multipart/form-data">
      <button id="upload-trigger" class="upload-button" type="button">업로드</button>
      <input id="pdf-input" name="pdf" type="file" accept="application/pdf,.pdf">
      <span id="selected-file-name" class="selected-file-name">선택된 파일 없음</span>
      <div class="options">
        <label>
          Render DPI
          <input name="render_dpi" type="number" min="72" max="300" step="12" value="180">
        </label>
        <label class="check"><input name="trim" type="checkbox" checked> Trim margins</label>
        <label class="check"><input name="auto_slice" type="checkbox" checked> Auto slice pages</label>
        <label class="check"><input name="use_vlm" type="checkbox"> Use VLM rewrite</label>
        <button type="submit">실행</button>
      </div>
    </form>
    <section class="workspace">
      <aside class="jobs left-rail">
        <header>
          <h2>Jobs</h2>
          <span id="job-count" class="badge">0</span>
        </header>
        <div id="job-list" class="job-list">
          <div class="empty-state">No jobs yet.</div>
        </div>
      </aside>
      <section class="pdf-stage">
        <div id="pdf-canvas" class="pdf-canvas">
          <div class="pdf-empty">PDF를 업로드하면 이 영역에서 원문을 확인할 수 있습니다.</div>
        </div>
        <div class="pdf-controls">
          <button class="ghost-button" type="button">↻</button>
          <div class="zoom-track" aria-hidden="true"></div>
          <span class="page-chip"><strong>1</strong> / <span id="page-total">-</span>⌄</span>
        </div>
      </section>
      <section class="preview result-panel">
        <div class="preview-toolbar">
          <strong id="selected-title">Select a completed job</strong>
          <div id="download-links" class="download-links"></div>
        </div>
        <div class="result-tabs" role="tablist">
          <button id="tab-preview" class="active" type="button" data-tab="preview">미리보기</button>
          <button id="tab-html" type="button" data-tab="html">HTML</button>
          <button id="tab-json" type="button" data-tab="json">JSON</button>
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
    const uploadTrigger = document.getElementById('upload-trigger');
    const fileInput = document.getElementById('pdf-input');
    const selectedFileName = document.getElementById('selected-file-name');
    const jobList = document.getElementById('job-list');
    const jobCount = document.getElementById('job-count');
    const selectedTitle = document.getElementById('selected-title');
    const downloadLinks = document.getElementById('download-links');
    const previewBody = document.getElementById('preview-body');
    const pdfCanvas = document.getElementById('pdf-canvas');
    const jobIdLabel = document.getElementById('job-id-label');
    const tabButtons = Array.from(document.querySelectorAll('[data-tab]'));
    let selectedJobId = null;
    let selectedJob = null;
    let selectedMarkdown = '';
    let selectedJson = null;
    let renderedPdfJobId = null;
    let activeTab = 'preview';

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
      return `<span class="badge ${{escapeHtml(job.status)}}">${{escapeHtml(job.status)}}</span>`;
    }}

    function pdfPreviewUrl(job) {{
      return `${{job.links.source_pdf}}#view=FitH&zoom=page-width&navpanes=0`;
    }}

    function renderPdfPreview(job) {{
      if (renderedPdfJobId === job.id) {{
        return;
      }}
      renderedPdfJobId = job.id;
      pdfCanvas.innerHTML = `
        <div class="pdf-shell">
          <iframe class="pdf-frame" src="${{pdfPreviewUrl(job)}}" title="PDF preview"></iframe>
        </div>
      `;
    }}

    async function refreshJobs() {{
      const response = await fetch('/api/jobs');
      const data = await response.json();
      const jobs = data.jobs || [];
      jobCount.textContent = String(jobs.length);
      if (!jobs.length) {{
        jobList.innerHTML = '<div class="empty-state">No jobs yet.</div>';
        return;
      }}
      if (!selectedJobId) {{
        selectedJobId = jobs[0].id;
      }}
      jobList.innerHTML = jobs.map((job) => `
        <button class="job-row ${{job.id === selectedJobId ? 'active' : ''}}" data-job-id="${{escapeHtml(job.id)}}">
          <strong>${{escapeHtml(job.filename)}}</strong>
          <span>${{statusLabel(job)}} VLM: ${{job.use_vlm ? 'on' : 'off'}} · DPI ${{job.render_dpi}}</span>
          ${{job.error ? `<span>${{escapeHtml(job.error)}}</span>` : ''}}
        </button>
      `).join('');
      const selected = jobs.find((job) => job.id === selectedJobId) || jobs[0];
      if (selected) {{
        selectedJobId = selected.id;
        await renderJob(selected);
      }}
    }}

    async function renderJob(job) {{
      selectedJob = job;
      selectedTitle.textContent = job.filename;
      jobIdLabel.textContent = job.id;
      downloadLinks.innerHTML = '';
      renderPdfPreview(job);
      if (job.status === 'failed') {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="error">${{escapeHtml(job.error || 'Parsing failed.')}}</div>`;
        return;
      }}
      if (job.status === 'uploaded') {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = '<div class="empty-state">업로드 완료. 실행을 누르면 파싱을 시작합니다.</div>';
        return;
      }}
      if (job.status !== 'done') {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="empty-state">Status: ${{escapeHtml(job.status)}}</div>`;
        return;
      }}
      downloadLinks.innerHTML = `
        <a href="${{job.links.markdown}}" title="Markdown 다운로드">MD</a>
        <a href="${{job.links.json}}" title="JSON 다운로드">JSON</a>
      `;
      const markdownResponse = await fetch(job.links.markdown);
      selectedMarkdown = await markdownResponse.text();
      const jsonResponse = await fetch(job.links.json);
      selectedJson = await jsonResponse.json();
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
      if (activeTab === 'json') {{
        previewBody.innerHTML = `<pre>${{escapeHtml(JSON.stringify(selectedJson, null, 2))}}</pre>`;
        return;
      }}
      if (activeTab === 'html') {{
        previewBody.innerHTML = `<pre>${{escapeHtml(selectedMarkdown)}}</pre>`;
        return;
      }}
      previewBody.innerHTML = `
        <div class="result-meta">
          <span>페이지 1</span>
        </div>
        <pre>${{escapeHtml(selectedMarkdown)}}</pre>
      `;
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
        const response = await fetch('/api/jobs', {{
          method: 'POST',
          body: new FormData(form)
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || 'Upload failed');
        }}
        selectedJobId = data.job.id;
        form.reset();
        selectedFileName.textContent = '선택된 파일 없음';
        form.querySelector('input[name="trim"]').checked = true;
        form.querySelector('input[name="auto_slice"]').checked = true;
        await refreshJobs();
      }} catch (error) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="error">${{escapeHtml(error.message)}}</div>`;
      }} finally {{
        uploadTrigger.disabled = false;
        uploadTrigger.textContent = '업로드';
      }}
    }}

    async function parseSelectedJob() {{
      const button = form.querySelector('button[type="submit"]');
      if (!selectedJobId) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = '<div class="error">먼저 PDF를 업로드해 주세요.</div>';
        fileInput.click();
        return;
      }}
      button.disabled = true;
      button.textContent = '실행 중...';
      try {{
        const response = await fetch(`/api/jobs/${{encodeURIComponent(selectedJobId)}}/parse`, {{
          method: 'POST'
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || 'Parse failed');
        }}
        selectedJobId = data.job.id;
        await refreshJobs();
      }} catch (error) {{
        previewBody.className = 'result-body';
        previewBody.innerHTML = `<div class="error">${{escapeHtml(error.message)}}</div>`;
      }} finally {{
        button.disabled = false;
        button.textContent = '실행';
      }}
    }}

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      await parseSelectedJob();
    }});

    uploadTrigger.addEventListener('click', () => {{
      fileInput.click();
    }});

    fileInput.addEventListener('change', async () => {{
      selectedFileName.textContent = fileInput.files.length
        ? fileInput.files[0].name
        : '선택된 파일 없음';
      if (fileInput.files.length) {{
        await uploadSelectedFile();
      }}
    }});

    jobList.addEventListener('click', async (event) => {{
      const row = event.target.closest('[data-job-id]');
      if (!row) {{
        return;
      }}
      selectedJobId = row.dataset.jobId;
      await refreshJobs();
    }});

    tabButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        activeTab = button.dataset.tab;
        renderResultTab();
      }});
    }});

    refreshJobs();
    setInterval(refreshJobs, 2000);
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
