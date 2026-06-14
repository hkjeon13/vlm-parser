from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


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


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "vlm-parser-demo/0.1"

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        config = load_demo_config(ROOT_DIR / ".env")
        self._send_html(render_page(config=config))

    def do_POST(self) -> None:
        if self.path != "/parse":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            fields, uploaded = self._parse_multipart()
            if uploaded is None or not uploaded.content:
                self._send_html(render_page(error="PDF 파일을 선택해 주세요."))
                return

            config = load_demo_config(ROOT_DIR / ".env")
            use_vlm = fields.get("use_vlm", "off") == "on"
            render_dpi = _int_field(fields.get("render_dpi", "180"), default=180)
            parser = make_parser(
                use_vlm=use_vlm,
                render_dpi=render_dpi,
                trim=fields.get("trim", "off") == "on",
                auto_slice=fields.get("auto_slice", "off") == "on",
                config=config,
            )

            if use_vlm and parser.vlm_client is None:
                self._send_html(
                    render_page(
                        config=config,
                        error=".env에 MODEL_API_KEY, MODEL_NAME, MODEL_BASE_URL을 모두 설정해야 VLM을 사용할 수 있습니다.",
                    )
                )
                return

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
                temp_pdf.write(uploaded.content)
                temp_pdf.flush()
                result = parser.parse(temp_pdf.name)

            payload = result.to_json()
            self._send_html(
                render_page(
                    config=config,
                    markdown=result.to_markdown(),
                    json_text=json.dumps(payload, ensure_ascii=False, indent=2),
                    filename=uploaded.filename,
                    used_vlm=parser.vlm.enabled,
                )
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_html(render_page(error=f"{type(exc).__name__}: {exc}"))

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
    used_vlm: bool = False,
) -> str:
    config = config or DemoConfig()
    has_vlm_config = bool(config.api_key and config.model and config.base_url)
    escaped_markdown = html.escape(markdown)
    escaped_json = html.escape(json_text)
    escaped_error = html.escape(error)
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
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 20px;
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: clamp(28px, 4vw, 44px); line-height: 1.05; }}
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
      header, form, .panes {{ grid-template-columns: 1fr; }}
      header {{ display: grid; }}
      .status {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>vlm-parser demo</h1>
      </div>
      <div class="status">
        <strong>{model_label}</strong>
        <span>VLM config: {vlm_status}</span>
      </div>
    </header>
    {error_section}
    <form action="/parse" method="post" enctype="multipart/form-data">
      <label>
        PDF
        <input name="pdf" type="file" accept="application/pdf,.pdf" required>
      </label>
      <div class="options">
        <label>
          Render DPI
          <input name="render_dpi" type="number" min="72" max="300" step="12" value="180">
        </label>
        <label class="check"><input name="trim" type="checkbox" checked> Trim margins</label>
        <label class="check"><input name="auto_slice" type="checkbox" checked> Auto slice pages</label>
        <label class="check"><input name="use_vlm" type="checkbox"> Use VLM rewrite</label>
        <button type="submit">Parse PDF</button>
      </div>
    </form>
    {result_section}
  </main>
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
