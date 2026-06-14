# Async Demo Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add asynchronous PDF parsing jobs to the demo, with upload APIs, status polling, result downloads, and a PDF/Markdown review UI.

**Architecture:** Keep the no-extra-dependency demo server and add an in-memory `JobStore` guarded by a lock. Uploaded PDFs are copied into a temporary job directory, worker threads run parsing in the background, and API endpoints expose job status plus downloadable JSON/Markdown/PDF artifacts.

**Tech Stack:** Python standard library `http.server`, `threading`, `tempfile`, JSON APIs, existing `PdfParser` and VLM client wiring.

---

### Task 1: Job Model and Store

**Files:**
- Modify: `demo/server.py`
- Test: `tests/test_demo_server.py`

- [x] Write tests for creating a queued job, preserving the uploaded PDF, and serializing status.
- [x] Implement `JobOptions`, `ParseJob`, and `JobStore`.
- [x] Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_demo_server.py`.

### Task 2: Background Processing

**Files:**
- Modify: `demo/server.py`
- Test: `tests/test_demo_server.py`

- [x] Write tests for marking a job done and exposing JSON/Markdown results.
- [x] Implement `process_job` with injectable parser factory for tests.
- [x] Keep VLM config validation before enqueueing VLM jobs.

### Task 3: HTTP API and UI

**Files:**
- Modify: `demo/server.py`
- Modify: `README.md`

- [x] Add `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/result.json`, `GET /api/jobs/{id}/result.md`, and `GET /api/jobs/{id}/source.pdf`.
- [x] Replace the synchronous form result view with a polling job list and PDF/Markdown preview.
- [x] Document the asynchronous API endpoints.

### Task 4: Verification and Deployment

**Files:**
- Verify: local workspace and remote `/data/psyche/Projects/vlm-parser`

- [x] Run local tests and compileall.
- [x] Browser-check the demo page.
- [x] Commit and push.
- [x] Pull on `server-4096`, restart the demo server, and verify HTTP 200.
