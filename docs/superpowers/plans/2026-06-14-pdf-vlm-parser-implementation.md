# PDF VLM Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 설계서 기준으로 PDF 정적 parsing, render image trim/chunk, optional VLM rewriting, JSON/Markdown 출력을 제공하는 Python 라이브러리 MVP를 구현한다.

**Architecture:** Public `PdfParser`는 공통 `Parser`에 `PdfDocumentAdapter`를 주입하는 thin wrapper다. Core pipeline은 document adapter가 제공하는 `DocumentUnit`을 병렬 처리하고, unit 내부 chunk는 순차 rewriting한다. PDF 전용 로직은 `documents/pdf`에 두고, image preprocessing, VLM, output assembly는 공통 모듈로 둔다.

**Tech Stack:** Python 3.11+, PyMuPDF (`pymupdf`), Pillow, httpx, pytest.

---

### Task 1: Package Skeleton And Public API

**Files:**
- Create: `pyproject.toml`
- Create: `src/vlm_parser/__init__.py`
- Create: `src/vlm_parser/parser.py`
- Create: `src/vlm_parser/core/options.py`
- Create: `tests/test_public_api.py`

- [ ] **Step 1: Write the failing test**

```python
from vlm_parser import PdfParser, ParseOptions, VlmOptions

def test_public_api_exposes_pdf_parser_and_options():
    parser = PdfParser(options=ParseOptions(render_dpi=144), vlm=VlmOptions(enabled=False))
    assert parser.options.render_dpi == 144
    assert parser.vlm.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_public_api.py`
Expected: FAIL because `vlm_parser` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create dataclass options and a `PdfParser` wrapper that stores options.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_public_api.py`
Expected: PASS.

### Task 2: Core Models And Output Serialization

**Files:**
- Create: `src/vlm_parser/core/models.py`
- Create: `src/vlm_parser/output/assembler.py`
- Test: `tests/test_output_models.py`

- [ ] **Step 1: Write failing tests for JSON and Markdown**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement dataclasses and `ParseResult.to_json()`, `to_markdown()`**
- [ ] **Step 4: Run tests and confirm pass**

### Task 3: Image Trim And Chunking

**Files:**
- Create: `src/vlm_parser/image/preprocess.py`
- Create: `src/vlm_parser/image/chunker.py`
- Test: `tests/test_image_processing.py`

- [ ] **Step 1: Write failing tests for white margin trim, corner-content fallback, and horizontal blank-band chunking**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement conservative trim and chunker**
- [ ] **Step 4: Run tests and confirm pass**

### Task 4: VLM Rewriter And Global Concurrency

**Files:**
- Create: `src/vlm_parser/vlm/client.py`
- Create: `src/vlm_parser/vlm/rewriter.py`
- Create: `src/vlm_parser/vlm/concurrency.py`
- Test: `tests/test_vlm_rewriter.py`

- [ ] **Step 1: Write failing tests that verify chunks are processed sequentially and global limiter is used**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Implement protocol-based VLM client, limiter, and page/unit rewriter**
- [ ] **Step 4: Run tests and confirm pass**

### Task 5: PDF Adapter And Static Parse Pipeline

**Files:**
- Create: `src/vlm_parser/documents/base.py`
- Create: `src/vlm_parser/documents/pdf/adapter.py`
- Create: `src/vlm_parser/documents/pdf/document.py`
- Create: `src/vlm_parser/documents/pdf/static_extractor.py`
- Create: `src/vlm_parser/documents/pdf/renderer.py`
- Create: `src/vlm_parser/core/parser.py`
- Create: `src/vlm_parser/core/pipeline.py`
- Test: `tests/test_pdf_static_parse.py`

- [ ] **Step 1: Write failing test that creates a small PDF with PyMuPDF and parses it through `PdfParser`**
- [ ] **Step 2: Run test and confirm failure**
- [ ] **Step 3: Implement PDF adapter and static pipeline**
- [ ] **Step 4: Run test and confirm pass**

### Task 6: End-To-End Verification

**Files:**
- Modify: implementation files as needed.

- [ ] **Step 1: Run full tests**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Compile source**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests`
Expected: exit code 0.

- [ ] **Step 3: Review git diff**

Run: `git diff --stat`
Expected: only implementation, tests, and plan files are changed by this task.
