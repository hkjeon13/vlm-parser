# vlm-parser

PDF를 먼저 지원하는 VLM 기반 문서 parser 라이브러리입니다.

기본은 PyMuPDF 기반 정적 parsing이며, 옵션으로 OpenAI-compatible VLM API를 사용해 페이지 렌더 이미지 chunk와 정적 추출 텍스트를 바탕으로 Markdown을 rewriting할 수 있습니다.

현재 구현 범위:

- PDF static text/block extraction
- PDF page render image 생성
- Render image margin trim
- Horizontal blank band 기반 chunk 분할
- JSON + Markdown 출력
- OpenAI-compatible VLM client 기반 chunk rewriting
- 추후 확장자를 추가하기 위한 `core` + `documents/pdf` adapter 구조

## 설치

GitHub에서 바로 설치할 수 있습니다.

```bash
pip install "git+https://github.com/hkjeon13/vlm-parser.git"
```

특정 branch나 commit을 고정하려면 다음처럼 설치합니다.

```bash
pip install "git+https://github.com/hkjeon13/vlm-parser.git@main"
```

Python 3.11 이상이 필요합니다.

## 기본 사용법

VLM 없이 PyMuPDF 정적 parsing만 사용할 수 있습니다.

```python
from vlm_parser import PdfParser

parser = PdfParser()
result = parser.parse("sample.pdf")

data = result.to_json()
markdown = result.to_markdown()

result.save_json("out/result.json")
result.save_markdown("out/result.md")
```

`result.to_json()`은 다음과 같은 상위 구조를 반환합니다.

```python
{
    "schema_version": "0.1",
    "source": {
        "path": "sample.pdf",
        "filename": "sample.pdf",
        "document_type": "pdf",
        "unit_count": 1,
        "page_count": 1,
        "parser": {
            "name": "vlm-parser",
            "version": "0.1.0",
        },
    },
    "document": {
        "markdown": "...",
        "metadata": {...},
    },
    "pages": [
        {
            "unit_type": "page",
            "page_number": 1,
            "static": {...},
            "render": {...},
            "vlm": None,
            "markdown": "...",
        }
    ],
}
```

## Parse 옵션

```python
from vlm_parser import PdfParser, ParseOptions

parser = PdfParser(
    options=ParseOptions(
        render_dpi=180,
        trim=True,
        auto_slice=True,
        max_page_workers=4,
    )
)

result = parser.parse("sample.pdf")
```

옵션 의미:

- `render_dpi`: PDF page render image DPI.
- `trim`: 렌더 이미지의 흰색 또는 단색 여백 trim 여부.
- `auto_slice`: horizontal blank band 기반 chunk 분할 여부.
- `max_page_workers`: 추후 page 병렬 처리에 사용할 worker 수.

## VLM 사용법

VLM rewriting을 사용하려면 `VlmOptions(enabled=True)`와 VLM client를 함께 전달합니다.

```python
from vlm_parser import PdfParser, VlmOptions
from vlm_parser.vlm.client import OpenAICompatibleVlmClient

vlm_client = OpenAICompatibleVlmClient(
    base_url="https://api.example.com/v1",
    api_key="YOUR_API_KEY",
    model="your-vision-model",
    timeout_seconds=60,
)

parser = PdfParser(
    vlm=VlmOptions(
        enabled=True,
        model="your-vision-model",
        max_concurrency=4,
    ),
    vlm_client=vlm_client,
)

result = parser.parse("sample.pdf")
result.save_markdown("out/result.md")
```

VLM 요청은 OpenAI-compatible `/chat/completions` 형식으로 전송됩니다. 각 PDF page는 render image chunk로 나뉘고, 같은 page 내부 chunk는 순차적으로 rewriting됩니다. Chunk rewriting에는 이전 chunk Markdown이 context로 전달됩니다.

## 출력 저장

```python
result.save_json("out/result.json")
result.save_markdown("out/result.md")
```

또는 메모리에서 바로 사용할 수 있습니다.

```python
json_data = result.to_json()
markdown_text = result.to_markdown()
```

## 데모 페이지

저장소에는 업로드한 PDF를 바로 테스트할 수 있는 간단한 웹 데모가 포함되어 있습니다.

```bash
python3 demo/server.py --host 0.0.0.0 --port 7860
```

브라우저에서 `http://localhost:7860`으로 접속한 뒤 PDF를 업로드하면 Markdown과 JSON 결과를 함께 확인할 수 있습니다.
파싱은 비동기 job으로 실행되며, 화면에서 job 목록과 상태를 확인하고 완료 후 JSON 또는 Markdown을 다운로드할 수 있습니다.

VLM rewriting을 테스트하려면 저장소 루트의 `.env`에 다음 값을 설정하고 데모에서 `Use VLM rewrite`를 켜면 됩니다.

```bash
MODEL_API_KEY=<API_KEY>
MODEL_NAME=google/gemini-2.5-flash
MODEL_BASE_URL=https://openrouter.ai/api/v1
```

`MODEL_BASE_URL`이 `https://openrouter.ai/api/v1`이면 데모가 OpenRouter 모델 목록에서 image input을 지원하는 text-output 모델을 불러와 선택 UI를 표시합니다. 완료된 job에는 총 파싱 시간, 페이지당 평균 시간, token 사용량, OpenRouter pricing 기반 API 비용이 함께 표시됩니다. 다른 OpenAI-compatible base URL을 쓰는 경우에는 모델 ID를 수동으로 사용하며 비용 계산은 생략됩니다.

데모 서버는 같은 기능을 API로도 제공합니다.

```bash
curl -F "pdf=@sample.pdf" \
  -F "render_dpi=180" \
  -F "trim=on" \
  -F "auto_slice=on" \
  http://localhost:7860/api/jobs

curl http://localhost:7860/api/jobs
curl http://localhost:7860/api/jobs/<job_id>
curl -o result.json http://localhost:7860/api/jobs/<job_id>/result.json
curl -o result.md http://localhost:7860/api/jobs/<job_id>/result.md
```

## 개발 환경

저장소를 직접 clone해서 개발할 때:

```bash
git clone https://github.com/hkjeon13/vlm-parser.git
cd vlm-parser
pip install -e ".[dev]"
```

테스트:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

컴파일 확인:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests
```

## 현재 제한

- 1차 구현 범위는 PDF입니다.
- OCR 전용 엔진은 아직 포함되어 있지 않습니다.
- VLM을 켜려면 현재는 `vlm_client`를 명시적으로 주입해야 합니다.
- Table 전용 구조 복원 엔진은 아직 없습니다.
- Document-level VLM cleanup은 아직 없습니다.
