from vlm_parser.core.models import (
    DocumentMetadata,
    DocumentResult,
    PageResult,
    ParserInfo,
    ParseResult,
    SourceInfo,
    StaticUnitResult,
)


def test_parse_result_serializes_json_and_markdown():
    result = ParseResult(
        source=SourceInfo(
            path="sample.pdf",
            filename="sample.pdf",
            file_size_bytes=100,
            unit_count=1,
            page_count=1,
            document_type="pdf",
            parser=ParserInfo(name="vlm-parser", version="0.1.0"),
        ),
        document=DocumentResult(
            markdown="# Title\n\nBody",
            metadata=DocumentMetadata(title="Sample"),
        ),
        pages=[
            PageResult(
                unit_id="p1",
                unit_number=1,
                page_number=1,
                width_pt=595.0,
                height_pt=842.0,
                rotation=0,
                static=StaticUnitResult(text="Title\nBody"),
                markdown="# Title\n\nBody",
            )
        ],
    )

    data = result.to_json()

    assert data["schema_version"] == "0.1"
    assert data["source"]["document_type"] == "pdf"
    assert data["source"]["unit_count"] == 1
    assert data["pages"][0]["unit_type"] == "page"
    assert data["pages"][0]["static"]["text"] == "Title\nBody"
    assert result.to_markdown() == "# Title\n\nBody"


def test_parse_result_saves_json_and_markdown(tmp_path):
    result = ParseResult(
        source=SourceInfo(
            path="sample.pdf",
            filename="sample.pdf",
            file_size_bytes=100,
            unit_count=0,
            page_count=0,
            document_type="pdf",
            parser=ParserInfo(name="vlm-parser", version="0.1.0"),
        ),
        document=DocumentResult(markdown="Saved markdown", metadata=DocumentMetadata()),
        pages=[],
    )
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"

    result.save_json(json_path)
    result.save_markdown(markdown_path)

    assert '"schema_version": "0.1"' in json_path.read_text()
    assert markdown_path.read_text() == "Saved markdown"
