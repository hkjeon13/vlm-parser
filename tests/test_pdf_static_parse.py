import fitz

from vlm_parser import PdfParser, VlmOptions


def test_pdf_parser_extracts_static_text_to_json_and_markdown(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF")
    doc.save(pdf_path)
    doc.close()

    result = PdfParser().parse(pdf_path)
    data = result.to_json()

    assert data["source"]["document_type"] == "pdf"
    assert data["source"]["page_count"] == 1
    assert data["pages"][0]["page_number"] == 1
    assert data["pages"][0]["static"]["text"].strip() == "Hello PDF"
    assert data["pages"][0]["render"]["original"]["width_px"] > 0
    assert data["pages"][0]["render"]["chunks"][0]["id"] == "p1-c1"
    assert "Hello PDF" in result.to_markdown()


def test_pdf_parser_reports_page_progress(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    first = doc.new_page()
    first.insert_text((72, 72), "First page")
    second = doc.new_page()
    second.insert_text((72, 72), "Second page")
    doc.save(pdf_path)
    doc.close()
    progress = []

    PdfParser(progress_callback=lambda current, total, label: progress.append((current, total, label))).parse(pdf_path)

    assert progress[0] == (0, 2, "Preparing 2 pages")
    assert progress[-1] == (2, 2, "Parsed page 2 of 2")


class FakeVlmClient:
    def rewrite_chunk(self, request):
        return f"rewritten {request.static.text.strip()}"


def test_pdf_parser_can_rewrite_with_injected_vlm_client(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello VLM")
    doc.save(pdf_path)
    doc.close()

    result = PdfParser(vlm=VlmOptions(enabled=True, model="fake-model"), vlm_client=FakeVlmClient()).parse(pdf_path)
    data = result.to_json()

    assert data["pages"][0]["vlm"]["status"] == "success"
    assert data["pages"][0]["markdown"] == "rewritten Hello VLM"
    assert result.to_markdown() == "rewritten Hello VLM"
