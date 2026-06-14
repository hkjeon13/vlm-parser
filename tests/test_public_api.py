from vlm_parser import PdfParser, ParseOptions, VlmOptions


def test_public_api_exposes_pdf_parser_and_options():
    parser = PdfParser(
        options=ParseOptions(render_dpi=144),
        vlm=VlmOptions(enabled=False),
    )

    assert parser.options.render_dpi == 144
    assert parser.vlm.enabled is False
