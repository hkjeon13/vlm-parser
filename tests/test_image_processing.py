from PIL import Image, ImageDraw

from vlm_parser.image.chunker import ChunkOptions, split_horizontal_chunks
from vlm_parser.image.preprocess import TrimOptions, trim_uniform_margins


def test_trim_uniform_margins_removes_white_border():
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 79, 69), fill="black")

    result = trim_uniform_margins(image, TrimOptions(background_tolerance=2))

    assert result.applied is True
    assert result.bbox_in_original == (20, 10, 80, 70)
    assert result.image.size == (60, 60)


def test_trim_uniform_margins_skips_when_corner_has_content():
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 6, 6), fill="black")
    draw.rectangle((20, 10, 79, 69), fill="black")

    result = trim_uniform_margins(image, TrimOptions(edge_content_guard_px=8))

    assert result.applied is False
    assert result.bbox_in_original == (0, 0, 100, 80)
    assert result.image.size == (100, 80)


def test_split_horizontal_chunks_uses_blank_band_before_max_height():
    image = Image.new("RGB", (100, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 45), fill="black")
    draw.rectangle((10, 75, 90, 110), fill="black")

    chunks = split_horizontal_chunks(
        image,
        ChunkOptions(
            min_chunk_height_px=30,
            max_chunk_height_px=70,
            blank_band_min_height_px=10,
        ),
    )

    assert [chunk.bbox_in_trimmed for chunk in chunks] == [
        (0, 0, 100, 60),
        (0, 60, 100, 120),
    ]
    assert [chunk.split_reason for chunk in chunks] == [
        "horizontal_blank_band",
        "end_of_image",
    ]
