from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops


@dataclass(frozen=True, slots=True)
class ChunkOptions:
    min_chunk_height_px: int = 400
    max_chunk_height_px: int = 1200
    blank_band_min_height_px: int = 24
    background_tolerance: int = 8


@dataclass(frozen=True, slots=True)
class ChunkBox:
    index: int
    bbox_in_trimmed: tuple[int, int, int, int]
    split_reason: str


def _row_is_background(image: Image.Image, y: int, tolerance: int) -> bool:
    row = image.crop((0, y, image.width, y + 1)).convert("RGB")
    background = Image.new("RGB", row.size, row.getpixel((0, 0)))
    diff = ImageChops.difference(row, background)
    if tolerance > 0:
        diff = diff.point(lambda value: 0 if value <= tolerance else 255)
    return diff.getbbox() is None


def _blank_bands(image: Image.Image, options: ChunkOptions) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(image.height):
        if _row_is_background(image, y, options.background_tolerance):
            if start is None:
                start = y
        elif start is not None:
            if y - start >= options.blank_band_min_height_px:
                bands.append((start, y))
            start = None
    if start is not None and image.height - start >= options.blank_band_min_height_px:
        bands.append((start, image.height))
    return bands


def split_horizontal_chunks(image: Image.Image, options: ChunkOptions | None = None) -> list[ChunkBox]:
    options = options or ChunkOptions()
    width, height = image.size
    if height <= options.max_chunk_height_px:
        return [ChunkBox(0, (0, 0, width, height), "end_of_image")]

    bands = _blank_bands(image, options)
    chunks: list[ChunkBox] = []
    start_y = 0

    while height - start_y > options.max_chunk_height_px:
        target_max = start_y + options.max_chunk_height_px
        candidates = [
            (band_start, band_end)
            for band_start, band_end in bands
            if start_y + options.min_chunk_height_px <= band_start <= target_max
        ]
        if not candidates:
            break
        band_start, band_end = min(candidates, key=lambda band: abs(((band[0] + band[1]) // 2) - target_max))
        split_y = (band_start + band_end) // 2
        chunks.append(ChunkBox(len(chunks), (0, start_y, width, split_y), "horizontal_blank_band"))
        start_y = split_y

    chunks.append(ChunkBox(len(chunks), (0, start_y, width, height), "end_of_image"))
    return chunks
