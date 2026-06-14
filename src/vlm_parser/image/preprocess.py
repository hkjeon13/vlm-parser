from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops


@dataclass(frozen=True, slots=True)
class TrimOptions:
    background_tolerance: int = 8
    edge_content_guard_px: int = 12
    max_trim_ratio: float = 0.7


@dataclass(frozen=True, slots=True)
class TrimResult:
    image: Image.Image
    bbox_in_original: tuple[int, int, int, int]
    applied: bool
    reason: str


def _corner_has_content(image: Image.Image, options: TrimOptions) -> bool:
    guard = max(1, options.edge_content_guard_px)
    width, height = image.size
    corners = [
        image.crop((0, 0, min(guard, width), min(guard, height))),
        image.crop((max(0, width - guard), 0, width, min(guard, height))),
        image.crop((0, max(0, height - guard), min(guard, width), height)),
        image.crop((max(0, width - guard), max(0, height - guard), width, height)),
    ]
    for corner in corners:
        background = Image.new(image.mode, corner.size, corner.getpixel((0, 0)))
        if ImageChops.difference(corner, background).getbbox() is not None:
            return True
    return False


def trim_uniform_margins(image: Image.Image, options: TrimOptions | None = None) -> TrimResult:
    options = options or TrimOptions()
    rgb = image.convert("RGB")
    width, height = rgb.size

    if _corner_has_content(rgb, options):
        return TrimResult(rgb, (0, 0, width, height), False, "corner_content_detected")

    background_color = rgb.getpixel((0, 0))
    background = Image.new("RGB", rgb.size, background_color)
    diff = ImageChops.difference(rgb, background)
    if options.background_tolerance > 0:
        diff = diff.point(lambda value: 0 if value <= options.background_tolerance else 255)
    bbox = diff.getbbox()

    if bbox is None:
        return TrimResult(rgb, (0, 0, width, height), False, "blank_image")

    x0, y0, x1, y1 = bbox
    trimmed_area = (x1 - x0) * (y1 - y0)
    original_area = width * height
    if original_area and trimmed_area / original_area < (1.0 - options.max_trim_ratio):
        return TrimResult(rgb, (0, 0, width, height), False, "trim_area_too_small")

    if bbox == (0, 0, width, height):
        return TrimResult(rgb, bbox, False, "no_uniform_margin")

    return TrimResult(rgb.crop(bbox), bbox, True, "uniform_margin_detected")
