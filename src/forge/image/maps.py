"""Forensic residual maps: where in the frame each signal is strong.

**This is not model saliency and must never be presented as such.** A saliency map answers
"which regions does the detector think are generated". These maps answer "where in this
frame is the compression residual, the noise floor or the local contrast unusual relative to
the rest of this same frame". No model is involved, nothing is trained, and the maps say
nothing about authorship on their own.

Why they are worth showing anyway: they are inspectable. A reader can look at a bright
region and go and see what is there. A model heatmap asks for trust; this asks for a look.

Three maps, matching three findings in `forensics.py` so the panel and the table cannot
drift apart:

  error_level    per-pixel residual after re-encoding at a fixed quality
  noise          per-cell dispersion of a Laplacian high-pass, the noise floor
  detail         per-cell local contrast, which is what smoothing and inpainting flatten

**Every map is normalized within itself.** The brightest pixel is the strongest signal in
THIS image, not a fixed scale. So a map of a perfectly ordinary photograph still has bright
regions, and brightness across two different images is not comparable. That is stated in the
returned payload and rendered on the panel, because a heatmap invites exactly the
interpretation it cannot support.

Maps are returned as base64 PNG data URIs sized to fit the panel, so the page stays one file
with no extra requests.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

MAX_EDGE = 512          # maps are for looking at, not for measuring
MIN_EDGE = 32
CELL_DIVISIONS = 48     # cells across the long edge for the cell-based maps


@dataclass
class Map:
    name: str
    title: str
    png_data_uri: str
    what_it_shows: str
    caveat: str
    peak_location: dict | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "title": self.title, "image": self.png_data_uri,
            "what_it_shows": self.what_it_shows, "caveat": self.caveat,
            "peak_location": self.peak_location, "detail": self.detail,
        }


def _to_png_uri(array) -> str:
    """Normalize within the image, colourize, return a PNG data URI."""
    import numpy as np
    from PIL import Image

    a = np.asarray(array, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)

    # Blue for low, through amber, to red at the peak. Readable in both page themes and
    # distinguishable without relying on hue alone, because brightness rises monotonically.
    r = np.clip(1.6 * a - 0.35, 0, 1)
    g = np.clip(1.5 * a - 0.15, 0, 1) * (1 - 0.55 * a)
    b = np.clip(0.85 - 1.5 * a, 0, 1) + 0.18 * a
    rgb = (np.dstack([r, g, np.clip(b, 0, 1)]) * 255).astype("uint8")

    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _peak(array) -> dict:
    """Where the strongest signal sits, in fractions of width and height.

    Fractions rather than pixels so the caller can point at the spot on a resized preview
    without knowing the working size.
    """
    import numpy as np

    a = np.asarray(array)
    y, x = np.unravel_index(int(np.argmax(a)), a.shape)
    return {"x": round(float(x) / max(1, a.shape[1] - 1), 4),
            "y": round(float(y) / max(1, a.shape[0] - 1), 4)}


def _cell_grid(plane, divisions: int, statistic):
    """Reduce a plane to a grid of per-cell statistics. Returns the grid, not a resize."""
    import numpy as np

    h, w = plane.shape
    long_edge = max(h, w)
    cell = max(4, long_edge // max(1, divisions))
    rows = max(1, h // cell)
    cols = max(1, w // cell)
    grid = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            block = plane[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell]
            grid[r, c] = statistic(block) if block.size else 0.0
    return grid


def _load(data: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")
        if max(rgb.size) > MAX_EDGE:
            scale = MAX_EDGE / max(rgb.size)
            rgb = rgb.resize(
                (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))),
                Image.Resampling.LANCZOS,
            )
        rgb.load()
        return rgb


def error_level_map(data: bytes, quality: int = 90) -> Map | None:
    import numpy as np
    from PIL import Image

    rgb = _load(data)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    with Image.open(io.BytesIO(buffer.getvalue())) as again:
        residual = np.abs(
            np.asarray(rgb, dtype=np.int16) - np.asarray(again.convert("RGB"), dtype=np.int16)
        ).mean(axis=2)

    return Map(
        name="error_level",
        title="Compression residual",
        png_data_uri=_to_png_uri(residual),
        what_it_shows=(
            "How much each region changes when the image is re-encoded at quality "
            f"{quality}. Edges and texture always light up; a bright PATCH with soft "
            "boundaries in otherwise flat area is the thing worth looking at."
        ),
        caveat=(
            "responds to any re-encode, including innocent ones, and is normalized within "
            "this image, so brightness here is not comparable to any other image"
        ),
        peak_location=_peak(residual),
        detail={"quality_probe": quality, "mean_residual": round(float(residual.mean()), 3)},
    )


def noise_grid(data: bytes):
    """Per-cell noise floor. Separated from the picture so it can be asserted on."""
    import numpy as np

    grey = np.asarray(_load(data).convert("L"), dtype=np.float64)
    if min(grey.shape) < MIN_EDGE:
        return None
    hp = (
        grey[1:-1, 1:-1] * 4
        - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
    )
    return _cell_grid(np.abs(hp), CELL_DIVISIONS, lambda b: float(b.std()))


def detail_grid(data: bytes):
    """Per-cell local contrast. Separated from the picture so it can be asserted on."""
    import numpy as np

    grey = np.asarray(_load(data).convert("L"), dtype=np.float64)
    if min(grey.shape) < MIN_EDGE:
        return None
    return _cell_grid(grey, CELL_DIVISIONS, lambda b: float(b.max() - b.min()))


def noise_map(data: bytes) -> Map | None:
    grid = noise_grid(data)
    if grid is None:
        return None
    return Map(
        name="noise",
        title="Noise floor",
        png_data_uri=_to_png_uri(grid),
        what_it_shows=(
            "Dispersion of a high-pass residual per cell. A photograph carries roughly "
            "uniform sensor noise, so large flat DARK regions mean the noise floor was "
            "removed there, by denoising, by inpainting, or because the region was "
            "generated."
        ),
        caveat=(
            "smooth skies, shallow depth of field and ordinary denoising all flatten noise "
            "in genuine photographs; this describes the frame, it does not accuse it"
        ),
        peak_location=_peak(grid),
        detail={"cells": int(grid.size)},
    )


def detail_map(data: bytes) -> Map | None:
    grid = detail_grid(data)
    if grid is None:
        return None
    return Map(
        name="detail",
        title="Local contrast",
        png_data_uri=_to_png_uri(grid),
        what_it_shows=(
            "Range between the lightest and darkest pixel in each cell. Generated regions "
            "and heavy retouching tend to be locally smoother than their surroundings."
        ),
        caveat=(
            "sky, water, walls and bokeh are legitimately smooth; low contrast is a "
            "property of the subject at least as often as of the pipeline"
        ),
        peak_location=_peak(grid),
        detail={"cells": int(grid.size)},
    )


def build_maps(data: bytes) -> list[dict]:
    """All three maps. A map that cannot be computed is omitted, never faked.

    Never raises: the panel is one part of a report, and a report that dies because a
    residual could not be computed on a 20-pixel image is worse than one missing a picture.
    """
    out: list[dict] = []
    for builder in (error_level_map, noise_map, detail_map):
        try:
            built = builder(data)
        except Exception:  # noqa: BLE001 - a missing map is a missing map
            built = None
        if built is not None:
            out.append(built.as_dict())
    return out
