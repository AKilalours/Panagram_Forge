"""The forensic maps must respond to what their captions claim they respond to.

A heatmap is trusted on sight. Testing that it renders is not enough: it has to be
demonstrably brighter or darker in the region a reader is told to look at. These tests build
frames with a known planted region and assert the map notices it.
"""

from __future__ import annotations

import io

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from forge.image.maps import (  # noqa: E402
    _to_png_uri, build_maps, detail_grid, noise_grid,
)

PATCH = (slice(80, 176), slice(80, 176))     # the planted region, in pixels


def _frame(flatten_patch: bool, size: int = 256, seed: int = 0) -> bytes:
    """Uniform noise, optionally with one flat square pasted in."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(128, 30, (size, size, 3)).clip(0, 255).astype("uint8")
    if flatten_patch:
        arr[PATCH] = 128
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")   # lossless: isolate the planted effect
    return buffer.getvalue()


def _patch_cells(grid):
    """The grid cells covering the planted patch, and everything else."""
    rows, cols = grid.shape
    r0, r1 = int(rows * 80 / 256), int(np.ceil(rows * 176 / 256))
    c0, c1 = int(cols * 80 / 256), int(np.ceil(cols * 176 / 256))
    mask = np.zeros(grid.shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    # Shrink by one cell so boundary cells, which straddle both regions, are excluded.
    inner = np.zeros_like(mask)
    inner[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = True
    return grid[inner], grid[~mask]


def test_the_noise_map_darkens_where_the_noise_floor_was_removed():
    """THE CLAIM ON THE PANEL: flat dark regions mean the noise floor is gone there."""
    grid = noise_grid(_frame(flatten_patch=True))
    planted, rest = _patch_cells(grid)
    assert planted.size and rest.size
    assert planted.max() < rest.min(), (
        f"planted flat region max {planted.max():.2f} should sit below the rest's "
        f"min {rest.min():.2f}"
    )


def test_the_noise_map_is_flat_when_nothing_was_planted():
    """The accepting side. Without a patch the map must not invent a hot region."""
    grid = noise_grid(_frame(flatten_patch=False))
    assert float(grid.std() / grid.mean()) < 0.25


def test_the_detail_map_darkens_in_a_locally_smooth_region():
    grid = detail_grid(_frame(flatten_patch=True))
    planted, rest = _patch_cells(grid)
    assert planted.max() < rest.min()


def test_normalization_is_within_the_image_so_a_flat_frame_is_not_all_hot():
    """A constant plane has no strongest region, and must not render as a peak everywhere."""
    uri = _to_png_uri(np.zeros((16, 16)))
    assert uri.startswith("data:image/png;base64,")
    flat = _to_png_uri(np.full((16, 16), 7.0))
    assert flat == uri, "a constant plane must render identically regardless of its value"


def test_a_map_is_omitted_rather_than_faked_when_it_cannot_be_computed():
    assert build_maps(b"not an image at all") == []
    tiny = io.BytesIO()
    Image.new("RGB", (8, 8)).save(tiny, format="PNG")
    names = {m["name"] for m in build_maps(tiny.getvalue())}
    assert "noise" not in names and "detail" not in names, (
        "an 8-pixel image cannot support a cell grid; those maps must be absent"
    )


def test_every_map_carries_its_caveat_and_a_peak():
    maps = build_maps(_frame(flatten_patch=True))
    assert maps
    for m in maps:
        assert m["caveat"], f"{m['name']} has no caveat"
        assert m["what_it_shows"]
        assert set(m["peak_location"]) == {"x", "y"}
        assert m["image"].startswith("data:image/png;base64,")


def test_the_maps_are_not_described_as_model_output():
    """Guards the wording. These are residual maps, not detector saliency."""
    for m in build_maps(_frame(flatten_patch=True)):
        blob = (m["title"] + m["what_it_shows"] + m["caveat"]).lower()
        for forbidden in ("saliency", "the detector thinks", "model believes", "confidence"):
            assert forbidden not in blob, f"{m['name']} claims model output: {forbidden}"
