"""Image generators carry the same reproducibility rules as the text ones, plus more.

A text generation is pinned by model, revision and decoding parameters. A diffusion output
also depends on scheduler, step count, guidance scale, resolution and seed. A record that
names only the model cannot be regenerated, so all of them are part of the spec.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PIL", reason="pillow is in the [image] extra")

from forge.image.generators import (  # noqa: E402
    DiffusersGenerator,
    FakeImageGenerator,
    GenerationSpec,
    UnpinnedRevisionError,
    require_pinned_revision,
)

SHA = "a" * 40
SPEC = GenerationSpec(steps=4, guidance=3.0, resolution=256, seed=11)


@pytest.mark.parametrize("bad", ["TODO_PIN_AT_FIRST_RUN", "main", "", None])
def test_unpinned_revisions_are_refused(bad) -> None:
    with pytest.raises(UnpinnedRevisionError):
        require_pinned_revision("sdxl", bad)


def test_a_pinned_revision_is_accepted() -> None:
    assert require_pinned_revision("sdxl", SHA) == SHA


def test_the_real_backend_refuses_an_unpinned_model_before_loading_anything() -> None:
    """Constructing must fail, not the first generate call after a 7GB download."""
    with pytest.raises(UnpinnedRevisionError):
        DiffusersGenerator("sdxl", "stabilityai/stable-diffusion-xl-base-1.0", "main")


def test_spec_records_every_field_that_changes_the_output() -> None:
    got = SPEC.as_dict()
    assert set(got) == {"steps", "guidance", "resolution", "scheduler", "seed"}


def test_fake_generator_is_deterministic() -> None:
    gen = FakeImageGenerator()
    assert gen.generate(["a prompt"], SPEC) == gen.generate(["a prompt"], SPEC)


def test_different_prompts_give_different_images() -> None:
    """Otherwise the pipeline would appear to work while producing one image forever."""
    gen = FakeImageGenerator()
    a, b = gen.generate(["prompt one", "prompt two"], SPEC)
    assert a != b


def test_the_seed_changes_the_output() -> None:
    gen = FakeImageGenerator()
    a = gen.generate(["p"], GenerationSpec(resolution=256, seed=1))
    b = gen.generate(["p"], GenerationSpec(resolution=256, seed=2))
    assert a != b


def test_batch_length_matches_prompt_length() -> None:
    gen = FakeImageGenerator()
    assert len(gen.generate([f"p{i}" for i in range(5)], SPEC)) == 5


def test_images_within_one_batch_differ_from_each_other() -> None:
    """A per-index seed offset, so one image can be regenerated without the whole batch."""
    gen = FakeImageGenerator()
    out = gen.generate(["same prompt"] * 3, SPEC)
    assert len(set(out)) == 3


def test_fake_generator_marks_itself() -> None:
    assert FakeImageGenerator().family == "fake"


def test_close_is_safe_to_call_twice() -> None:
    gen = FakeImageGenerator()
    gen.close()
    gen.close()
