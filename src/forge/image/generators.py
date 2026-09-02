"""Image generator backends behind one interface.

Mirrors forge/generation/generators/base.py, deliberately: same pinned-revision rule, same
fake backend for offline verification, same refusal to run against an unpinned model.

THE PINNING RULE, restated because it costs nothing until the day it costs everything.
"Generated with SDXL" is not a reproducible statement. A HuggingFace repo can move under a
stable name, and the same config would then produce a different dataset under the same
dataset version. Every generator here reports a concrete revision or refuses to run.

DIFFUSION-SPECIFIC DETERMINISM. A text generator is pinned by model, revision and decoding
parameters. A diffusion model needs more: scheduler, step count, guidance scale, resolution
and seed all change the output, and a run that records only the model id cannot be repeated.
All of them live in GenerationSpec and all of them are written to every record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

UNPINNED_MARKERS = ("TODO_PIN_AT_FIRST_RUN", "main", "", None)


class UnpinnedRevisionError(RuntimeError):
    pass


def require_pinned_revision(family: str, revision: str | None) -> str:
    if revision in UNPINNED_MARKERS:
        raise UnpinnedRevisionError(
            f"image generator family {family!r} has revision {revision!r}. Pin an exact "
            "repo revision before generating. An unpinned model means this dataset "
            "version cannot be reproduced."
        )
    return revision  # type: ignore[return-value]


@dataclass(frozen=True)
class GenerationSpec:
    """Everything that changes the output. Anything missing here is unreproducible."""

    steps: int = 30
    guidance: float = 5.0
    resolution: int = 1024
    scheduler: str = "default"
    seed: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class ImageGenerator(Protocol):
    family: str
    model_id: str
    revision: str

    def generate(self, prompts: list[str], spec: GenerationSpec) -> list[bytes]: ...

    def close(self) -> None: ...


class DiffusersGenerator:
    """Real backend. One model in memory at a time, released before the next.

    The text track learned this the expensive way: it cached one engine per generator
    family, every family's weights were resident at once, and the second model found a
    gigabyte free on a 24GB card. Diffusion pipelines are just as large, so the same
    discipline applies from the start rather than after the first failed run.
    """

    def __init__(self, family: str, model_id: str, revision: str, dtype: str = "float16") -> None:
        self.family = family
        self.model_id = model_id
        self.revision = require_pinned_revision(family, revision)
        self.dtype = dtype
        self._pipe = None

    def _load(self):  # pragma: no cover - needs a GPU
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            from forge.generation.generators.base import require_free_gpu

            require_free_gpu()
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=getattr(torch, self.dtype),
            ).to("cuda")
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def generate(self, prompts: list[str], spec: GenerationSpec) -> list[bytes]:  # pragma: no cover
        import io

        import torch

        pipe = self._load()
        out: list[bytes] = []
        for index, prompt in enumerate(prompts):
            # A per-prompt seed, derived from the batch seed, so a single image can be
            # regenerated on its own without replaying the whole batch.
            generator = None
            if spec.seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(spec.seed + index)
            image = pipe(
                prompt,
                num_inference_steps=spec.steps,
                guidance_scale=spec.guidance,
                height=spec.resolution,
                width=spec.resolution,
                generator=generator,
            ).images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out

    def close(self) -> None:  # pragma: no cover - needs a GPU
        if self._pipe is None:
            return
        self._pipe = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


class FakeImageGenerator:
    """Deterministic stand-in so the mirror pipeline runs without a GPU.

    Output depends only on the prompt, so it is reproducible, and it is unrelated to any
    source image, which is what lets the pipeline's "no pixels crossed" invariant actually
    be tested. It is NOT a model and its output is NOT training data; the mirror engine
    records the family on every record so fake-derived data cannot be mistaken for real.
    """

    family = "fake"
    model_id = "forge/fake-image-generator"
    revision = "v1"

    def generate(self, prompts: list[str], spec: GenerationSpec) -> list[bytes]:
        import hashlib
        import io

        from PIL import Image

        size = min(spec.resolution, 512)
        out: list[bytes] = []
        for index, prompt in enumerate(prompts):
            seed = int.from_bytes(
                hashlib.sha256(f"{prompt}:{spec.seed}:{index}".encode()).digest()[:4], "big"
            )
            period = 3 + (seed % 11)
            data = bytearray()
            for y in range(size):
                for x in range(size):
                    data += bytes(
                        (
                            ((x // period) * 53 + seed) % 256,
                            ((y // (period + 2)) * 31 + seed // 3) % 256,
                            ((x + y) * (seed % 7 + 1)) % 256,
                        )
                    )
            buf = io.BytesIO()
            Image.frombytes("RGB", (size, size), bytes(data)).save(buf, format="PNG")
            out.append(buf.getvalue())
        return out

    def close(self) -> None:
        return None
