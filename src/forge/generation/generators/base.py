"""Generator backends behind one interface.

Note the asymmetry with serving. vLLM belongs HERE, on the generation side, where the
workload is autoregressive decoding of hundreds of thousands of documents and
continuous batching genuinely helps. It does not belong on the detector's serving path,
which runs a bidirectional encoder with a fixed window and no KV cache. See
src/forge/inference/server.py.

Every backend must report a concrete `revision`. `require_pinned_revision` refuses to
run against an unpinned model, because "generated with Qwen 7B" is not a reproducible
statement: the upstream repo can move and the same config would then produce different
data under the same dataset version.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

UNPINNED_MARKERS = ("TODO_PIN_AT_FIRST_RUN", "main", "", None)


@dataclass(frozen=True)
class Decoding:
    temperature: float
    top_p: float
    max_new_tokens: int = 1024
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


class UnpinnedRevisionError(RuntimeError):
    pass


def require_pinned_revision(family: str, revision: str | None) -> str:
    if revision in UNPINNED_MARKERS:
        raise UnpinnedRevisionError(
            f"generator family {family!r} has revision {revision!r}. Pin an exact repo "
            "revision in configs/generation/generators.yaml before generating. An "
            "unpinned model means this dataset version cannot be reproduced."
        )
    return revision  # type: ignore[return-value]


@runtime_checkable
class Generator(Protocol):
    family: str
    model_id: str
    revision: str

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]: ...


class VLLMGenerator:
    """Offline batch generation for open-weight families."""

    def __init__(self, family: str, model_id: str, revision: str, tensor_parallel_size: int = 1) -> None:
        self.family = family
        self.model_id = model_id
        self.revision = require_pinned_revision(family, revision)
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    def _load(self):  # pragma: no cover - needs a GPU
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                model=self.model_id,
                revision=self.revision,
                tensor_parallel_size=self.tensor_parallel_size,
            )
        return self._llm

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:  # pragma: no cover
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_tokens=decoding.max_new_tokens,
            seed=decoding.seed,
        )
        outs = self._load().generate(prompts, params)
        return [o.outputs[0].text for o in outs]


class TransformersGenerator:
    """Fallback for machines without vLLM, such as an Apple Silicon laptop.

    Much slower. Useful for generating a few thousand documents to validate the mirror
    prompt before committing GPU hours to the full 400k run.
    """

    def __init__(self, family: str, model_id: str, revision: str, device: str = "auto") -> None:
        self.family = family
        self.model_id = model_id
        self.revision = require_pinned_revision(family, revision)
        self.device = device
        self._pipe = None

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:  # pragma: no cover
        from transformers import pipeline

        if self._pipe is None:
            self._pipe = pipeline(
                "text-generation", model=self.model_id, revision=self.revision, device_map=self.device
            )
        outs = self._pipe(
            prompts,
            do_sample=not decoding.greedy,
            temperature=decoding.temperature or None,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            return_full_text=False,
        )
        return [o[0]["generated_text"] for o in outs]


class APIGenerator:
    """The held-out frontier family. Records the served model string per request, because
    an API model can change underneath a stable name."""

    def __init__(self, family: str, model_id: str, endpoint: str) -> None:
        self.family = family
        self.model_id = model_id
        self.endpoint = endpoint
        self.revision = "recorded_at_run_time"

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:  # pragma: no cover
        raise NotImplementedError("Phase 5: wire the held-out API family")


class FakeGenerator:
    """Deterministic stand-in so the whole mirror pipeline is verifiable without a GPU.

    It produces text of roughly the requested length from the prompt's own attributes.
    It is NOT a model and its output is NOT training data. The runner records
    provider="fake" on every record so a fake-generated dataset can never be mistaken
    for a real one downstream.
    """

    family = "fake"
    model_id = "forge/fake-generator"
    revision = "v1"

    _FILLER = (
        "The account sets out the background before turning to the details that follow. "
        "Several considerations bear on the question, and they are taken in turn below. "
        "Evidence from the period is uneven, which limits how firmly any conclusion can "
        "be drawn. A number of practitioners disagreed with the prevailing approach. "
        "Later commentary revisited the matter without reaching a settled view. "
        "The arrangement persisted for some time before circumstances changed again. "
    )

    def __init__(self, family: str = "fake", model_id: str | None = None, revision: str = "v1") -> None:
        self.family = family
        self.model_id = model_id or f"forge/fake-{family}"
        self.revision = revision

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:
        out = []
        for p in prompts:
            target = _target_from_prompt(p)
            # jitter deterministically off the prompt hash so lengths vary like a real model
            h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            n = max(int(target * (0.85 + (h % 30) / 100.0)), 20)
            words = (self._FILLER * (n // 50 + 2)).split()[:n]
            # Emit paragraph breaks. A stand-in that returns one unbroken block cannot
            # exercise anything downstream that depends on document structure, and
            # splice construction would silently yield zero documents. Real generators
            # paragraph their output, and one that does not is a bug worth surfacing.
            per_para = max(n // (3 + (h % 3)), 25)
            paras = [" ".join(words[i : i + per_para]) for i in range(0, len(words), per_para)]
            out.append("\n\n".join(paras))
        return out


_TARGET_RE = None


def _target_from_prompt(prompt: str, default: int = 300) -> int:
    import re as _re

    m = _re.search(r"approximately (\d+) tokens", prompt)
    return int(m.group(1)) if m else default
