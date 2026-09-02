"""vLLM must not size its KV cache from a model's advertised context length.

THE BUG THIS CAME FROM. Generation crashed on the second generator family with

    ValueError: To serve at least one request with the model's max seq len (131072),
    48.01 GiB KV cache is needed, which is larger than the available KV cache memory
    (13.35 GiB)

Phi-3.5-mini advertises a 131072-token context. vLLM reserves KV cache for the full
advertised context unless max_model_len is passed, so a model FORGE never asks for more
than ~1200 tokens from demanded 48 GiB on a 24 GB card.

The first family, Qwen2.5-3B, has a 32k context and started fine, which is why this only
appeared after a model download and an engine launch, several minutes into a paid GPU
run rather than at import time.

These tests use a fake vllm module. They assert the argument is passed, which is the part
that was missing, not that the engine works, which needs a GPU.
"""

from __future__ import annotations

import sys
import types

import pytest

from forge.generation.generators.base import (
    DEFAULT_MAX_MODEL_LEN,
    ContextTooSmallError,
    Decoding,
    VLLMGenerator,
)

SHA = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Install a stand-in vllm module and record the kwargs LLM() is constructed with."""
    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def generate(self, prompts: list[str], params: object) -> list:
            raise AssertionError("these tests must not reach generation")

    module = types.ModuleType("vllm")
    module.LLM = FakeLLM  # type: ignore[attr-defined]
    module.SamplingParams = lambda **kw: kw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    return captured


def test_default_context_is_bounded_and_small() -> None:
    """A default in the tens of thousands would reintroduce the crash on a 24 GB card."""
    assert 1024 <= DEFAULT_MAX_MODEL_LEN <= 8192


def test_generator_carries_the_bounded_default() -> None:
    gen = VLLMGenerator("phi", "microsoft/Phi-3.5-mini-instruct", SHA)
    assert gen.max_model_len == DEFAULT_MAX_MODEL_LEN


def test_max_model_len_is_passed_to_the_engine(fake_vllm: dict) -> None:
    """The regression. Omitting this kwarg is the entire bug."""
    gen = VLLMGenerator("phi", "microsoft/Phi-3.5-mini-instruct", SHA)
    gen._load()
    assert "max_model_len" in fake_vllm, (
        "vLLM was constructed without max_model_len, so it will reserve KV cache for the "
        "model's advertised context and refuse to start on a 24 GB GPU"
    )
    assert fake_vllm["max_model_len"] == DEFAULT_MAX_MODEL_LEN


def test_an_explicit_override_reaches_the_engine(fake_vllm: dict) -> None:
    gen = VLLMGenerator("qwen", "Qwen/Qwen2.5-3B-Instruct", SHA, max_model_len=8192)
    gen._load()
    assert fake_vllm["max_model_len"] == 8192


def test_revision_is_still_pinned_alongside_the_new_kwarg(fake_vllm: dict) -> None:
    """Guard against the new argument displacing the reproducibility one."""
    gen = VLLMGenerator("qwen", "Qwen/Qwen2.5-3B-Instruct", SHA)
    gen._load()
    assert fake_vllm["revision"] == SHA
    assert fake_vllm["model"] == "Qwen/Qwen2.5-3B-Instruct"


def test_the_engine_is_built_once_and_cached(fake_vllm: dict) -> None:
    gen = VLLMGenerator("qwen", "Qwen/Qwen2.5-3B-Instruct", SHA)
    first = gen._load()
    assert gen._load() is first


def test_output_longer_than_the_context_is_refused_up_front() -> None:
    """Truncation would surface two layers away as length-ratio rejections.

    If max_new_tokens exceeds the reserved context, every generation is silently cut
    short and the mirror validator rejects them all for the wrong stated reason. Fail
    here, where the message names the actual cause.
    """
    gen = VLLMGenerator("qwen", "Qwen/Qwen2.5-3B-Instruct", SHA, max_model_len=512)
    with pytest.raises(ContextTooSmallError) as e:
        gen.generate(["prompt"], Decoding(temperature=0.7, top_p=0.9, max_new_tokens=640))
    assert "max_new_tokens" in str(e.value) and "max_model_len" in str(e.value)


def test_the_minimal_config_decoding_fits_the_default_context() -> None:
    """The roster the paid run uses must fit, checked against the real config."""
    from forge.common.config import load

    cfg = load("configs/generation/generators_minimal.yaml")
    max_new = cfg["decoding_grid"]["max_new_tokens"]
    assert max_new < DEFAULT_MAX_MODEL_LEN, (
        f"decoding_grid.max_new_tokens={max_new} does not fit in "
        f"DEFAULT_MAX_MODEL_LEN={DEFAULT_MAX_MODEL_LEN}"
    )
