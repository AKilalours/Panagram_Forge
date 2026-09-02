"""Building an engine must refuse a GPU that is already occupied.

THE INCIDENT THIS CAME FROM. A three-hour generation run was launched while an
earlier probe still held the card. Both arms died with

    ValueError: Free memory on device (0.56/23.53 GiB) on startup is less than
    desired GPU memory utilization (0.9, 21.17 GiB)

several minutes in, after the corpus had been read and every prompt rendered.
The message names the symptom, not the cause, and the two causes worth telling
apart are very different: a stale job the operator forgot to kill, or close()
failing to release the previous family's engine, which would be a bug in
VLLMGenerator itself.
"""

from __future__ import annotations

import pytest

from forge.generation.generators import base as base_mod
from forge.generation.generators.base import GpuNotFreeError, require_free_gpu

GIB = 1024**3


def _fake_cuda(monkeypatch: pytest.MonkeyPatch, free_gib: float, total_gib: float = 23.5):
    """Stand in for torch.cuda without needing a GPU."""

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info():
            return int(free_gib * GIB), int(total_gib * GIB)

    import sys
    import types

    torch = types.ModuleType("torch")
    torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_passes_when_the_card_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch, free_gib=23.0)
    require_free_gpu()


def test_refuses_when_a_previous_job_still_holds_the_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression, using the exact numbers from the incident."""
    _fake_cuda(monkeypatch, free_gib=0.56)
    with pytest.raises(GpuNotFreeError) as e:
        require_free_gpu()
    assert "0.56" in str(e.value)


def test_the_message_names_both_plausible_causes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error that does not distinguish operator error from a leak wastes a debug cycle."""
    _fake_cuda(monkeypatch, free_gib=1.0)
    with pytest.raises(GpuNotFreeError) as e:
        require_free_gpu()
    message = str(e.value)
    assert "nvidia-smi" in message
    assert "close()" in message


def test_threshold_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch, free_gib=5.0)
    require_free_gpu(min_free_gib=4.0)
    with pytest.raises(GpuNotFreeError):
        require_free_gpu(min_free_gib=6.0)


def test_absent_gpu_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A laptop with no CUDA must not be blocked from importing or from fake runs."""
    import sys
    import types

    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    require_free_gpu()


def test_a_torch_that_cannot_report_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a diagnostic block a run it was only meant to explain."""
    import sys
    import types

    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info():
            raise RuntimeError("driver says no")

    torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    require_free_gpu()


def test_the_loader_calls_the_guard() -> None:
    """The check must sit on the path that actually builds an engine."""
    import inspect

    source = inspect.getsource(base_mod.VLLMGenerator._load)
    assert "require_free_gpu()" in source
    assert source.index("require_free_gpu()") < source.index("LLM(")
