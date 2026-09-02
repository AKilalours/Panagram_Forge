"""Gradient checkpointing must be non-reentrant, or DeBERTa-v2 dies on its first backward.

THE BUG THIS CAME FROM. `make min-smoke` downloaded the backbone, ran one forward, and died:

    RuntimeError: Trying to backward through the graph a second time (or directly access
    saved tensors after they have already been freed)

raised from inside torch/utils/checkpoint.py's own backward, which is what points at
checkpointing rather than at the training loop.

WHY. `gradient_checkpointing_enable()` defaults to REENTRANT checkpointing, which recomputes
each block during backward and assumes the block is a closed function of its inputs.
DeBERTa-v2's disentangled attention needs relative position embeddings, and the encoder
computes those ONCE, outside the per-layer blocks, then hands the same tensor to every
layer. The first layer's backward frees that shared tensor's graph and the next layer's
recomputation walks it again.

The important test here is `test_reentrant_checkpointing_actually_fails`. Without it, the
non-reentrant assertion proves nothing: if reentrant checkpointing happened to work on this
model, the fix would be cargo cult and the real cause would still be unknown. Asserting that
the old setting genuinely breaks is what makes the new setting an explanation rather than a
guess. This is the same discipline as testing the accepting side of a rejection rule.
"""

from __future__ import annotations

import importlib.util

import pytest

from forge.training.train import enable_gradient_checkpointing


class _Spy:
    """Records how checkpointing was requested."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.calls.append(gradient_checkpointing_kwargs or {})


class _Old:
    """A transformers old enough to have no kwargs passthrough."""

    def gradient_checkpointing_enable(self) -> None:  # noqa: D401
        pass


class _Unsupported:
    """An encoder with no checkpointing at all."""


def test_checkpointing_is_requested_non_reentrant() -> None:
    """The regression, stated directly."""
    spy = _Spy()
    assert enable_gradient_checkpointing(spy) is True
    assert spy.calls == [{"use_reentrant": False}], spy.calls


def test_an_encoder_without_checkpointing_is_not_an_error() -> None:
    """A missing memory optimisation is not a reason to refuse to train."""
    assert enable_gradient_checkpointing(_Unsupported()) is False


def test_a_transformers_too_old_to_choose_refuses_loudly() -> None:
    """Silently falling back to reentrant would reintroduce the crash later and elsewhere."""
    with pytest.raises(RuntimeError, match="non-reentrant"):
        enable_gradient_checkpointing(_Old())


# --------------------------------------------------------------- the part that needs torch

# NOT importorskip at module level: that skips the ENTIRE file, including the three tests
# above, which need neither torch nor transformers. A file that reports "skipped" where it
# should report "passed" is the same lie as a check that fires on nothing.
try:
    import torch
except ImportError:
    torch = None

requires_backend = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="the reproduction needs a real backward pass; the spy tests above do not",
)


def _deberta(reentrant_safe: bool = False, **overrides):
    """A DeBERTa-v2 small enough to run in a test, carrying the fields that MATTER.

    Every field below was verified to be load-bearing. Drop `pos_att_type` and the relative
    embeddings never enter the graph at all; drop `norm_rel_ebd="layer_norm"` and they enter
    as a leaf parameter. In either case reentrant checkpointing runs cleanly, the
    reproduction test passes vacuously, and the file proves nothing. deberta-v3-base sets
    both, which is why the crash appeared on the real backbone and on nothing smaller.

    Three layers, not two: the failure needs one layer's backward to free the shared graph
    before another layer's recomputation reaches it.
    """
    from transformers import DebertaV2Config, DebertaV2Model

    config = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        # The two that make the difference. See the module docstring.
        pos_att_type=["p2c", "c2p"],
        norm_rel_ebd="none" if reentrant_safe else "layer_norm",
        share_att_key=True,
        position_biased_input=False,
        layer_norm_eps=1e-7,
    )
    config.update(overrides)
    return DebertaV2Model(DebertaV2Config(**config))


def _one_backward(encoder) -> None:
    ids = torch.randint(0, 64, (2, 16))
    mask = torch.ones_like(ids)
    encoder(input_ids=ids, attention_mask=mask).last_hidden_state.sum().backward()


@requires_backend
def test_a_backward_pass_survives_with_the_fix() -> None:
    """The reproduction, fixed. This is the assertion make min-smoke was making the hard way."""
    encoder = _deberta()
    assert enable_gradient_checkpointing(encoder) is True
    encoder.train()
    _one_backward(encoder)          # must not raise


@requires_backend
def test_reentrant_checkpointing_actually_fails() -> None:
    """THE TEST THAT MAKES THE FIX AN EXPLANATION.

    If this passes, reentrant checkpointing works on this model, the crash had another
    cause, and use_reentrant=False is a coincidence rather than a fix. Failing here is the
    evidence.
    """
    encoder = _deberta()
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    encoder.train()
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        _one_backward(encoder)


@requires_backend
def test_gradients_actually_reach_the_first_layer() -> None:
    """Checkpointing that runs but produces no gradients would train nothing, quietly.

    A checkpointed block whose inputs do not require grad silently returns no gradient. The
    loss would still go down, driven by the heads alone, and the encoder would never move.
    """
    encoder = _deberta()
    enable_gradient_checkpointing(encoder)
    encoder.train()
    _one_backward(encoder)
    first = encoder.encoder.layer[0].attention.self.query_proj.weight
    assert first.grad is not None, "the first layer received no gradient"
    assert torch.isfinite(first.grad).all()
    assert first.grad.abs().sum().item() > 0.0, "the first layer's gradient is all zeros"


@requires_backend
def test_the_layer_normed_relative_embedding_is_what_breaks_it() -> None:
    """Isolates the mechanism to a single config field, so the diagnosis is not a story.

    DeBERTa-v2 computes the relative position embeddings ONCE in the encoder and hands the
    same tensor to every layer. With norm_rel_ebd="layer_norm" that tensor is the output of
    a LayerNorm, so it carries an autograd graph that lives OUTSIDE the checkpointed blocks:
    the first layer's backward frees it, the next layer's recomputation walks it again.

    With norm_rel_ebd="none" the same tensor is a leaf parameter with no graph behind it,
    and reentrant checkpointing is fine. That is the whole difference, and deberta-v3-base
    sets "layer_norm".
    """
    encoder = _deberta(reentrant_safe=True)
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    encoder.train()
    _one_backward(encoder)          # must not raise: no shared graph to walk twice
