"""Checkpoint selection must actually select.

The bug this file exists for: selection compared `val.fpr_at_budget` against the best so
far. That number is measured at a threshold chosen to hit the budget, so it is pinned to
the nearest achievable value and never moves. Every epoch of both Tier 1 arms reported
0.000502 (one false positive in 1,993 human documents), the `<=` comparison accepted every
tie, and `best.pt` came out byte-identical to `last.pt` in both runs.

The first test below fails against that old rule and passes against the new one. It is the
test that was missing.
"""

from __future__ import annotations

from forge.training.train import TrainState, ValResult, is_better_checkpoint

BUDGET = 0.001
PINNED = 1 / 1993  # 0.000502, the only value the Tier 1 runs ever reported


def val(fnr: float, fpr: float = PINNED) -> ValResult:
    return ValResult(loss=0.0, fpr_at_budget=fpr, threshold=0.99, fnr=fnr, auroc=0.99,
                     ece=0.01, temperature=1.0, n_human=1993, n_ai=2086)


def run(sequence: list[ValResult]) -> int:
    """Replay a run's evaluations, return the step the selector would keep."""
    state = TrainState()
    for step, v in enumerate(sequence, start=1):
        if is_better_checkpoint(v, state, BUDGET):
            state.best_metric = v.fpr_at_budget
            state.best_fnr = v.fnr
            state.best_in_budget = v.fpr_at_budget <= BUDGET
            state.best_step = step
    return state.best_step


def test_a_flat_fpr_does_not_hand_the_win_to_the_last_epoch():
    """THE REGRESSION TEST. FPR is constant, FNR gets worse; keep the good checkpoint.

    Under the old rule every one of these ties and the last step wins, which is what
    happened in both real runs. Under the new rule the second step wins on FNR.
    """
    assert run([val(fnr=0.020), val(fnr=0.004), val(fnr=0.031)]) == 2


def test_the_pinned_value_really_is_constant_across_the_sequence():
    """Guards the premise: if FPR ever varied, the old rule would not have been silent."""
    seq = [val(fnr=0.020), val(fnr=0.004), val(fnr=0.031)]
    assert len({v.fpr_at_budget for v in seq}) == 1


def test_an_improving_run_keeps_improving():
    assert run([val(fnr=0.09), val(fnr=0.05), val(fnr=0.01)]) == 3


def test_a_tie_on_fnr_keeps_the_earlier_checkpoint():
    """Strict comparison. A tie must not drift to the last epoch, which was the old failure."""
    assert run([val(fnr=0.01), val(fnr=0.01), val(fnr=0.01)]) == 1


def test_inside_the_budget_beats_a_better_fnr_outside_it():
    """The constraint outranks the objective: an over-budget model is not deployable."""
    over = val(fnr=0.001, fpr=0.05)     # far better FNR, but 50x the budget
    under = val(fnr=0.400, fpr=PINNED)  # bad FNR, but inside the budget
    assert run([over, under]) == 2
    assert run([under, over]) == 1      # order must not change the answer


def test_a_run_that_never_meets_the_budget_still_saves_the_closest():
    """Otherwise best.pt would not exist and evaluation would have nothing to load."""
    assert run([val(fnr=0.1, fpr=0.05), val(fnr=0.9, fpr=0.002), val(fnr=0.0, fpr=0.03)]) == 2


def test_the_first_evaluation_is_always_selected():
    assert run([val(fnr=0.5)]) == 1
    assert run([val(fnr=0.5, fpr=0.9)]) == 1
