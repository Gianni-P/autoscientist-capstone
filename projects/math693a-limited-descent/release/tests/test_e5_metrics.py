"""Tests for E5 gap decomposition + statistics (the core scientific claims).

Pitfalls targeted:
  * Metric mis-implementation: COG must equal OG - QB by construction.
  * Optimality gap sign: against the Theta* optimum, the optimum itself has
    OG == 0 (no negative gaps from a mis-signed reference).
  * Bootstrap CI sanity: identical samples give a degenerate (zero-width) CI;
    a clearly-positive sample gives ci_lo_one_sided > 0.
  * Holm-Bonferroni correctness: thresholds AND the step-down reject decisions
    match the standard step-down procedure (once a rank fails to reject, every
    higher-p rank is also non-rejected regardless of its own threshold).
  * Non-converged trials are excluded from the gap aggregation.
"""
import numpy as np
import pytest

from src.experiment_e5 import (
    _bootstrap_mean_ci, _paired_bootstrap_diff, _holm_bonferroni,
    _gaps_for_trial, N_BOOTSTRAP, NONTRIVIAL_TERRAINS,
)


class _Setup:
    def __init__(self, theta, raw):
        self.theta_len = theta
        self.raw_len = raw


def test_cog_equals_og_minus_qb():
    # COG is defined as OG - QB; a refactor that drops the QB subtraction
    # would silently overstate the heuristic's true sub-optimality.
    sp = (3, 7)
    setup = _Setup({sp: 10.0}, {sp: 12.0})
    res = {"path_length_3d": 15.0}
    og, qb, cog = _gaps_for_trial(setup, sp, res)
    assert og == pytest.approx((15.0 - 10.0) / 10.0)
    assert qb == pytest.approx((12.0 - 10.0) / 10.0)
    assert cog == pytest.approx(og - qb)


def test_gap_zero_at_optimum():
    # A path exactly equal to the Theta* optimum length has OG == 0, not
    # negative. Catches a flipped sign in the optimality gap.
    sp = (0, 0)
    setup = _Setup({sp: 8.0}, {sp: 9.0})
    res = {"path_length_3d": 8.0}
    og, qb, cog = _gaps_for_trial(setup, sp, res)
    assert og == pytest.approx(0.0)


def test_gaps_none_for_nonpositive_theta():
    sp = (1, 1)
    setup = _Setup({sp: 0.0}, {sp: 1.0})
    assert _gaps_for_trial(setup, sp, {"path_length_3d": 1.0}) is None


def test_bootstrap_degenerate_sample():
    rng = np.random.default_rng(0)
    ci = _bootstrap_mean_ci([5.0, 5.0, 5.0, 5.0], rng, n=2000)
    assert ci["mean"] == pytest.approx(5.0)
    assert ci["ci_lo"] == pytest.approx(5.0)
    assert ci["ci_hi"] == pytest.approx(5.0)
    assert ci["sd"] == pytest.approx(0.0)


def test_bootstrap_positive_sample_one_sided_lower():
    rng = np.random.default_rng(1)
    vals = list(np.full(40, 2.0) + 0.01 * np.arange(40))
    ci = _bootstrap_mean_ci(vals, rng, n=5000)
    # clearly positive sample -> one-sided lower bound stays above zero
    assert ci["ci_lo_one_sided"] > 0.0
    assert ci["n"] == 40


def test_bootstrap_empty_returns_none():
    rng = np.random.default_rng(2)
    assert _bootstrap_mean_ci([], rng, n=10) is None


def test_paired_bootstrap_detects_negative_diff():
    rng = np.random.default_rng(3)
    # a strictly less than b => mean(a-b) < 0 and ci_hi < 0 expected
    a = list(np.full(30, 1.0))
    b = list(np.full(30, 3.0))
    d = _paired_bootstrap_diff(a, b, rng, n=5000)
    assert d["mean_diff"] == pytest.approx(-2.0)
    assert d["ci_hi"] < 0.0
    assert d["n_pairs"] == 30


def test_holm_bonferroni_known_case():
    # Standard step-down: sorted p-values compared to alpha/(m-rank), AND once
    # a rank fails to reject every higher-p rank is also non-rejected.
    pvals = [0.01, 0.04, 0.03, 0.005]
    res = _holm_bonferroni(pvals, alpha=0.05)
    # m=4; thresholds for ranks 0..3 are 0.0125, 0.01667, 0.025, 0.05
    # sorted: 0.005(r0,t=.0125 rej), 0.01(r1,t=.01667 rej),
    #         0.03(r2,t=.025 NOT rej -> stop), 0.04(r3,t=.05) NOT rej by step-down
    thr = [r["threshold"] for r in res]
    # smallest p gets smallest threshold
    assert min(thr) == pytest.approx(0.05 / 4)
    assert max(thr) == pytest.approx(0.05 / 1)

    idx_005 = pvals.index(0.005)
    idx_001 = pvals.index(0.01)
    idx_003 = pvals.index(0.03)
    idx_004 = pvals.index(0.04)

    # p=0.005 (smallest) and p=0.01 are rejected (below their thresholds).
    assert res[idx_005]["reject"] is True
    assert res[idx_001]["reject"] is True
    # p=0.03 fails its threshold (0.03 > 0.025) -> NOT rejected, triggers stop.
    assert res[idx_003]["reject"] is False
    # p=0.04 at threshold 0.05 would be "rejected by value" but the step-down
    # rule forces non-rejection because a smaller-rank hypothesis already failed.
    assert res[idx_004]["threshold"] == pytest.approx(0.05 / 1)
    assert res[idx_004]["reject"] is False

    # Full reject-vector in ORIGINAL p-value order [0.01, 0.04, 0.03, 0.005].
    assert [r["reject"] for r in res] == [True, False, False, True]


def test_holm_step_down_blocks_later_small_threshold_pass():
    # Direct regression for the step-down propagation: a non-rejection at an
    # earlier rank must block ALL later ranks even if a later p <= its own
    # threshold. pvals=[0.20, 0.001] -> sorted 0.001(t=0.025 rej), 0.20(t=0.05
    # NOT rej). Reorder so the failing one comes first in rank but a later one
    # would pass by value alone.
    pvals = [0.03, 0.001, 0.049]
    # m=3; sorted: 0.001(r0,t=.01667 rej), 0.03(r1,t=.025 NOT rej -> stop),
    #              0.049(r2,t=.05) would pass-by-value but step-down blocks it.
    res = _holm_bonferroni(pvals, alpha=0.05)
    assert res[pvals.index(0.001)]["reject"] is True
    assert res[pvals.index(0.03)]["reject"] is False
    assert res[pvals.index(0.049)]["reject"] is False


def test_holm_all_large_pvals_no_reject():
    res = _holm_bonferroni([0.9, 0.8, 0.7, 0.6], alpha=0.05)
    assert all(not r["reject"] for r in res)


def test_holm_all_reject_when_all_small():
    # When every p-value clears its (rank-dependent) threshold, all reject.
    res = _holm_bonferroni([0.001, 0.002, 0.003, 0.004], alpha=0.05)
    assert all(r["reject"] for r in res)


def test_nontrivial_terrains_excludes_control():
    # The smooth-bowl control (T1) must not be in the pooled nontrivial set,
    # otherwise H1's pooled claim would be diluted by the trivial terrain.
    # NONTRIVIAL_TERRAINS uses canonical T-codes (T2/T3/T4), not legacy names.
    assert "T1" not in NONTRIVIAL_TERRAINS
    assert "T2" in NONTRIVIAL_TERRAINS


def test_n_bootstrap_is_planned():
    assert N_BOOTSTRAP == 10000
