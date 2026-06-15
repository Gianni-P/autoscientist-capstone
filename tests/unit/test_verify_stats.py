"""Unit tests for autoscientist.verify.stats."""

from __future__ import annotations

from autoscientist.verify import stats
from autoscientist.verify.stats import (
    check_multicollinearity,
    check_normality,
    check_sample_size,
)

# -- multicollinearity -------------------------------------------------------


def test_multicollinearity_perfect_pair_fails():
    v = check_multicollinearity({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [10, 20, 30, 40, 50, 60],
        "z": [3, 1, 4, 1, 5, 9],
    })
    assert v.status == "fail"
    pairs = v.evidence["fails"]
    assert any(("x" in (p["a"], p["b"]) and "y" in (p["a"], p["b"])) for p in pairs)


def test_multicollinearity_warn_threshold_humans():
    # Custom thresholds ensure we land in the warn band regardless of
    # the exact noisy r-value we synthesize.
    v = check_multicollinearity(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2, 1, 4, 5, 3, 7, 5, 9, 6, 10],   # |r| ≈ 0.86
            "z": [9, 1, 3, 2, 4, 7, 5, 8, 6, 10],
        },
        fail_threshold=0.95,
        warn_threshold=0.80,
    )
    assert v.status == "needs_human"


def test_multicollinearity_independent_passes():
    v = check_multicollinearity({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],   # |r|=1 actually
    })
    # perfect anti-correlation is also collinear → fail
    assert v.status == "fail"


def test_multicollinearity_skipped_when_too_few_columns():
    v = check_multicollinearity({"x": [1, 2, 3]})
    assert v.status == "skipped"


def test_multicollinearity_no_input():
    assert check_multicollinearity(None).status == "skipped"
    assert check_multicollinearity({}).status == "skipped"


# -- normality ---------------------------------------------------------------


def test_normality_balanced_passes():
    v = check_normality({"x": [-2, -1, -1, 0, 0, 0, 0, 1, 1, 2]})
    assert v.status == "pass"


def test_normality_extreme_skew_flags():
    v = check_normality({"x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]})
    assert v.status == "needs_human"
    row = v.evidence["violators"][0]
    assert row["name"] == "x"
    assert row["skew"] > 2.0


def test_normality_too_few_samples_skipped():
    v = check_normality({"x": [1, 2, 3, 4, 5]})
    assert v.status == "skipped"


def test_normality_no_input():
    assert check_normality(None).status == "skipped"
    assert check_normality({}).status == "skipped"


# -- sample size -------------------------------------------------------------


def test_sample_size_rare_class_fails():
    v = check_sample_size(class_counts={"pneumonia": 3, "normal": 1000})
    assert v.status == "fail"
    assert v.evidence["min_class"] == "pneumonia"


def test_sample_size_warn_band():
    v = check_sample_size(class_counts={"pos": 7, "neg": 1000})
    assert v.status == "needs_human"


def test_sample_size_pass():
    v = check_sample_size(class_counts={"pos": 200, "neg": 1000})
    assert v.status == "pass"


def test_sample_size_epv_rule_binary():
    # rare=50, n_predictors=10 → EPV needs 100 → fail
    v = check_sample_size(
        class_counts={"pos": 50, "neg": 1000}, n_predictors=10,
    )
    assert v.status == "fail"


def test_sample_size_epv_rule_passes_when_enough_events():
    v = check_sample_size(
        class_counts={"pos": 200, "neg": 1000}, n_predictors=10,
    )
    assert v.status == "pass"


def test_sample_size_no_class_counts_skipped():
    v = check_sample_size(class_counts=None)
    assert v.status == "skipped"


# -- runner ------------------------------------------------------------------


def test_run_stats_emits_four_verdicts():
    # discrimination_floor added 2026-05-31 (near-chance AUROC guard, item 4).
    out = stats.run_stats({})
    ids = {v.check_id for v in out}
    assert ids == {"multicollinearity", "normality", "sample_size", "discrimination_floor"}
    assert all(v.status == "skipped" for v in out)
