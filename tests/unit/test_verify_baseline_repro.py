"""Unit tests for autoscientist.verify.baseline_repro."""

from __future__ import annotations

import pytest

from autoscientist.verify import baseline_repro
from autoscientist.verify.baseline_repro import BaselineClaim, check_baseline


def test_within_absolute_tolerance_passes():
    claim = BaselineClaim(
        name="CheXNet", dataset="NIH", metric="AUROC",
        published_value=0.768, observed_value=0.770,
        tolerance_abs=0.01,
    )
    v = check_baseline(claim)
    assert v.status == "pass"
    assert v.evidence["abs_delta"] == pytest.approx(0.002, abs=1e-9)


def test_outside_tolerance_fails():
    claim = BaselineClaim(
        name="CheXNet", dataset="NIH", metric="AUROC",
        published_value=0.768, observed_value=0.620,
        tolerance_abs=0.02,
    )
    v = check_baseline(claim)
    assert v.status == "fail"
    assert v.evidence["abs_delta"] == pytest.approx(0.148, abs=1e-9)


def test_relative_tolerance_used_when_abs_missing():
    claim = BaselineClaim(
        name="x", dataset="d", metric="m",
        published_value=100.0, observed_value=104.0,
        tolerance_rel=0.05,   # 5% of 100 = 5 abs
    )
    v = check_baseline(claim)
    assert v.status == "pass"
    assert v.evidence["tolerance_abs"] == pytest.approx(5.0, abs=1e-9)


def test_abs_tolerance_takes_precedence():
    claim = BaselineClaim(
        name="x", dataset="d", metric="m",
        published_value=100.0, observed_value=104.0,
        tolerance_abs=2.0, tolerance_rel=0.10,
    )
    v = check_baseline(claim)
    assert v.status == "fail"  # 4 > 2
    assert v.evidence["tolerance_abs"] == 2.0


def test_default_tolerance_when_neither_provided():
    claim = BaselineClaim(
        name="x", dataset="d", metric="m",
        published_value=0.5, observed_value=0.51,
    )
    v = check_baseline(claim)
    assert v.status == "pass"   # default 0.02 covers 0.01


def test_run_with_no_claims_skipped_aggregate():
    out = baseline_repro.run_baseline_repro({})
    aggregate = next(v for v in out
                     if v.check_id == "baseline_reproduced_within_tolerance")
    assert aggregate.status == "skipped"


def test_at_least_one_pass_makes_aggregate_pass():
    state = {"baseline_claims": [
        {"name": "a", "dataset": "d", "metric": "m",
         "published_value": 0.5, "observed_value": 0.51, "tolerance_abs": 0.02},
        {"name": "b", "dataset": "d", "metric": "m",
         "published_value": 0.5, "observed_value": 0.10, "tolerance_abs": 0.02},
    ]}
    out = baseline_repro.run_baseline_repro(state)
    aggregate = next(v for v in out
                     if v.check_id == "baseline_reproduced_within_tolerance")
    assert aggregate.status == "pass"
    assert aggregate.evidence["n_pass"] == 1
    assert aggregate.evidence["n_fail"] == 1


def test_all_fails_make_aggregate_fail():
    state = {"baseline_claims": [
        {"name": "a", "dataset": "d", "metric": "m",
         "published_value": 0.5, "observed_value": 0.20, "tolerance_abs": 0.02},
    ]}
    out = baseline_repro.run_baseline_repro(state)
    aggregate = next(v for v in out
                     if v.check_id == "baseline_reproduced_within_tolerance")
    assert aggregate.status == "fail"


def test_novelty_without_baseline_overrides_to_fail():
    state = {"claims_novelty": True}
    out = baseline_repro.run_baseline_repro(state)
    aggregate = next(v for v in out
                     if v.check_id == "baseline_reproduced_within_tolerance")
    assert aggregate.status == "fail"
    assert aggregate.severity == "fail"
    assert aggregate.evidence.get("novelty_claimed") is True


def test_novelty_with_passing_baseline_passes():
    state = {
        "claims_novelty": True,
        "baseline_claims": [
            {"name": "a", "dataset": "d", "metric": "m",
             "published_value": 0.5, "observed_value": 0.51, "tolerance_abs": 0.02},
        ],
    }
    out = baseline_repro.run_baseline_repro(state)
    aggregate = next(v for v in out
                     if v.check_id == "baseline_reproduced_within_tolerance")
    assert aggregate.status == "pass"


def test_malformed_claim_emits_error_verdict():
    state = {"baseline_claims": [{"name": "a"}]}  # missing required fields
    out = baseline_repro.run_baseline_repro(state)
    err = next(v for v in out if v.status == "error")
    assert err.check_id == "baseline_repro::malformed"
