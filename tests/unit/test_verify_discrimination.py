"""Tests for the near-chance / discrimination-floor guard (2026-05-31 audit, item 4).

Adversarial intent: feed synthetic near-chance and worse-than-chance AUROCs and
confirm the guard fires; feed clearly-discriminating AUROCs and confirm it stays
silent. The real E1/E2 PadChest transfer (AUROC 0.36-0.55, CIs straddling 0.5)
is the motivating case the human caught and the system did not.
"""

from __future__ import annotations

import pytest

from autoscientist.verify import run_all
from autoscientist.verify.stats import check_discrimination_floor


def test_ci_includes_chance_is_needs_human():
    v = check_discrimination_floor(
        [{"label": "pad_auroc", "point_estimate": 0.45,
          "ci_lower": 0.41, "ci_upper": 0.49 + 0.02}]  # CI 0.41-0.51 includes 0.5
    )
    assert v.status == "needs_human"
    assert "chance" in v.detail.lower()


def test_ci_entirely_below_chance_is_fail():
    # E1 N=5000 seed42 pad: 0.363 [0.330, 0.397] — whole CI below 0.5.
    v = check_discrimination_floor(
        [{"label": "pad_auroc", "point_estimate": 0.363,
          "ci_lower": 0.330, "ci_upper": 0.397}]
    )
    assert v.status == "fail"
    assert v.evidence["worse_than_chance"]


def test_point_near_chance_without_ci_is_needs_human():
    v = check_discrimination_floor([{"label": "auroc", "value": 0.51}])  # within 0.02
    assert v.status == "needs_human"


def test_clearly_above_chance_passes():
    v = check_discrimination_floor(
        [{"label": "nih_auroc", "point_estimate": 0.76,
          "ci_lower": 0.73, "ci_upper": 0.79}]
    )
    assert v.status == "pass"


def test_non_primary_near_chance_does_not_escalate():
    v = check_discrimination_floor([
        {"label": "primary", "point_estimate": 0.80, "ci_lower": 0.75, "ci_upper": 0.85},
        {"label": "aux", "point_estimate": 0.50, "ci_lower": 0.45, "ci_upper": 0.55,
         "primary": False},
    ])
    assert v.status == "pass"


def test_empty_skips():
    assert check_discrimination_floor(None).status == "skipped"
    assert check_discrimination_floor([]).status == "skipped"


def test_default_severity_does_not_hard_block():
    """Worse-than-chance is a 'fail' status but default severity needs_human →
    it surfaces a checkpoint (needs_human outcome), it does NOT hard-block."""
    v = check_discrimination_floor(
        [{"label": "x", "point_estimate": 0.36, "ci_lower": 0.33, "ci_upper": 0.40}]
    )
    assert v.status == "fail"
    assert v.escalates_human and not v.escalates_block


def test_run_all_surfaces_near_chance():
    """End-to-end through run_all: near-chance AUROC → outcome needs_human."""
    state = {
        "auroc_results": [
            {"label": "pad_test_auroc", "point_estimate": 0.43,
             "ci_lower": 0.39, "ci_upper": 0.47},
        ],
    }
    rep = run_all(state, domain="medical_imaging")
    floor = next(v for v in rep.verdicts if v.check_id == "discrimination_floor")
    assert floor.status in ("needs_human", "fail")
    assert rep.outcome in ("needs_human", "block")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
