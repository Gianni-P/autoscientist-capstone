"""Unit tests for the verify orchestrator and report aggregation."""

from __future__ import annotations

from autoscientist.verify import run_all
from autoscientist.verify.types import Verdict, VerifyReport


def test_verdict_validation():
    import pytest

    with pytest.raises(ValueError):
        Verdict(check_id="x", title="X", status="bogus",  # type: ignore[arg-type]
                severity="fail", detail="")
    with pytest.raises(ValueError):
        Verdict(check_id="x", title="X", status="pass",
                severity="bogus", detail="")  # type: ignore[arg-type]


def test_report_outcome_clean():
    rep = VerifyReport.from_verdicts([
        Verdict(check_id="a", title="A", status="pass",
                severity="fail", detail=""),
        Verdict(check_id="b", title="B", status="skipped",
                severity="fail", detail=""),
    ])
    assert rep.outcome == "clean"
    assert "1 pass" in rep.summary
    assert "1 skipped" in rep.summary


def test_report_outcome_block_when_any_fail_with_fail_severity():
    rep = VerifyReport.from_verdicts([
        Verdict(check_id="a", title="A", status="pass",
                severity="fail", detail=""),
        Verdict(check_id="b", title="B", status="fail",
                severity="fail", detail=""),
    ])
    assert rep.outcome == "block"


def test_report_outcome_needs_human():
    rep = VerifyReport.from_verdicts([
        Verdict(check_id="a", title="A", status="needs_human",
                severity="fail", detail=""),
    ])
    assert rep.outcome == "needs_human"


def test_report_warn_severity_does_not_escalate():
    rep = VerifyReport.from_verdicts([
        Verdict(check_id="a", title="A", status="fail",
                severity="warn", detail=""),
    ])
    assert rep.outcome == "clean"


def test_fail_with_needs_human_severity_escalates_human_only():
    rep = VerifyReport.from_verdicts([
        Verdict(check_id="a", title="A", status="fail",
                severity="needs_human", detail=""),
    ])
    assert rep.outcome == "needs_human"


def test_block_outranks_needs_human():
    rep = VerifyReport.from_verdicts([
        Verdict(check_id="a", title="A", status="needs_human",
                severity="fail", detail=""),
        Verdict(check_id="b", title="B", status="fail",
                severity="fail", detail=""),
    ])
    assert rep.outcome == "block"


def test_run_all_clean_state():
    state = {
        "split_strategy": "patient",
        "multi_source": False,
        "claims_generalization": False,
        "test_time_augmentation": False,
        "class_counts_train": {"pos": 100, "neg": 900},
        "class_counts_test": {"pos": 20, "neg": 180},
        "train_ids": ["p1", "p2"], "test_ids": ["p3"],
        "baseline_claims": [
            {"name": "a", "dataset": "d", "metric": "m",
             "published_value": 0.5, "observed_value": 0.50,
             "tolerance_abs": 0.02},
        ],
    }
    rep = run_all(state, domain="medical_imaging")
    assert rep.outcome == "clean"
    # exactly one canonical baseline aggregate verdict
    canonical = [v for v in rep.verdicts
                 if v.check_id == "baseline_reproduced_within_tolerance"]
    assert len(canonical) == 1
    assert canonical[0].category == "pitfall"


def test_run_all_blocks_on_id_overlap():
    state = {
        "train_ids": ["p1", "p2"], "test_ids": ["p2", "p3"],
        "split_strategy": "patient",
    }
    rep = run_all(state, domain="medical_imaging")
    assert rep.outcome == "block"
    overlap = next(v for v in rep.verdicts if v.check_id == "id_overlap")
    assert overlap.status == "fail"


def test_run_all_categorizes_verdicts():
    rep = run_all({}, domain="medical_imaging")
    cats = {v.category for v in rep.verdicts}
    assert {"leakage", "stats", "pitfall"} <= cats


def test_run_all_to_dict_round_trip():
    rep = run_all({}, domain="medical_imaging")
    d = rep.to_dict()
    assert "outcome" in d and "verdicts" in d
    assert isinstance(d["verdicts"], list)
    assert all("check_id" in v and "status" in v for v in d["verdicts"])
