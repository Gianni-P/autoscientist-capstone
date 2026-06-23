"""Unit tests for the provenance / claim-verification verify module."""

from __future__ import annotations

from autoscientist.verify.provenance import (
    _to_float,
    check_claim_coverage,
    check_provenance_entries,
    run_provenance,
)


def test_to_float_parses_latex_and_units():
    assert _to_float("2\\,880") == 2880.0
    assert _to_float("$6.40$") == 6.40
    assert _to_float("45%") == 45.0
    assert _to_float("1,234") == 1234.0
    assert _to_float(0.83) == 0.83
    assert _to_float("not a number") is None


def test_entries_pass_when_value_in_cited_source():
    prov = [{"claim": "mean COG 6.40", "value": 6.40,
             "source_file": "e2_summary.json",
             "source_key": "terrain_summaries[0].mean_cog"}]
    results = {"prod_s0_e2/e2_summary.json":
               {"experiment": "E2",
                "terrain_summaries": [{"terrain": "x", "mean_cog": 6.40}]}}
    v = check_provenance_entries(prov, results)
    assert v.status == "pass"
    assert v.evidence["n_ok"] == 1


def test_entry_value_not_in_source_fails():
    prov = [{"claim": "mean COG 9.99", "value": 9.99,
             "source_file": "e2_summary.json"}]
    results = {"e2_summary.json": {"terrain_summaries": [{"mean_cog": 6.40}]}}
    v = check_provenance_entries(prov, results)
    assert v.status == "fail"
    assert v.evidence["n_mismatch"] == 1


def test_entry_missing_source_fails():
    prov = [{"claim": "x", "value": 1.0, "source_file": "ghost.json"}]
    results = {"e1_summary.json": {"value": 1.0}}
    v = check_provenance_entries(prov, results)
    assert v.status == "fail"
    assert v.evidence["n_missing_source"] == 1


def test_no_manifest_skipped():
    v = check_provenance_entries(None, {"x": {}})
    assert v.status == "skipped"


def test_claim_coverage_flags_uncovered_number():
    v = check_claim_coverage("The corrected optimality gap was 0.842.",
                             provenance=[], plan=None)
    assert v.status == "needs_human"
    assert 0.842 in v.evidence["uncovered_sample"]


def test_claim_coverage_pass_structural_and_plan_numbers():
    plan = {"experiments": [{"id": "E1"}], "grid_n": 300}
    text = "Across 5 starts on a 300x300 grid (2026) with alpha 0.05."
    v = check_claim_coverage(text, provenance=[], plan=plan)
    assert v.status == "pass", v.evidence


def test_claim_coverage_pass_when_value_in_provenance():
    prov = [{"claim": "gap 0.842", "value": 0.842, "source_file": "e1.json"}]
    v = check_claim_coverage("The gap was 0.842.", provenance=prov, plan=None)
    assert v.status == "pass"


def test_no_paper_text_skipped():
    v = check_claim_coverage("", provenance=[], plan=None)
    assert v.status == "skipped"


def test_run_provenance_via_state():
    state = {
        "provenance": [{"claim": "g", "value": 6.40, "source_file": "e2.json"}],
        "results": {"e2.json": {"mean_cog": 6.40}},
        "paper_text": "The gap was 6.40.",
        "plan": {"experiments": [{"id": "E2"}]},
    }
    vs = run_provenance(state)
    assert {v.check_id for v in vs} == {"provenance_entries", "claim_coverage"}
    entries = next(v for v in vs if v.check_id == "provenance_entries")
    assert entries.status == "pass"
