"""Tests for the paper_writer payload reconstruction path.

results_validator (an LLM) frequently forwards a placeholder plan + empty
results to paper_writer (observed 2026-06-18, run_e93293803c98:
``"plan": "<the methodology plan>"`` and ``"results": {"metrics": []}``),
leaving paper_writer nothing to write — it then emits a shell of
``[RESULT FROM run]``/``[CITATION NEEDED]`` markers that peer_reviewer rejects
outright. The runner detects the thin payload and rebuilds it from the run's
real plan + the materialised result JSON in the sandbox. These tests cover the
detector and the reconstruction helper directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoscientist.runtime.payload_files import (
    build_paper_writer_payload_from_sandbox,
)
from autoscientist.runtime.runner import _is_thin_paper_writer_payload


def _make_results_sandbox(root: Path, project_id: str = "p1") -> Path:
    runs = root / project_id / "sandbox" / "runs" / "validator_run"
    runs.mkdir(parents=True)
    (runs / "e1_summary.json").write_text(json.dumps({
        "experiment": "E1",
        "n_trials": 40,
        "terrain_summaries": [
            {"terrain": "elliptic_paraboloid", "mean_qb": 0.0161, "max_qb": 0.082},
            {"terrain": "monkey_saddle", "mean_qb": 8.05e-05, "max_qb": 0.00017},
        ],
        "internal_validity_passed": False,
    }))
    return root / project_id / "sandbox"


# ---------------------------------------------------------------------------
# _is_thin_paper_writer_payload
# ---------------------------------------------------------------------------

def test_thin_empty() -> None:
    assert _is_thin_paper_writer_payload("") is True
    assert _is_thin_paper_writer_payload("   \n ") is True


def test_thin_placeholder_plan() -> None:
    payload = json.dumps({"plan": "<the methodology plan>",
                          "results": {"metrics": []}})
    assert _is_thin_paper_writer_payload(payload) is True


def test_thin_result_from_run_marker() -> None:
    assert _is_thin_paper_writer_payload(
        'The mean was [RESULT FROM run] across terrains.'
    ) is True


def test_thin_empty_results_object() -> None:
    payload = json.dumps({"plan": {"rq": "real plan"}, "results": {"metrics": []}})
    assert _is_thin_paper_writer_payload(payload) is True
    payload2 = json.dumps({"plan": {"rq": "real plan"}, "results": {}})
    assert _is_thin_paper_writer_payload(payload2) is True


def test_not_thin_when_real_numbers_present() -> None:
    payload = json.dumps({
        "plan": {"rq": "real plan"},
        "results": {"terrain_summaries": [{"terrain": "x", "mean_qb": 0.01}]},
    })
    assert _is_thin_paper_writer_payload(payload) is False


def test_not_thin_unparseable_but_substantial() -> None:
    # No JSON object, no placeholder markers, substantial prose — leave it.
    assert _is_thin_paper_writer_payload("a real handoff summary " * 40) is False


# ---------------------------------------------------------------------------
# build_paper_writer_payload_from_sandbox
# ---------------------------------------------------------------------------

def test_rebuild_returns_none_without_results(tmp_path: Path) -> None:
    (tmp_path / "p1" / "sandbox").mkdir(parents=True)
    assert build_paper_writer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, plan_text="plan",
    ) is None


def test_rebuild_carries_real_numbers_and_plan(tmp_path: Path) -> None:
    _make_results_sandbox(tmp_path)
    raw = build_paper_writer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path,
        plan_text=json.dumps({"research_question": "RQ?", "hypotheses": []}),
        validator_summary={"verdict": "advance", "checks": []},
    )
    assert raw is not None
    payload = json.loads(raw)
    # Plan embedded as structured data (not an escaped blob).
    assert payload["plan"]["research_question"] == "RQ?"
    # Real result numbers present, keyed by relative path.
    summ = payload["results"]["validator_run/e1_summary.json"]
    assert summ["n_trials"] == 40
    assert summ["terrain_summaries"][0]["mean_qb"] == 0.0161
    assert payload["validator_summary"]["verdict"] == "advance"
    assert "_reconstructed_by_runner" in payload


def test_rebuild_plan_text_passthrough_when_not_json(tmp_path: Path) -> None:
    _make_results_sandbox(tmp_path)
    raw = build_paper_writer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, plan_text="a plain text plan",
    )
    payload = json.loads(raw)
    assert payload["plan"] == "a plain text plan"


def test_rebuild_truncates_large_jsonl(tmp_path: Path) -> None:
    sandbox = _make_results_sandbox(tmp_path)
    runs = sandbox / "runs" / "validator_run"
    lines = "\n".join(json.dumps({"trial": i}) for i in range(200))
    (runs / "e1_trials.jsonl").write_text(lines)
    raw = build_paper_writer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, plan_text="p",
    )
    rows = json.loads(raw)["results"]["validator_run/e1_trials.jsonl"]
    # Bounded + truncation marker appended.
    assert any("_truncated_by_runner" in r for r in rows if isinstance(r, dict))
    assert len(rows) <= 61


def test_rebuild_caps_total_size_and_keeps_summaries(tmp_path: Path) -> None:
    """Many fat trial dumps must not blow the payload — summaries survive, the
    raw .jsonl dumps are budget-limited, and the total stays context-sized."""
    root, pid = tmp_path, "p1"
    runs = root / pid / "sandbox" / "runs"
    for e in range(1, 6):
        d = runs / f"prod_s0_e{e}"
        d.mkdir(parents=True)
        (d / f"e{e}_summary.json").write_text(json.dumps({
            "experiment": f"E{e}",
            "terrain_summaries": [{"terrain": "t", "mean_qb": 0.01 * e, "max_qb": 0.1}],
        }))
        # ~120 KB of raw trials per experiment (60 rows x ~2 KB) → 600 KB total
        big = "\n".join(json.dumps({"trial": i, "blob": "x" * 2000}) for i in range(60))
        (d / f"e{e}_trials.jsonl").write_text(big)

    raw = build_paper_writer_payload_from_sandbox(
        project_id=pid, projects_root=root, plan_text="p",
    )
    res = json.loads(raw)["results"]
    # Every summary (the numbers) is preserved...
    for e in range(1, 6):
        assert f"prod_s0_e{e}/e{e}_summary.json" in res
        assert res[f"prod_s0_e{e}/e{e}_summary.json"]["terrain_summaries"][0]["mean_qb"] == 0.01 * e
    # ...the oversized trials are budget-limited (so some were omitted)...
    assert "_omitted_by_runner" in res
    # ...and the ACTUAL SENT payload (indent=2 bytes, not the compact results
    # sub-object) is within the hard context ceiling.
    assert len(raw.encode("utf-8")) <= 360_000


def test_rebuild_caps_fat_jsonl_with_no_summary(tmp_path: Path) -> None:
    """A run dir with a fat trial .jsonl and NO summary: the first-admitted file
    must still be byte-bounded (regression from adversarial review — .jsonl had
    no byte cap and the first file was admitted unconditionally → ~480 KB)."""
    root, pid = tmp_path, "p1"
    d = root / pid / "sandbox" / "runs" / "prod_s0_e1"
    d.mkdir(parents=True)
    big = "\n".join(json.dumps({"trial": i, "blob": "x" * 8000}) for i in range(60))
    (d / "e1_trials.jsonl").write_text(big)  # ~480 KB raw, no summary
    raw = build_paper_writer_payload_from_sandbox(
        project_id=pid, projects_root=root, plan_text="p",
    )
    assert raw is not None
    assert len(raw.encode("utf-8")) <= 360_000     # SENT payload is bounded
    res = json.loads(raw)["results"]
    assert "prod_s0_e1/e1_trials.jsonl" in res      # present, just byte-trimmed


def test_rebuild_single_huge_jsonl_bounded(tmp_path: Path) -> None:
    """A single trial log with very large rows (60 x 50 KB ≈ 3 MB) must be
    byte-capped, never admitted whole."""
    root, pid = tmp_path, "p1"
    d = root / pid / "sandbox" / "runs" / "run"
    d.mkdir(parents=True)
    huge = "\n".join(json.dumps({"trial": i, "blob": "y" * 50000}) for i in range(60))
    (d / "trials.jsonl").write_text(huge)
    raw = build_paper_writer_payload_from_sandbox(
        project_id=pid, projects_root=root, plan_text="p",
    )
    assert raw is not None
    assert len(raw.encode("utf-8")) <= 360_000
