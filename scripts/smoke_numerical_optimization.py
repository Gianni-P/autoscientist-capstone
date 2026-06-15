"""Smoke test - numerical_optimization domain pitfall coverage.

Guards the new `config/domains/numerical_optimization.toml` and the six
handlers added to `verify/pitfalls.py` for the math693a-limited-descent
project (constrained / safe-descent path-finding). Mirrors the structure of
scripts/smoke_phase7.py (the medical-imaging domain smoke):

  * The TOML carries all ten checks with their declared severities
    (six optimization-specific + four reused generic).
  * Registry coverage: every TOML id has a registered handler. A silent
    ``skipped`` because no one wrote a handler is a regression.
  * A "fixed limited-descent study" state dict drives every pitfall to
    ``pass`` end to end.
  * Targeted state mutations - the failure modes the *original* project hit
    or flagged - land each pitfall on its expected failing verdict
    (fail / needs_human / warn-fail).
  * Orchestrator outcome aggregation respects the severities.
  * Zero LLM spend.

    uv run python scripts/smoke_numerical_optimization.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_numopt.db"
_RUNS = _REPO / "runs_smoke_numopt"
if _DB.exists():
    _DB.unlink()
if _RUNS.exists():
    shutil.rmtree(_RUNS)

os.environ["AUTOSCIENTIST_DB_PATH"] = str(_DB)

_DOMAIN = "numerical_optimization"


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def passed(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    raise AssertionError(msg)


def expect(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)
    passed(msg)


# A *fixed* limited-descent study: the original's confounds repaired.
#   * a defined sink so all rotation strategies share an endpoint;
#   * a bounded rotation loop + convergence;
#   * the analytic gradient validated against finite differences;
#   * a grid Dijkstra optimum to measure an optimality gap against;
#   * >= 5 start points (injured-hiker locations) with variance + CIs;
#   * unconstrained steepest descent reproduces the analytic minimizer.
# This state must drive all ten pitfalls to pass.
LIMITED_DESCENT_CLEAN_STATE: dict = {
    # constraint_feasibility_verified - grade <= theta at every step
    "constraint_satisfied": True,
    "constraint_violations": 0,
    # descent_terminates - bounded rotation loop + converged within budget
    "rotation_turn_cap": 180,
    "converged": True,
    "hit_iteration_cap": False,
    # well_defined_convergence_target - shared sink => endpoints comparable
    "objective_has_defined_target": True,
    "convergence_target": [2.0, -1.0],
    "endpoints_comparable": True,
    "compares_path_length": True,
    # gradient_validated
    "gradient_validation": "finite_difference",
    # optimality_gap_reported - claim backed by a reference optimum + gap
    "claims_optimality": True,
    "optimum_reference": "grid_dijkstra_safe_shortest_path",
    "optimality_gap": 0.12,
    # step_discretization_sensitivity
    "step_sensitivity_analyzed": True,
    # multi_seed_reporting (reused) - >= 5 start points with variance
    "seeds": [0, 1, 2, 3, 4],
    "report_seed_variance": True,
    # counterintuitive_signs_flagged (reused) - heuristic gap is >= 0 as expected
    "hypothesized_effects": {"rotation_gap_vs_optimum": "positive"},
    "observed_effects": {
        "rotation_gap_vs_optimum": {"sign": "positive", "value": 0.12},
    },
    # confidence_intervals_reported (reused)
    "comparison_claims": [
        "rotation_heuristic path length > cone_projection path length",
    ],
    "confidence_intervals_present": True,
    "confidence_interval_method": "bootstrap_2000_over_starts",
    # baseline_reproduced_within_tolerance (reused; delegates to baseline_repro)
    "baseline_claims": [
        {
            "name": "unconstrained_sd_recovers_minimizer",
            "dataset": "mountain_two",
            "metric": "minimizer_L2_error",
            "published_value": 0.0,
            "observed_value": 0.004,
            "tolerance_abs": 0.01,
            "higher_is_better": False,
        },
        {
            "name": "grid_optimum_on_hand_checked_terrain",
            "dataset": "quadratic_basin",
            "metric": "shortest_safe_path_length",
            "published_value": 2.8284,
            "observed_value": 2.8290,
            "tolerance_abs": 0.02,
        },
    ],
}


def main() -> int:
    from autoscientist.verify import pitfalls, run_all
    from autoscientist.verify.pitfalls import (
        domain_config_path,
        get_handler,
        load_pitfall_config,
        run_pitfalls,
    )

    def pf(state: dict):
        return run_pitfalls(state, domain=_DOMAIN)

    def verdict(state: dict, check_id: str):
        return next(x for x in pf(state) if x.check_id == check_id)

    def without(*keys: str) -> dict:
        return {k: v for k, v in LIMITED_DESCENT_CLEAN_STATE.items() if k not in keys}

    # ------------------------------------------------------------------
    # 1) TOML carries all ten checks with the right severities.
    # ------------------------------------------------------------------
    section("Pitfall TOML: numerical_optimization checks present with severities")
    cfg_path = domain_config_path(_DOMAIN)
    expect(cfg_path.exists(), f"{_DOMAIN}.toml exists at {cfg_path}")
    checks = load_pitfall_config(cfg_path)
    by_id = {c.id: c for c in checks}

    expected_new = {
        "constraint_feasibility_verified": "fail",
        "descent_terminates": "fail",
        "well_defined_convergence_target": "fail",
        "gradient_validated": "needs_human",
        "optimality_gap_reported": "needs_human",
        "step_discretization_sensitivity": "warn",
    }
    expected_reused = {
        "baseline_reproduced_within_tolerance",
        "multi_seed_reporting",
        "counterintuitive_signs_flagged",
        "confidence_intervals_reported",
    }
    for cid, sev in expected_new.items():
        expect(cid in by_id, f"optimization check '{cid}' present in TOML")
        expect(by_id[cid].severity == sev,
               f"'{cid}' declared severity '{sev}' (got '{by_id[cid].severity}')")
        expect(len(by_id[cid].description) > 40,
               f"'{cid}' carries a non-trivial description")
    expect(expected_reused <= set(by_id),
           f"reused generic checks present (missing={expected_reused - set(by_id)})")
    expect(len(checks) == 10, f"exactly 10 checks declared (got {len(checks)})")

    # ------------------------------------------------------------------
    # 2) Registry coverage: every TOML id has a handler.
    # ------------------------------------------------------------------
    section("Registry: every TOML id has a registered handler")
    unhandled = [c.id for c in checks if get_handler(c.id) is None]
    expect(unhandled == [], f"unhandled pitfalls: {unhandled}")

    # ------------------------------------------------------------------
    # 3) The fixed limited-descent state passes every pitfall.
    # ------------------------------------------------------------------
    section("Clean state: all 10 pitfalls pass")
    pf_by = {v.check_id: v for v in pf(LIMITED_DESCENT_CLEAN_STATE)}
    for c in checks:
        v = pf_by.get(c.id)
        expect(v is not None, f"verdict emitted for '{c.id}'")
        expect(v.status == "pass",
               f"'{c.id}' is pass on clean state "
               f"(got status={v.status}, detail={v.detail!r})")

    # ------------------------------------------------------------------
    # 4) Targeted mutations: each pitfall fires on the original's flaws.
    # ------------------------------------------------------------------
    section("Targeted mutations: each pitfall fires correctly")

    # 4a. Over-steep step anywhere -> constraint_feasibility_verified fail.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE, "constraint_violations": 3},
                "constraint_feasibility_verified")
    expect(v.status == "fail" and "3" in v.detail,
           f"constraint violated -> fail (got {v.status})")

    # 4b. Unbounded rotation loop (the original's appendix bug) -> fail.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE, "rotation_turn_cap": False},
                "descent_terminates")
    expect(v.status == "fail" and "turn cap" in v.detail,
           f"no turn cap -> descent_terminates fail (got {v.status})")

    # 4c. Hit the iteration cap without converging -> fail.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE,
                 "hit_iteration_cap": True, "converged": False},
                "descent_terminates")
    expect(v.status == "fail" and "iteration cap" in v.detail,
           f"iteration cap without convergence -> fail (got {v.status})")

    # 4d. No defined target while comparing path lengths (the central confound).
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE,
                 "objective_has_defined_target": False,
                 "convergence_target": None,
                 "endpoints_comparable": False},
                "well_defined_convergence_target")
    expect(v.status == "fail" and "ill-posed" in v.detail,
           f"undefined target + path comparison -> fail (got {v.status})")

    # 4e. Hand-coded gradient never checked -> needs_human.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE,
                 "gradient_validation": "analytic_unverified"},
                "gradient_validated")
    expect(v.status == "needs_human",
           f"unverified gradient -> needs_human (got {v.status})")

    # 4f. Optimality claim with no reference optimum -> needs_human.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE, "optimum_reference": None,
                 "optimality_gap": None},
                "optimality_gap_reported")
    expect(v.status == "needs_human",
           f"optimality claim without reference -> needs_human (got {v.status})")

    # 4g. No step/rotation sensitivity analysis -> warn-fail.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE, "step_sensitivity_analyzed": False},
                "step_discretization_sensitivity")
    expect(v.status == "fail" and v.severity == "warn",
           f"no sensitivity analysis -> warn-fail (got {v.status}/{v.severity})")

    # 4h. Single start point (the original used one) -> multi_seed_reporting fail.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE, "seeds": [0]},
                "multi_seed_reporting")
    expect(v.status == "fail" and "1" in v.detail,
           f"single start -> multi_seed_reporting fail (got {v.status})")

    # 4i. cw/ccw asymmetry inverts the expected gap sign -> counterintuitive fail.
    v = verdict({**LIMITED_DESCENT_CLEAN_STATE,
                 "observed_effects": {
                     "rotation_gap_vs_optimum": {"sign": "negative", "value": -0.04},
                 }},
                "counterintuitive_signs_flagged")
    expect(v.status == "fail",
           f"heuristic 'shorter than optimum' inversion -> fail (got {v.status})")

    # 4j. Comparison claim without CIs -> needs_human.
    v = verdict(without("confidence_intervals_present"),
                "confidence_intervals_reported")
    expect(v.status == "needs_human",
           f"comparison without CIs -> needs_human (got {v.status})")

    # 4k. Optimality/novelty without a reproduced baseline -> baseline fail.
    v = verdict(without("baseline_claims"),
                "baseline_reproduced_within_tolerance")
    expect(v.status != "pass",
           f"no baseline reproduced -> not pass (got {v.status})")

    # ------------------------------------------------------------------
    # 5) Orchestrator outcome aggregation honours the severities.
    # ------------------------------------------------------------------
    section("Orchestrator: aggregated outcome reflects severities")
    rep_clean = run_all(LIMITED_DESCENT_CLEAN_STATE, domain=_DOMAIN)
    expect(rep_clean.outcome == "clean",
           f"clean state -> outcome=clean (got {rep_clean.outcome}: {rep_clean.summary})")

    # A 'fail'-severity mutation must escalate to block.
    rep_block = run_all({**LIMITED_DESCENT_CLEAN_STATE, "constraint_violations": 3},
                        domain=_DOMAIN)
    expect(rep_block.outcome == "block",
           f"constraint violation -> outcome=block (got {rep_block.outcome})")
    block_ids = {v.check_id for v in rep_block.by_status("fail")}
    expect("constraint_feasibility_verified" in block_ids,
           "constraint_feasibility_verified present in fail set")

    # A 'needs_human'-severity mutation escalates to needs_human (not block).
    rep_human = run_all({**LIMITED_DESCENT_CLEAN_STATE,
                         "gradient_validation": "analytic_unverified"},
                        domain=_DOMAIN)
    expect(rep_human.outcome == "needs_human",
           f"unverified gradient -> outcome=needs_human (got {rep_human.outcome})")

    # A 'warn'-severity mutation must NOT escalate.
    rep_warn = run_all({**LIMITED_DESCENT_CLEAN_STATE,
                        "step_sensitivity_analyzed": False}, domain=_DOMAIN)
    expect(rep_warn.outcome == "clean",
           f"warn-severity sensitivity fail does not escalate "
           f"(got {rep_warn.outcome}: {rep_warn.summary})")
    warn_v = next(v for v in rep_warn.verdicts
                  if v.check_id == "step_discretization_sensitivity")
    expect(warn_v.status == "fail" and warn_v.severity == "warn",
           "warn-severity verdict recorded with status=fail, severity=warn")

    # ------------------------------------------------------------------
    # 6) Spend invariant.
    # ------------------------------------------------------------------
    section("Spend invariant: zero spend (offline)")
    from autoscientist.state.db import open_db
    conn = open_db(_DB)
    try:
        spend = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger"
        ).fetchone()["s"]
        expect(spend == 0.0, f"zero spend (got ${spend})")
    finally:
        conn.close()

    # Touch the pitfalls module name so the linter sees the import is used.
    assert pitfalls is not None

    print("\n*** All numerical_optimization smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
