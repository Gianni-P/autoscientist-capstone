"""Phase 7 smoke test - domain hardening (medical-imaging pitfall coverage).

KICKOFF.md S9 Phase 7 - "Build out config/domains/medical_imaging.toml
against the v1 test project. As pitfalls emerge during the v1 run,
codify them." This smoke nails the v1-relevant additions:

  * The TOML carries every Phase-5 pitfall plus the Phase-7 additions
    (multi_seed_reporting, hyperparameter_tuning_split,
    weak_label_provenance_disclosed, view_projection_documented,
    confidence_intervals_reported).
  * Registry coverage: every TOML id has a registered handler. A
    silent ``skipped`` because no one wrote a handler is a regression.
  * A pneumonia-data-efficiency-shaped state dict (KICKOFF S8) drives
    every pitfall to ``pass`` end to end.
  * Targeted state mutations land each new pitfall on its expected
    failing verdict (fail / needs_human / warn-fail).
  * Orchestrator outcome aggregation respects the new severities.
  * Zero LLM spend.

    uv run python scripts/smoke_phase7.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_phase7.db"
_RUNS = _REPO / "runs_smoke_phase7"
if _DB.exists():
    _DB.unlink()
if _RUNS.exists():
    shutil.rmtree(_RUNS)

os.environ["AUTOSCIENTIST_DB_PATH"] = str(_DB)


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


# Pneumonia-data-efficiency state per KICKOFF S8 - patient-level NIH
# split, external PadChest validation, 4-point training-size sweep,
# 3 seeds per N, baseline reproduced against CheXNet.
PNEUMONIA_CLEAN_STATE: dict = {
    # Splitting / sites (Phase 5 pitfalls)
    "split_strategy": "patient",
    "multi_source": True,
    "sites": ["NIH-ChestX-ray14", "PadChest"],
    "site_stratified": True,
    "claims_generalization": True,
    "external_validation_datasets": ["PadChest"],
    "test_time_augmentation": False,
    "hypothesized_effects": {"training_size": "positive"},
    "observed_effects": {
        "training_size": {"sign": "positive", "value": 0.06},
    },
    "claims_novelty": False,
    "class_counts_train": {"pneumonia": 5000, "normal": 95000},
    "class_counts_test": {"pneumonia": 200, "normal": 1500},

    # Phase 7 additions
    "seeds": [0, 1, 2],
    "report_seed_variance": True,
    "hyperparameter_tuning_split": "validation",
    "label_provenance": {
        "NIH-ChestX-ray14": "nlp_derived",
        "PadChest": "nlp_derived",
    },
    "weak_label_limitation_disclosed": True,
    "view_projections": ["PA", "AP"],
    "view_projection_stratified": True,
    "comparison_claims": [
        "AUROC at N=100k > AUROC at N=1k (in-domain)",
        "Generalization gap shrinks with training size",
    ],
    "confidence_intervals_present": True,
    "confidence_interval_method": "bootstrap_2000",

    # Leakage + baseline (orchestrator)
    "train_ids": [f"NIH-{i}" for i in range(50)],
    "val_ids":   [f"NIH-{i}" for i in range(50, 60)],
    "test_ids":  [f"NIH-{i}" for i in range(60, 80)],
    "baseline_claims": [
        {
            "name": "CheXNet-Rajpurkar2017",
            "dataset": "NIH-ChestX-ray14",
            "metric": "AUROC-pneumonia",
            "published_value": 0.7680,
            "observed_value": 0.7705,
            "tolerance_abs": 0.01,
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

    # ------------------------------------------------------------------
    # 1) TOML carries the Phase 7 additions with the right severities.
    # ------------------------------------------------------------------
    section("Pitfall TOML: Phase 7 additions present with declared severities")
    cfg_path = domain_config_path("medical_imaging")
    expect(cfg_path.exists(), f"medical_imaging.toml exists at {cfg_path}")
    checks = load_pitfall_config(cfg_path)
    by_id = {c.id: c for c in checks}

    expected_phase5 = {
        "patient_level_split", "site_stratification",
        "external_validation_present",
        "no_test_time_augmentation_in_baseline",
        "counterintuitive_signs_flagged",
        "baseline_reproduced_within_tolerance",
        "class_balance_reported",
    }
    expected_phase7 = {
        "multi_seed_reporting": "fail",
        "hyperparameter_tuning_split": "fail",
        "weak_label_provenance_disclosed": "needs_human",
        "view_projection_documented": "warn",
        "confidence_intervals_reported": "needs_human",
    }
    expect(expected_phase5 <= set(by_id),
           f"Phase 5 pitfalls intact (missing={expected_phase5 - set(by_id)})")
    for cid, sev in expected_phase7.items():
        expect(cid in by_id, f"Phase 7 addition '{cid}' present in TOML")
        expect(by_id[cid].severity == sev,
               f"'{cid}' declared severity '{sev}' "
               f"(got '{by_id[cid].severity}')")
        expect(len(by_id[cid].description) > 40,
               f"'{cid}' carries a non-trivial description")

    # ------------------------------------------------------------------
    # 2) Registry coverage: every TOML id has a handler. A silent
    #    'skipped: no handler registered' is a regression.
    # ------------------------------------------------------------------
    section("Registry: every TOML id has a registered handler")
    unhandled = [c.id for c in checks if get_handler(c.id) is None]
    expect(unhandled == [], f"unhandled pitfalls: {unhandled}")

    # ------------------------------------------------------------------
    # 3) The pneumonia-data-efficiency clean state passes every pitfall.
    # ------------------------------------------------------------------
    section("Clean pneumonia state: all 12 pitfalls pass")
    pf = run_pitfalls(PNEUMONIA_CLEAN_STATE, domain="medical_imaging")
    pf_by = {v.check_id: v for v in pf}
    for c in checks:
        v = pf_by.get(c.id)
        expect(v is not None, f"verdict emitted for '{c.id}'")
        expect(v.status == "pass",
               f"'{c.id}' is pass on clean state "
               f"(got status={v.status}, detail={v.detail!r})")

    # ------------------------------------------------------------------
    # 4) Targeted mutations: each Phase 7 pitfall fires on its trigger.
    # ------------------------------------------------------------------
    section("Targeted mutations: each Phase 7 pitfall fires correctly")

    # 4a. Single seed -> multi_seed_reporting fail
    s = {**PNEUMONIA_CLEAN_STATE, "seeds": [0]}
    v = next(x for x in run_pitfalls(s, domain="medical_imaging")
             if x.check_id == "multi_seed_reporting")
    expect(v.status == "fail" and "1" in v.detail,
           f"single-seed run -> multi_seed_reporting fail (got {v.status})")

    # 4b. >= 3 seeds but no variance reported -> fail
    s = {k: v for k, v in PNEUMONIA_CLEAN_STATE.items()
         if k != "report_seed_variance"}
    v = next(x for x in run_pitfalls(s, domain="medical_imaging")
             if x.check_id == "multi_seed_reporting")
    expect(v.status == "fail" and "variance" in v.detail,
           f"3 seeds without variance -> fail (got {v.status})")

    # 4c. Tuning on test split -> hyperparameter_tuning_split fail
    s = {**PNEUMONIA_CLEAN_STATE, "hyperparameter_tuning_split": "test"}
    v = next(x for x in run_pitfalls(s, domain="medical_imaging")
             if x.check_id == "hyperparameter_tuning_split")
    expect(v.status == "fail",
           f"tuning on test split -> fail (got {v.status})")

    # 4d. Weak labels, undisclosed -> needs_human
    s = {k: v for k, v in PNEUMONIA_CLEAN_STATE.items()
         if k != "weak_label_limitation_disclosed"}
    v = next(x for x in run_pitfalls(s, domain="medical_imaging")
             if x.check_id == "weak_label_provenance_disclosed")
    expect(v.status == "needs_human",
           f"weak labels undisclosed -> needs_human (got {v.status})")
    expect("NIH-ChestX-ray14" in v.evidence["weak_datasets"],
           "weak_datasets evidence names NIH")

    # 4e. Mixed views without handling -> view_projection fail (warn-sev)
    s = {k: v for k, v in PNEUMONIA_CLEAN_STATE.items()
         if k != "view_projection_stratified"}
    v = next(x for x in run_pitfalls(s, domain="medical_imaging")
             if x.check_id == "view_projection_documented")
    expect(v.status == "fail" and v.severity == "warn",
           f"mixed views unhandled -> warn-fail (got {v.status}/{v.severity})")

    # 4f. Comparison claims without CIs -> needs_human
    s = {k: v for k, v in PNEUMONIA_CLEAN_STATE.items()
         if k != "confidence_intervals_present"}
    v = next(x for x in run_pitfalls(s, domain="medical_imaging")
             if x.check_id == "confidence_intervals_reported")
    expect(v.status == "needs_human",
           f"comparison without CIs -> needs_human (got {v.status})")

    # ------------------------------------------------------------------
    # 5) Orchestrator outcome aggregation honours the new severities.
    # ------------------------------------------------------------------
    section("Orchestrator: aggregated outcome reflects new severities")
    rep_clean = run_all(PNEUMONIA_CLEAN_STATE)
    expect(rep_clean.outcome == "clean",
           f"clean pneumonia state -> outcome=clean "
           f"(got {rep_clean.outcome}: {rep_clean.summary})")

    # A 'fail'-severity Phase 7 mutation must escalate to block.
    rep_block = run_all({**PNEUMONIA_CLEAN_STATE,
                        "hyperparameter_tuning_split": "test"})
    expect(rep_block.outcome == "block",
           f"tuning on test -> outcome=block (got {rep_block.outcome})")
    block_ids = {v.check_id for v in rep_block.by_status("fail")}
    expect("hyperparameter_tuning_split" in block_ids,
           "hyperparameter_tuning_split present in fail set")

    # A 'needs_human'-severity mutation escalates to needs_human (not block).
    rep_human = run_all({k: v for k, v in PNEUMONIA_CLEAN_STATE.items()
                         if k != "confidence_intervals_present"})
    expect(rep_human.outcome == "needs_human",
           f"missing CIs -> outcome=needs_human (got {rep_human.outcome})")

    # A 'warn'-severity mutation must NOT escalate.
    rep_warn = run_all({k: v for k, v in PNEUMONIA_CLEAN_STATE.items()
                        if k != "view_projection_stratified"})
    expect(rep_warn.outcome == "clean",
           f"warn-severity view-projection fail does not escalate "
           f"(got {rep_warn.outcome}: {rep_warn.summary})")
    warn_v = next(v for v in rep_warn.verdicts
                  if v.check_id == "view_projection_documented")
    expect(warn_v.status == "fail" and warn_v.severity == "warn",
           "warn-severity verdict is recorded with status=fail, severity=warn")

    # ------------------------------------------------------------------
    # 6) Spend invariant.
    # ------------------------------------------------------------------
    section("Spend invariant: zero spend (Phase 7 is offline)")
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

    print("\n*** All Phase 7 smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
