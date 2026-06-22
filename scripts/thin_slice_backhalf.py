"""Thin vertical slice: drive CP4 -> CP5 through the four never-run agents.

2026-05-31 audit, item 3. results_validator / paper_writer / peer_reviewer /
repo_publisher have ZERO messages in the production DB — the autonomous chain
never reached them. This script feeds results_validator a CANNED payload built
from the real pneumonia E0/E1/E2 numbers and auto-resolves the checkpoints, so
the back half of the pipeline executes end-to-end for a few dollars and any
integration bugs in those agents surface.

Design choices (see the audit report):
  * Starts at results_validator, not code_review: code_review's pass-handoff
    forwards only code files, not results, so it can't feed results_validator a
    {plan, results} payload without actually running experiments. CP3 (gated on
    code_review, which HAS run before) is demonstrated separately by
    scripts/demo_cp3_open.py.
  * Non-destructive: a side-by-side project_id (pneumonia-backhalf-slice) with
    its own sandbox. Shares no state with pneumonia-data-efficiency.
  * The checkpoint resolver approves the agent's REAL handoff payload when the
    routing is correct (testing real payload threading), and only force-corrects
    + logs a finding when an agent misroutes.
  * A per-project soft cap (config.toml) + the global monthly cap bound spend.

Run inside WSL with the API key loaded:
    set -a; source .env; set +a
    ./.venv/bin/python scripts/thin_slice_backhalf.py
"""

from __future__ import annotations

import json
import sys

from autoscientist.checkpoints import manager as checkpoints
from autoscientist.runtime import runner
from autoscientist.runtime.budget import project_spent
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db

PROJECT_ID = "pneumonia-backhalf-slice"
# Stage 4 (results_validator) now advances to figure_gen (which renders the
# paper's figures and then hands to paper_writer); stage 5 to repo_publisher.
FORWARD_TARGET = {4: "figure_gen", 5: "repo_publisher"}
MAX_RESOLVES = 8
SPEND_ABORT_USD = 12.0


def canned_payload() -> str:
    """A coherent, advanceable negative-result study built from the REAL numbers.

    Hypotheses are phrased so the observed results CONFIRM their predicted
    direction (no sign flip => results_validator should 'advance'): in-domain
    AUROC rises with N, and external transfer collapses to near-chance under
    domain shift. The baseline (E0) reproduced within tolerance.
    """
    plan = {
        "research_question": (
            "How does training-set size affect cross-institutional generalization "
            "of CNN pneumonia detection (NIH ChestX-ray14 -> PadChest)?"
        ),
        "datasets": {
            "train": "NIH-ChestX-ray14",
            "external_validation": "PadChest",
            "label_provenance": "nlp_derived (both)",
        },
        "design": "ResNet-50, N in {1k,5k,25k,100k}, 3 seeds, patient-level split, "
                  "bootstrap CIs (100 resamples). Metrics: AUROC, Brier, ECE.",
        "hypotheses": [
            {"name": "in_domain_data_efficiency",
             "predicted_direction": "in-domain (NIH) AUROC increases with training size N"},
            {"name": "transfer_degradation_under_domain_shift",
             "predicted_direction": "external (PadChest) AUROC is far below in-domain and "
                                    "near chance (transfer fails)"},
        ],
        "baselines": [{"name": "CheXNet-Rajpurkar2017", "metric": "AUROC-pneumonia",
                       "published_value": 0.768}],
    }
    results = {
        "metrics": [
            {"experiment_id": "E0", "split": "NIH_indomain", "metric": "AUROC",
             "value": 0.764, "ci": [0.733, 0.795], "note": "baseline reproduction, 3 seeds"},
            {"experiment_id": "E1", "N": 1000, "split": "NIH_indomain", "metric": "AUROC",
             "value": 0.55, "ci": [0.49, 0.61]},
            {"experiment_id": "E1", "N": 25000, "split": "NIH_indomain", "metric": "AUROC",
             "value": 0.673, "ci": [0.61, 0.72]},
            {"experiment_id": "E1", "N": 100000, "split": "NIH_indomain", "metric": "AUROC",
             "value": 0.631, "ci": [0.59, 0.66], "note": "slight saturation vs N=25k"},
            {"experiment_id": "E1", "N": 25000, "split": "PadChest_external", "metric": "AUROC",
             "value": 0.452, "ci": [0.41, 0.49]},
            {"experiment_id": "E2", "N": 100000, "split": "PadChest_external", "metric": "AUROC",
             "value": 0.428, "ci": [0.38, 0.47]},
        ],
        "baseline_repro": {"name": "CheXNet-Rajpurkar2017", "target": 0.768,
                           "achieved": 0.764, "in_tolerance": True, "tolerance_abs": 0.06},
        "verify_output": {
            "outcome": "needs_human",
            "summary": "Deterministic harness: baseline reproduced; external AUROCs near "
                       "chance (discrimination_floor=needs_human) — CONSISTENT with the "
                       "transfer-degradation hypothesis; in-domain rises with N then "
                       "saturates at N=100k (non-monotonic, blocks_paper=false).",
            "notable": ["discrimination_floor: external AUROC ~0.43-0.45 (near chance) — expected under domain shift",
                        "in-domain N=100k < N=25k (saturation, not a sign flip vs hypotheses)"],
        },
    }
    return json.dumps({"plan": plan, "results": results})


def _build_forced_payload(stage: int, default_payload: str) -> str:
    """Construct a payload to force progression when an agent misrouted.

    Reuses the upstream agent's handoff payload when it parses, patching the
    validator verdict to 'advance' for CP4 so paper_writer won't refuse.
    """
    try:
        obj = json.loads(default_payload) if default_payload else {}
    except json.JSONDecodeError:
        obj = {}
    if stage == 4:
        vs = obj.get("validator_summary")
        if not isinstance(vs, dict):
            vs = {}
        vs["verdict"] = "advance"
        obj.setdefault("plan", json.loads(canned_payload())["plan"])
        obj.setdefault("results", json.loads(canned_payload())["results"])
        obj["validator_summary"] = vs
    return json.dumps(obj) if obj else (default_payload or "{}")


def _patch_to_agent(conn, checkpoint_id: str, to_agent: str) -> None:
    row = conn.execute(
        "SELECT payload FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
    ).fetchone()
    env = json.loads(row["payload"])
    env["to_agent"] = to_agent
    conn.execute(
        "UPDATE checkpoints SET payload = ? WHERE checkpoint_id = ?",
        (json.dumps(env), checkpoint_id),
    )


def resolve_pending(db_path: str, run_id: str, findings: list[str]) -> bool:
    """Resolve the run's latest pending checkpoint. Returns True if one was resolved."""
    conn = open_db(db_path)
    try:
        cp = checkpoints.latest_checkpoint(conn, run_id)
        if cp is None or cp.status != "pending":
            return False
        want = FORWARD_TARGET.get(cp.stage)
        print(f"  -> CP{cp.stage} ({cp.stage_name}) pending; agent routed to "
              f"'{cp.to_agent}' (want '{want}')")
        if want and cp.to_agent != want:
            findings.append(
                f"CP{cp.stage}: upstream agent routed to '{cp.to_agent}', not '{want}'; "
                f"operator force-corrected routing to exercise the back half."
            )
            _patch_to_agent(conn, cp.checkpoint_id, want)
            modified = _build_forced_payload(cp.stage, cp.default_payload)
            checkpoints.resolve(conn, checkpoint_id=cp.checkpoint_id, decision="modify",
                                instructions="operator: proceed; write up as negative result",
                                modified_payload=modified)
        else:
            checkpoints.resolve(conn, checkpoint_id=cp.checkpoint_id, decision="approve",
                                instructions="operator: approved")
        conn.commit()
        return True
    finally:
        conn.close()


def _run_status(db_path: str, run_id: str) -> str:
    conn = open_db(db_path)
    try:
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row["status"] if row else "unknown"
    finally:
        conn.close()


def report(db_path: str, run_id: str, findings: list[str]) -> None:
    conn = open_db(db_path)
    try:
        print("\n" + "=" * 90)
        print("SLICE REPORT")
        print("=" * 90)
        print(f"run_id: {run_id}  final status: {_run_status(db_path, run_id)}")

        cps = conn.execute(
            "SELECT stage, status FROM checkpoints WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        print("\nCheckpoints opened (this run):")
        for c in cps:
            print(f"  CP{c['stage']}  {c['status']}")

        print("\nMessages per agent (this run) — proves the four never-run agents executed:")
        rows = conn.execute(
            "SELECT agent_name, COUNT(*) n FROM messages WHERE run_id = ? "
            "GROUP BY agent_name ORDER BY n DESC", (run_id,),
        ).fetchall()
        for r in rows:
            print(f"  {r['agent_name']:<20} {r['n']} messages")

        spent = project_spent(conn, PROJECT_ID)
        print(f"\nProject spend (this exercise): ${spent:.4f}")

        if findings:
            print("\nIntegration findings:")
            for f in findings:
                print(f"  - {f}")
    finally:
        conn.close()


def main() -> int:
    cfg = load_config(reload=True)
    db_path = str(cfg.db_path())

    # Pre-flight spend guard.
    conn = open_db(db_path)
    try:
        before = project_spent(conn, PROJECT_ID)
    finally:
        conn.close()
    print(f"Starting thin slice. Project '{PROJECT_ID}' spend before: ${before:.4f}")

    run_id = runner.run(
        starting_agent="results_validator",
        project_id=PROJECT_ID,
        initial_payload=canned_payload(),
        enable_checkpoints=True,
        cfg=cfg,
    )
    print(f"run_id: {run_id}")

    findings: list[str] = []
    for i in range(MAX_RESOLVES):
        status = _run_status(db_path, run_id)
        print(f"[iter {i}] status={status}")
        if status != "paused":
            break
        conn = open_db(db_path)
        try:
            spent = project_spent(conn, PROJECT_ID)
        finally:
            conn.close()
        if spent > SPEND_ABORT_USD:
            findings.append(f"ABORTED: project spend ${spent:.2f} exceeded ${SPEND_ABORT_USD}")
            break
        if not resolve_pending(db_path, run_id, findings):
            break
        runner.resume_run(run_id, cfg=cfg)

    report(db_path, run_id, findings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
