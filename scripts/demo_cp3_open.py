"""Demonstrate that CP3 (preliminary review) opens and routes forward.

2026-05-31 audit, item 3 — the CP3 half. CP3 is gated on code_review handing
off to results_validator (manager.stage_for_agent("code_review", handoff_to=...)
returns stage 3 only when the target is NOT code_gen). code_review HAS run
before (it is not one of the four never-run agents), so this is a cheap, focused
proof that the CP3 gate opens on a forward (pass) verdict and routes to
results_validator. It then REJECTS the checkpoint so the back half is not
re-run (that is covered by thin_slice_backhalf.py).

Run inside WSL with the API key loaded:
    ./.venv/bin/python scripts/demo_cp3_open.py
"""

from __future__ import annotations

import json
import sys

from autoscientist.checkpoints import manager as checkpoints
from autoscientist.runtime import runner
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db

PROJECT_ID = "pneumonia-backhalf-slice"

# A clean, obviously-passing review payload: a small patient-level split module
# plus its determinism/leakage test. Nothing here warrants a revise/block, so
# code_review should verdict "pass" and hand off to results_validator → CP3.
CLEAN_PAYLOAD = json.dumps({
    "src_files": [
        {
            "path": "src/data.py",
            "content": (
                "def patient_level_split(patient_ids, *, test_frac, seed):\n"
                "    # hash-based, deterministic, patient-level (no image-level leakage)\n"
                "    import hashlib\n"
                "    uniq = sorted(set(patient_ids)); train=[]; test=[]\n"
                "    for pid in uniq:\n"
                "        b = int(hashlib.sha256(f'{seed}:{pid}'.encode()).hexdigest()[:8],16)/0xFFFFFFFF\n"
                "        (test if b < test_frac else train).append(pid)\n"
                "    return train, test\n"
            ),
        }
    ],
    "test_files": [
        {
            "path": "tests/test_patient_split.py",
            "content": (
                "from src.data import patient_level_split\n"
                "def test_disjoint():\n"
                "    tr, te = patient_level_split([f'p{i}' for i in range(200)], test_frac=0.2, seed=42)\n"
                "    assert not (set(tr) & set(te))\n"
            ),
        }
    ],
    "run_cmd_src": "python -c 'import src.data'",
    "run_cmd_tests": "pytest tests/test_patient_split.py",
    "note": "Tests pass; patient-level split is deterministic and leakage-free. Recommend proceeding.",
})


def main() -> int:
    cfg = load_config(reload=True)
    db_path = str(cfg.db_path())

    run_id = runner.run(
        starting_agent="code_review",
        project_id=PROJECT_ID,
        initial_payload=CLEAN_PAYLOAD,
        enable_checkpoints=True,
        cfg=cfg,
    )
    print(f"run_id: {run_id}")

    conn = open_db(db_path)
    try:
        row = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
        status = row["status"] if row else "unknown"
        cp = checkpoints.latest_checkpoint(conn, run_id)
        print(f"run status: {status}")
        if cp is None:
            print("NO checkpoint opened — code_review did not gate (unexpected).")
            return 1
        print(f"checkpoint: CP{cp.stage} ({cp.stage_name}) status={cp.status} "
              f"from={cp.from_agent} -> to={cp.to_agent}")
        ok = cp.stage == 3 and cp.to_agent == "results_validator"
        print("CP3 opened and routed to results_validator: " + ("YES" if ok else "NO"))
        if cp.status == "pending":
            # Reject so the back half is not re-run on resume (no extra spend).
            checkpoints.resolve(conn, checkpoint_id=cp.checkpoint_id, decision="reject",
                                instructions="demo: CP3 open confirmed; rejecting to avoid re-running back half")
            conn.commit()
            print("CP3 rejected (clean close; back half not re-run).")
        return 0 if ok else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
