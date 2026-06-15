"""Reject the degenerate CP3 on run_5273a6fe… -> mark that run cancelled.

The CP3 payload was the code_review "(no payload)" complaint caused by the
empty-content forced-handoff bug (now fixed in runner.py /
payload_files.build_code_review_payload_from_sandbox). Discard it so the
re-run starts clean.
"""
from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.runtime.runner import resume_run
from autoscientist.state.db import open_db

CP_ID = "cp_2ba4b7c76f76465ab4371a01a7767372"
RUN_ID = "run_bb0e896323f848ea81961a0d2852962f"

cfg = load_config()
conn = open_db(cfg.db_path())
try:
    rec = manager.resolve(
        conn,
        checkpoint_id=CP_ID,
        decision=manager.DECISION_REJECT,
        instructions="re-running to test the hardened code_gen.md (API-consistency / no phantom imports)",
    )
    conn.commit()
    print(f"rejected: {CP_ID} -> status={rec.status}")
finally:
    conn.close()

resumed = resume_run(RUN_ID)
print(f"resumed (will exit cancelled): {resumed}")
