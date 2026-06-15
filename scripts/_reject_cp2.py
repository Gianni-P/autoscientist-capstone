"""Reject CP2 -> mark run cancelled, then re-launch from lit_review."""
from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.runtime.runner import resume_run
from autoscientist.state.db import open_db

CP_ID = "cp_59fb099fd9d24dc383727130964af40b"
RUN_ID = "run_2f9ec1d3172642ddbff80740757283be"

cfg = load_config()
conn = open_db(cfg.db_path())
try:
    rec = manager.resolve(
        conn,
        checkpoint_id=CP_ID,
        decision=manager.DECISION_REJECT,
        instructions="methodology hit max_tokens cap; rerunning with bumped 16K limit",
    )
    conn.commit()
    print(f"rejected: {CP_ID} -> status={rec.status}")
finally:
    conn.close()

resumed = resume_run(RUN_ID)
print(f"resumed (will exit cancelled): {resumed}")
