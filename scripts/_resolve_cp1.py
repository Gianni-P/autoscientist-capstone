"""Approve checkpoint cp_fadf92b9cc1743cf965466e79731ab34 and resume the run."""
from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.runtime.runner import resume_run
from autoscientist.state.db import open_db

CP_ID = "cp_fadf92b9cc1743cf965466e79731ab34"
RUN_ID = "run_2f9ec1d3172642ddbff80740757283be"

cfg = load_config()
conn = open_db(cfg.db_path())
try:
    rec = manager.resolve(
        conn,
        checkpoint_id=CP_ID,
        decision=manager.DECISION_APPROVE,
    )
    conn.commit()
    print(f"resolved: {CP_ID} -> status={rec.status}")
finally:
    conn.close()

print(f"resuming run {RUN_ID} ...")
resumed = resume_run(RUN_ID)
print(f"resumed run_id: {resumed}")
