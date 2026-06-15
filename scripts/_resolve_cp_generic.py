"""Approve and resume from a checkpoint.

Usage: python scripts/_resolve_cp_generic.py <run_id> <checkpoint_id> [decision]
decision defaults to 'approve'.
"""
import sys
from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.runtime.runner import resume_run
from autoscientist.state.db import open_db

if len(sys.argv) < 3:
    print("usage: <run_id> <checkpoint_id> [approve|reject|modify]")
    sys.exit(2)
RUN_ID = sys.argv[1]
CP_ID = sys.argv[2]
decision_str = sys.argv[3] if len(sys.argv) > 3 else "approve"
decision_map = {
    "approve": manager.DECISION_APPROVE,
    "reject": manager.DECISION_REJECT,
    "modify": manager.DECISION_MODIFY,
}
if decision_str not in decision_map:
    print(f"invalid decision: {decision_str}")
    sys.exit(2)

cfg = load_config()
conn = open_db(cfg.db_path())
try:
    rec = manager.resolve(conn, checkpoint_id=CP_ID, decision=decision_map[decision_str])
    conn.commit()
    print(f"resolved {CP_ID} -> {rec.status}")
finally:
    conn.close()

resumed = resume_run(RUN_ID)
print(f"resumed: {resumed}")
