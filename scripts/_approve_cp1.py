"""Approve checkpoint and resume the pipeline."""
import sys
sys.path.insert(0, '/home/gdp/autoscientist/src')

from autoscientist.state.db import open_db

RUN_ID = "run_73e0f6c14e374cb1b0e92dc44421f688"
CP_ID = "cp_f8fc892688e243dda7c53910f46580f5"

conn = open_db('/home/gdp/autoscientist/autoscientist.db')

# Approve checkpoint
conn.execute(
    "UPDATE checkpoints SET status = 'approved', resolved_at = datetime('now') "
    "WHERE checkpoint_id = ?",
    (CP_ID,)
)
# Update run status from paused -> running
conn.execute(
    "UPDATE runs SET status = 'running' WHERE run_id = ?",
    (RUN_ID,)
)
conn.commit()
print(f"Approved {CP_ID}")
print(f"Run {RUN_ID} status -> running")

# Verify
cp = conn.execute(
    "SELECT status FROM checkpoints WHERE checkpoint_id = ?", (CP_ID,)
).fetchone()
print(f"Checkpoint status: {cp['status']}")

run = conn.execute(
    "SELECT status FROM runs WHERE run_id = ?", (RUN_ID,)
).fetchone()
print(f"Run status: {run['status']}")
conn.close()
