"""Approve CP2 (methodology approval) to proceed to code_gen."""
import sys
sys.path.insert(0, '/home/gdp/autoscientist/src')
from autoscientist.state.db import open_db

RUN_ID = "run_73e0f6c14e374cb1b0e92dc44421f688"
CP_ID = "cp_693315f7013147b6b23ac2308d32c1f8"

conn = open_db('/home/gdp/autoscientist/autoscientist.db')
conn.execute(
    "UPDATE checkpoints SET status = 'approved', resolved_at = datetime('now') "
    "WHERE checkpoint_id = ?",
    (CP_ID,)
)
conn.commit()

cp = conn.execute("SELECT status FROM checkpoints WHERE checkpoint_id = ?", (CP_ID,)).fetchone()
print(f"CP2 status: {cp['status']}")
run = conn.execute("SELECT status FROM runs WHERE run_id = ?", (RUN_ID,)).fetchone()
print(f"Run status: {run['status']}")
conn.close()
print(f"Approved {CP_ID} — ready to resume to code_gen (Qwen 27B)")
