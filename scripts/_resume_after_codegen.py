"""Flip the run from 'completed' back to 'paused' (CP2 stayed approved),
then resume so code_gen re-runs with whatever model alias is now in models.toml."""
from autoscientist.runtime.config import load_config
from autoscientist.runtime.runner import resume_run
from autoscientist.state.db import open_db

RUN_ID = "run_78f82b18daef406481c1d80c6c199550"

cfg = load_config(reload=True)
conn = open_db(cfg.db_path())
try:
    row = conn.execute("SELECT status FROM runs WHERE run_id=?", (RUN_ID,)).fetchone()
    print(f"current run status: {row['status']}")
    conn.execute("UPDATE runs SET status='paused', ended_at=NULL WHERE run_id=?", (RUN_ID,))
    conn.commit()
    print("flipped to paused")
finally:
    conn.close()

resumed = resume_run(RUN_ID)
print(f"resumed: {resumed}")
