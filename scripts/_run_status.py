"""Print run status + checkpoints + last messages for a run (read-only)."""

import sys

from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db

run_id = sys.argv[1]
cfg = load_config()
conn = open_db(cfg.db_path())
try:
    r = conn.execute("SELECT status, note FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if r is None:
        print("run not found:", run_id)
        sys.exit(1)
    print(f"RUN {run_id}  status={r['status']}  note={r['note']}")
    for cp in manager.list_for_run(conn, run_id):
        print(f"  CP {cp.checkpoint_id} stage={cp.stage} status={cp.status} "
              f"{cp.from_agent}->{cp.to_agent}")
    rows = conn.execute(
        "SELECT agent_name, role, substr(replace(content, char(10), ' '), 1, 160) c "
        "FROM messages WHERE run_id=? ORDER BY rowid DESC LIMIT 6", (run_id,),
    ).fetchall()
    print("  -- last messages --")
    for m in reversed(rows):
        print(f"  [{m['agent_name']}/{m['role']}] {m['c']}")
finally:
    conn.close()
