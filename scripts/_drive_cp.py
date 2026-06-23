"""Resolve a checkpoint (optionally with per-leg model_overrides, e.g. for
Opus-orchestrator mode) and resume the run. Operational helper for headless runs.

Usage:
  python scripts/_drive_cp.py <run_id> <cp_id> [approve|reject|modify] [overrides_json]

Example — run the next leg's code_gen/test_gen as Opus-orchestrator:
  python scripts/_drive_cp.py run_x cp_y approve '{"code_gen":"orchestrator","test_gen":"orchestrator"}'
"""

import json
import sys

from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.runtime.runner import resume_run
from autoscientist.state.db import open_db

if len(sys.argv) < 3:
    print("usage: <run_id> <cp_id> [approve|reject|modify] [overrides_json]")
    sys.exit(2)

run_id = sys.argv[1]
cp_id = sys.argv[2]
decision_str = sys.argv[3] if len(sys.argv) > 3 else "approve"
overrides = None
instructions = None
# Remaining args (any order): "orch:a,b" -> orchestrator mode; "instr:<path>" ->
# read operator instructions from a file (quote-safe); a leading-"{" JSON string
# -> explicit model_overrides map.
for _a in sys.argv[4:]:
    _a = _a.strip()
    if not _a:
        continue
    if _a.startswith("orch:"):
        _agents = [x.strip() for x in _a[len("orch:"):].split(",") if x.strip()]
        overrides = {x: "orchestrator" for x in _agents}
    elif _a.startswith("instr:"):
        with open(_a[len("instr:"):], encoding="utf-8") as _f:
            instructions = _f.read()
    else:
        overrides = json.loads(_a)

decision_map = {
    "approve": manager.DECISION_APPROVE,
    "reject": manager.DECISION_REJECT,
    "modify": manager.DECISION_MODIFY,
    "rerun": manager.DECISION_RERUN,
}
if decision_str not in decision_map:
    print(f"invalid decision: {decision_str}")
    sys.exit(2)

cfg = load_config()
conn = open_db(cfg.db_path())
try:
    rec = manager.resolve(
        conn, checkpoint_id=cp_id, decision=decision_map[decision_str],
        instructions=instructions, model_overrides=overrides,
    )
    conn.commit()
    print(f"resolved {cp_id} -> {rec.status}  overrides={overrides}  "
          f"instructions={'yes' if instructions else 'no'}", flush=True)
finally:
    conn.close()

if decision_str == "reject":
    print("rejected — not resuming")
    sys.exit(0)

print(f"resuming {run_id} ...", flush=True)
res = resume_run(run_id)
print(f"resume returned: {res}")
