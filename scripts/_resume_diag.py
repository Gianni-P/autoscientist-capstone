"""Read-only probe: what state is the math693a run in after the reject?

Run from WSL only:  .venv/bin/python scripts/_resume_diag.py
Opens the DB read-only (mode=ro) so it can't perturb WAL. Safe to delete.
"""
import json
import sqlite3

DB = "/home/gdp/autoscientist/autoscientist.db"
PROJ = "math693a-limited-descent"

c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
c.row_factory = sqlite3.Row

print("=== recent runs for project ===")
runs = c.execute(
    "SELECT run_id, status, note, started_at, ended_at FROM runs "
    "WHERE project_id = ? ORDER BY started_at DESC LIMIT 6",
    (PROJ,),
).fetchall()
for r in runs:
    print(f"- {r['run_id']}  status={r['status']}  ended={r['ended_at']}  note={r['note']}")

if not runs:
    print("NO RUNS for project", PROJ)
    raise SystemExit

rid = runs[0]["run_id"]
print(f"\n=== checkpoints for most-recent run {rid} ===")
for r in c.execute(
    "SELECT checkpoint_id, stage, status, created_at, resolved_at, operator_input, payload "
    "FROM checkpoints WHERE run_id = ? ORDER BY created_at ASC",
    (rid,),
):
    pl = json.loads(r["payload"]) if r["payload"] else {}
    op = json.loads(r["operator_input"]) if r["operator_input"] else None
    print(
        f"- {r['checkpoint_id']} stage={r['stage']} status={r['status']} "
        f"from={pl.get('from_agent')} to={pl.get('to_agent')} "
        f"decision={(op or {}).get('decision')} extra={pl.get('extra')} "
        f"instr={(op or {}).get('instructions')}"
    )

print(f"\n=== run_controls (manual-pause row) for {rid} ===")
rc = c.execute("SELECT * FROM run_controls WHERE run_id = ?", (rid,)).fetchone()
print(dict(rc) if rc else "none")

print("\n=== code_gen INPUT payload (first user msg to code_gen) ===")
m = c.execute(
    "SELECT content, created_at FROM messages "
    "WHERE run_id = ? AND agent_name = 'code_gen' AND role = 'user' "
    "ORDER BY created_at ASC LIMIT 1",
    (rid,),
).fetchone()
if m:
    print("created_at:", m["created_at"], " payload_len:", len(m["content"]))
    print("---- first 1800 chars ----")
    print(m["content"][:1800])
else:
    print("no code_gen user message found")

print("\n=== recent messages (agent / role / model) ===")
for r in c.execute(
    "SELECT agent_name, role, model, created_at FROM messages "
    "WHERE run_id = ? ORDER BY created_at DESC LIMIT 14",
    (rid,),
):
    print(f"- {r['created_at']}  {r['agent_name']:18} {r['role']:9} {r['model']}")

c.close()
