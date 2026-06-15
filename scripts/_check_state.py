"""Read-only: what did the regen run produce? Run from WSL only."""
import json
import sqlite3

DB = "/home/gdp/autoscientist/autoscientist.db"
PROJ = "math693a-limited-descent"

c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
c.row_factory = sqlite3.Row

print("=== all runs for project (newest first) ===")
runs = c.execute(
    "SELECT run_id, status, note, started_at, ended_at FROM runs "
    "WHERE project_id = ? ORDER BY started_at DESC LIMIT 8",
    (PROJ,),
).fetchall()
for r in runs:
    print(f"- {r['run_id']}\n    status={r['status']} started={r['started_at']} "
          f"ended={r['ended_at']}\n    note={r['note']}")

print("\n=== ALL pending checkpoints (any run) ===")
pend = c.execute(
    "SELECT checkpoint_id, run_id, stage, created_at, payload FROM checkpoints "
    "WHERE status = 'pending' ORDER BY created_at DESC",
).fetchall()
if not pend:
    print("none pending")
for r in pend:
    pl = json.loads(r["payload"]) if r["payload"] else {}
    print(f"- {r['checkpoint_id']} run={r['run_id']} stage={r['stage']} "
          f"from={pl.get('from_agent')} to={pl.get('to_agent')} at={r['created_at']}")

if runs:
    rid = runs[0]["run_id"]
    print(f"\n=== checkpoints for newest run {rid} ===")
    cps = c.execute(
        "SELECT checkpoint_id, stage, status, created_at FROM checkpoints "
        "WHERE run_id = ? ORDER BY created_at ASC", (rid,)).fetchall()
    if not cps:
        print("no checkpoints for this run")
    for r in cps:
        print(f"- {r['checkpoint_id']} stage={r['stage']} status={r['status']} at={r['created_at']}")

    print(f"\n=== last 16 messages for newest run {rid} (agent/role/model) ===")
    for r in c.execute(
        "SELECT agent_name, role, model, content, created_at FROM messages "
        "WHERE run_id = ? ORDER BY created_at DESC LIMIT 16", (rid,)):
        snippet = (r["content"] or "").replace("\n", " ")[:80]
        print(f"- {r['created_at']} {r['agent_name']:16} {r['role']:9} "
              f"{str(r['model']):18} | {snippet}")

c.close()
