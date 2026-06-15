import sqlite3, datetime as dt
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
rid = "run_78f82b18daef406481c1d80c6c199550"

# Find the latest run.resume line in the log and compute time since
log = "/home/gdp/autoscientist/runs/run_78f82b18daef406481c1d80c6c199550/logs/run.jsonl"
import json, os
last_resume_ts = None
with open(log) as f:
    for line in f:
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("event") == "run.resume":
            last_resume_ts = j.get("timestamp")
print(f"Last resume: {last_resume_ts}")

if last_resume_ts:
    # Spend since last resume
    print("\n=== spend since latest run.resume ===")
    for r in conn.execute(
        "SELECT agent_name, COUNT(*) AS n, ROUND(SUM(cost_usd),5) AS spent "
        "FROM budget_ledger WHERE run_id=? AND created_at>? AND cache_hit=0 "
        "GROUP BY agent_name ORDER BY spent DESC",
        (rid, last_resume_ts),
    ):
        print(f"  {r['agent_name']:18s} ${r['spent']:.5f} ({r['n']} calls)")
    trun = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE run_id=? AND created_at>? AND cache_hit=0",
        (rid, last_resume_ts),
    ).fetchone()['s']
    tot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE cache_hit=0").fetchone()['s']
    print(f"  TOTAL since restart: ${trun:.5f}")
    print(f"  CUMULATIVE MONTHLY:  ${tot:.5f}")

    # Count agent_done events since last resume
    print("\n=== recent activity (last 15 events of any kind) ===")
    with open(log) as f:
        lines = f.readlines()
    started = False
    events = []
    for line in lines:
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("event") == "run.resume" and j.get("timestamp") == last_resume_ts:
            started = True
            continue
        if started:
            ev = j.get("event", "")
            if ev in ("ollama.complete.done", "claude.complete.done", "run.agent_done",
                      "tools.dispatch.ok", "run.checkpoint_opened", "tool_loop_max_rounds_reached",
                      "run.no_handoff_terminal", "run.end", "execute.start"):
                ts = j.get("timestamp", "")[11:19]
                agent = j.get("agent") or j.get("name") or ""
                model = j.get("model", "")
                ct = j.get("completion_tokens") or ""
                cc = j.get("content_chars")
                tc = j.get("n_tool_calls")
                events.append(f"  {ts} {ev:30s} agent={agent:15s} ct={ct} cc={cc} n_tc={tc}")
    for e in events[-15:]:
        print(e)
    print(f"\nTotal events since restart: {len(events)}")

    # Wall-clock since restart
    try:
        last_resume_dt = dt.datetime.fromisoformat(last_resume_ts.replace("Z","+00:00"))
        elapsed = (dt.datetime.now(dt.timezone.utc) - last_resume_dt).total_seconds()
        print(f"\nElapsed since restart: {int(elapsed)}s = {elapsed/60:.1f} min")
    except Exception as e:
        print(f"could not parse timestamp: {e}")
