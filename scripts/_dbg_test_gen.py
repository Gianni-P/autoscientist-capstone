import sqlite3, re, json
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
rid = "run_78f82b18daef406481c1d80c6c199550"

# Spend on this run
print("=== run spend ===")
for r in conn.execute(
    "SELECT agent_name, COUNT(*) AS n, ROUND(SUM(cost_usd),5) AS spent, "
    "SUM(CASE cache_hit WHEN 1 THEN 1 ELSE 0 END) AS hits "
    "FROM budget_ledger WHERE run_id=? GROUP BY agent_name ORDER BY spent DESC",
    (rid,),
):
    print(f"  {r['agent_name']:15s} ${r['spent']:.5f}  ({r['n']} calls, {r['hits']} cache hits)")
trun = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE run_id=? AND cache_hit=0", (rid,)).fetchone()['s']
tot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE cache_hit=0").fetchone()['s']
print(f"THIS RUN: ${trun:.5f}   CUMULATIVE: ${tot:.5f}")

# Look at code_gen's final assistant content
print("\n=== code_gen's final HANDOFF message ===")
cg_last = conn.execute(
    "SELECT rowid, content, completion_tokens FROM messages "
    "WHERE run_id=? AND agent_name='code_gen' AND role='assistant' AND length(content) > 100 "
    "ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
if cg_last:
    text = cg_last["content"]
    print(f"rowid={cg_last['rowid']} len={len(text)} ct={cg_last['completion_tokens']}")
    m = re.search(r"HANDOFF:\s*(\w+)", text)
    print(f"HANDOFF: {m.group() if m else 'NONE'}")
    print(f"Ends with: {text[-200:]!r}")
    print(f"First 600 chars:\n{text[:600]}")

# Look at test_gen final
print("\n=== test_gen's last assistant message ===")
tg_last = conn.execute(
    "SELECT rowid, content, completion_tokens FROM messages "
    "WHERE run_id=? AND agent_name='test_gen' AND role='assistant' "
    "ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
if tg_last:
    text = tg_last["content"]
    print(f"rowid={tg_last['rowid']} len={len(text)} ct={tg_last['completion_tokens']}")
    m = re.search(r"HANDOFF:\s*(\w+)", text)
    print(f"HANDOFF: {m.group() if m else 'NONE'}")
    print(f"Ends with: {text[-400:]!r}")

# What files exist in the sandbox now?
print("\n=== sandbox files (recursive, first 30) ===")
import os
sb = "/home/gdp/autoscientist/projects/pneumonia-data-efficiency/sandbox"
count = 0
for root, dirs, files in os.walk(sb):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"  {size:>10d} {os.path.relpath(path, sb)}")
        count += 1
        if count >= 30: break
    if count >= 30: break
print(f"(showing {count} files)")
