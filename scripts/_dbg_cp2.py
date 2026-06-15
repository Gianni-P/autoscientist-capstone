import sqlite3, json, re
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
cp_id = "cp_59fb099fd9d24dc383727130964af40b"
rid = "run_2f9ec1d3172642ddbff80740757283be"

# Cumulative spend on this run
print("=== run spend so far ===")
for r in conn.execute(
    "SELECT agent_name, COUNT(*) AS n, ROUND(SUM(cost_usd),5) AS spent, "
    "SUM(CASE cache_hit WHEN 1 THEN 1 ELSE 0 END) AS hits "
    "FROM budget_ledger WHERE run_id=? GROUP BY agent_name ORDER BY spent DESC",
    (rid,),
):
    print(f"  {r['agent_name']:15s} ${r['spent']:.5f}  ({r['n']} calls, {r['hits']} cache hits)")
tot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE run_id=? AND cache_hit=0", (rid,)).fetchone()['s']
mtot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE cache_hit=0").fetchone()['s']
print(f"TOTAL THIS RUN: ${tot:.5f}    CUMULATIVE: ${mtot:.5f}")

# CP2 payload
print(f"\n=== checkpoint {cp_id} ===")
row = conn.execute("SELECT stage, payload, status FROM checkpoints WHERE checkpoint_id=?", (cp_id,)).fetchone()
print(f"stage={row['stage']} status={row['status']}")
payload = json.loads(row["payload"])
print(f"from {payload.get('from_agent')} -> {payload.get('to_agent')}")
agent_raw = payload.get("agent_output_raw", "")
parsed = payload.get("parsed", None)
print(f"agent_output_raw length: {len(agent_raw)}")
print(f"parsed is None: {parsed is None}")
if parsed:
    print(f"parsed top-level keys: {list(parsed.keys())}")

# Look at HANDOFF placement
m = re.search(r"HANDOFF:\s*(\w+)\b", agent_raw)
if m:
    pos = m.start()
    print(f"HANDOFF at pos {pos} of {len(agent_raw)} (chars after: {len(agent_raw)-pos})")
    print(f"--- 200 chars before HANDOFF ---")
    print(agent_raw[max(0, pos-200):pos])
    print(f"--- HANDOFF and after (first 400) ---")
    print(agent_raw[pos:pos+400])
else:
    print("NO HANDOFF FOUND")

# Show end of output (truncation check)
print("\n--- last 600 chars ---")
print(agent_raw[-600:])

# Default payload (what goes to code_gen on approval)
print(f"\n--- default_payload (sent to code_gen on approve) length: {len(payload.get('default_payload', ''))} ---")
dp = payload.get("default_payload", "")
print(dp[:1500] + ("..." if len(dp) > 1500 else ""))
