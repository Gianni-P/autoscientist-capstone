import sqlite3, json
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
cp_id = "cp_32514b4840d04a3caf5739a132768b7b"
rid = "run_78f82b18daef406481c1d80c6c199550"

# Spend
print("=== spend (this run) ===")
for r in conn.execute(
    "SELECT agent_name, COUNT(*) AS n, ROUND(SUM(cost_usd),5) AS spent, "
    "SUM(CASE cache_hit WHEN 1 THEN 1 ELSE 0 END) AS hits "
    "FROM budget_ledger WHERE run_id=? GROUP BY agent_name ORDER BY spent DESC",
    (rid,),
):
    print(f"  {r['agent_name']:15s} ${r['spent']:.5f}  ({r['n']} calls, {r['hits']} cache hits)")
trun = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE run_id=? AND cache_hit=0", (rid,)).fetchone()['s']
tot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE cache_hit=0").fetchone()['s']
print(f"THIS RUN: ${trun:.5f}   CUMULATIVE MONTHLY: ${tot:.5f}")

row = conn.execute("SELECT payload, status FROM checkpoints WHERE checkpoint_id=?", (cp_id,)).fetchone()
payload = json.loads(row["payload"])
raw = payload["agent_output_raw"]
parsed = payload.get("parsed")
print(f"\n=== CP2 status={row['status']} ===")
print(f"agent_output_raw length: {len(raw)} chars")
print(f"parsed is None: {parsed is None}")
if parsed:
    plan_keys = list(parsed.get('plan', parsed).keys()) if isinstance(parsed.get('plan'), dict) else list(parsed.keys())
    print(f"parsed plan keys: {plan_keys}")

# Find HANDOFF
import re
m = re.search(r"HANDOFF:\s*(\w+)\b", raw)
if m:
    pos = m.start()
    print(f"HANDOFF at pos {pos}/{len(raw)}; chars after HANDOFF: {len(raw)-pos}")
else:
    print("NO HANDOFF FOUND")

# Default payload length & ending
dp = payload.get("default_payload", "")
print(f"default_payload length: {len(dp)}")
print("--- default_payload last 400 chars ---")
print(dp[-400:])

# Try parse plan in first JSON
start = raw.find("{")
depth = 0; end = -1
for i in range(start, len(raw)):
    c = raw[i]
    if c == "{": depth += 1
    elif c == "}":
        depth -= 1
        if depth == 0: end = i + 1; break
first_blob = raw[start:end]
print(f"\n=== first JSON block: chars {start}-{end} (len={end-start}) ===")
try:
    j = json.loads(first_blob)
    plan = j.get("plan", j)
    print(f"plan keys: {list(plan.keys())}")
    print()
    print("Q:", plan.get("research_question","?")[:300])
    print(f"hypotheses: {len(plan.get('hypotheses', []))}")
    print(f"datasets: {[d.get('name') for d in plan.get('datasets', [])]}")
    print(f"baselines: {len(plan.get('baselines', []))}")
    print(f"metrics: {[m.get('name') for m in plan.get('metrics', [])]}")
    print(f"experiments: {[e.get('id') for e in plan.get('experiments', [])]}")
    print(f"pitfall_acks: {len(plan.get('pitfall_acks', []))}")
    sc = plan.get("stop_conditions", {})
    print(f"stop_conditions keys: {list(sc.keys())}")
    print(f"  early_abort: {sc.get('early_abort','?')[:300]}")
except Exception as e:
    print(f"parse err: {e}")
