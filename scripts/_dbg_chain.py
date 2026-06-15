import sqlite3, re
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
rid = "run_78f82b18daef406481c1d80c6c199550"

# Spend
print("=== run spend ===")
for r in conn.execute(
    "SELECT agent_name, COUNT(*) AS n, ROUND(SUM(cost_usd),5) AS spent, "
    "SUM(CASE cache_hit WHEN 1 THEN 1 ELSE 0 END) AS hits "
    "FROM budget_ledger WHERE run_id=? GROUP BY agent_name ORDER BY spent DESC",
    (rid,),
):
    print(f"  {r['agent_name']:18s} ${r['spent']:.5f}  ({r['n']} calls, {r['hits']} cache hits)")
trun = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE run_id=? AND cache_hit=0", (rid,)).fetchone()['s']
tot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE cache_hit=0").fetchone()['s']
print(f"THIS RUN: ${trun:.5f}   CUMULATIVE: ${tot:.5f}")

# Find each agent's final assistant message in chronological order
print("\n=== agent transitions (final assistant per agent visit, in DB order) ===")
prev_agent = None
visit = []
for r in conn.execute(
    "SELECT rowid, agent_name, role, length(content) AS clen, content, completion_tokens FROM messages "
    "WHERE run_id=? AND role='assistant' ORDER BY rowid",
    (rid,),
):
    if r['agent_name'] != prev_agent:
        if visit:
            last = visit[-1]
            handoff_match = re.search(r"HANDOFF:\s*(\w+)", last['content'] or '')
            handoff = handoff_match.group(1) if handoff_match else "NONE"
            print(f"  {prev_agent:20s} -> {handoff}  (final cc={last['clen']} ct={last['completion_tokens']}, {len(visit)} assistant msgs)")
        prev_agent = r['agent_name']
        visit = []
    visit.append(r)
# Print last
if visit:
    last = visit[-1]
    handoff_match = re.search(r"HANDOFF:\s*(\w+)", last['content'] or '')
    handoff = handoff_match.group(1) if handoff_match else "NONE"
    print(f"  {prev_agent:20s} -> {handoff}  (final cc={last['clen']} ct={last['completion_tokens']}, {len(visit)} assistant msgs)")

# Show the last code_gen response (49857 chars)
print("\n=== last code_gen assistant message (chars 0-1000) ===")
final = conn.execute(
    "SELECT content, completion_tokens FROM messages WHERE run_id=? AND agent_name='code_gen' AND role='assistant' "
    "AND length(content) > 5000 ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
if final:
    print(f"ct={final['completion_tokens']}")
    print(final['content'][:1000])
    print("\n... last 600 chars ...")
    print(final['content'][-600:])
