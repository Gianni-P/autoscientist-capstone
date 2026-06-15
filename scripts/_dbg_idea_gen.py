import sqlite3, re, json
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
rid = "run_eb2aad39a7f04c2cb33cc0fe58880cce"
row = conn.execute(
    "SELECT content FROM messages "
    "WHERE run_id=? AND agent_name='idea_gen' AND role='assistant' "
    "ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
content = row["content"]

# Try to count complete idea blocks
idea_count = len(re.findall(r'"title":', content))
print(f"approx 'title:' occurrences in idea_gen output: {idea_count}")
# Spend tally
total_cost = 0.0
for r in conn.execute(
    "SELECT agent_name, COUNT(*) AS n, ROUND(SUM(cost_usd),5) AS spent "
    "FROM budget_ledger WHERE run_id=? AND cache_hit=0 GROUP BY agent_name",
    (rid,),
):
    print(f"  {r['agent_name']:15s} ${r['spent']:.5f} ({r['n']} calls)")
    total_cost += r['spent'] or 0
print(f"TOTAL: ${total_cost:.5f}")
