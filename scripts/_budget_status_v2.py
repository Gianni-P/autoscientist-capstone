"""Quick lifetime + per-agent spend summary for operator pre-launch sanity check."""
import sqlite3

conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
total = conn.execute(
    "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger WHERE cache_hit=0"
).fetchone()["s"]
print(f"real spend lifetime: ${float(total):.4f}")
print("per-agent (real $ only):")
for r in conn.execute(
    "SELECT agent_name, ROUND(SUM(cost_usd), 4) AS spent, COUNT(*) AS n "
    "FROM budget_ledger WHERE cache_hit=0 GROUP BY agent_name "
    "ORDER BY spent DESC"
):
    spent = float(r[1] or 0.0)
    print(f"  {r[0]:24s} ${spent:7.4f}  ({r[2]} calls)")

# Current month
import datetime
month = datetime.datetime.utcnow().strftime("%Y-%m")
month_total = conn.execute(
    "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger "
    "WHERE cache_hit=0 AND month_key = ?",
    (month,),
).fetchone()["s"]
print(f"\ncurrent month ({month}) spend: ${float(month_total):.4f} / $150 cap")
