import sqlite3, os
conn = sqlite3.connect(os.environ.get('AUTOSCIENTIST_DB_PATH', 'autoscientist.db'))
conn.row_factory = sqlite3.Row
print('--- per-agent monthly spend ---')
for r in conn.execute(
    'SELECT agent_name, ROUND(SUM(cost_usd), 4) AS spent, COUNT(*) AS n_calls '
    'FROM budget_ledger WHERE cache_hit=0 GROUP BY agent_name ORDER BY spent DESC'
):
    print(f'  {r["agent_name"]:20s}  ${r["spent"]:.4f}  ({r["n_calls"]} calls)')
total = conn.execute('SELECT COALESCE(SUM(cost_usd), 0) AS s FROM budget_ledger WHERE cache_hit=0').fetchone()['s']
hits = conn.execute('SELECT COUNT(*) AS n FROM budget_ledger WHERE cache_hit=1').fetchone()['n']
print(f'TOTAL real spend: ${total:.4f}')
print(f'cache hits: {hits}')
