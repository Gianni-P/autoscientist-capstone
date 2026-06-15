import sqlite3
conn = sqlite3.connect('autoscientist.db')
cursor = conn.execute('PRAGMA table_info(checkpoints)')
for row in cursor:
    print(row)
print()
conn.row_factory = sqlite3.Row
cp = conn.execute(
    "SELECT * FROM checkpoints WHERE run_id = 'run_73e0f6c14e374cb1b0e92dc44421f688' ORDER BY created_at DESC LIMIT 1"
).fetchone()
if cp:
    print("Columns:", cp.keys())
    for k in cp.keys():
        v = cp[k]
        if isinstance(v, str) and len(v) > 200:
            v = v[:200] + '...'
        print(f"  {k}: {v}")
