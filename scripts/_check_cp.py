import sqlite3
conn = sqlite3.connect('/home/gdp/autoscientist/autoscientist.db')
cp = conn.execute(
    "SELECT stage, status FROM checkpoints WHERE run_id = 'run_73e0f6c14e374cb1b0e92dc44421f688' ORDER BY created_at DESC LIMIT 1"
).fetchone()
print('Stage:', cp[0], 'Status:', cp[1])
conn.close()
