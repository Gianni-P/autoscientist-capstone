import sqlite3
conn = sqlite3.connect('/home/gdp/autoscientist/autoscientist.db')
run = conn.execute(
    "SELECT status FROM runs WHERE run_id = 'run_73e0f6c14e374cb1b0e92dc44421f688'"
).fetchone()
print('Run status:', run[0])
conn.close()
