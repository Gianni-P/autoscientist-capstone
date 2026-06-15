"""Fix run status to 'paused' so resume_run accepts it."""
import sys
sys.path.insert(0, '/home/gdp/autoscientist/src')
from autoscientist.state.db import open_db

conn = open_db('/home/gdp/autoscientist/autoscientist.db')
conn.execute(
    "UPDATE runs SET status = 'paused' WHERE run_id = 'run_73e0f6c14e374cb1b0e92dc44421f688'"
)
conn.commit()
print("Fixed run status to 'paused'")

r = conn.execute(
    "SELECT status FROM runs WHERE run_id = 'run_73e0f6c14e374cb1b0e92dc44421f688'"
).fetchone()
print(f"Confirmed status: {r['status']}")
conn.close()
