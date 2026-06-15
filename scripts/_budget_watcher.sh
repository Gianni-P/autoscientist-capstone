#!/usr/bin/env bash
# Poll cumulative monthly spend every 30 s; if > $LIMIT, kill the runner.
LIMIT=${1:-35.00}
DB=/home/gdp/autoscientist/autoscientist.db
echo "watching cumulative spend, kill if > \$$LIMIT"
while true; do
  SPEND=$(/home/gdp/.local/bin/uv run --project /home/gdp/autoscientist python -c "
import sqlite3
c=sqlite3.connect('$DB')
v=c.execute('SELECT COALESCE(SUM(cost_usd),0) FROM budget_ledger WHERE cache_hit=0').fetchone()[0]
print(f'{v:.4f}')
" 2>/dev/null)
  if [ -z "$SPEND" ]; then
    sleep 5; continue
  fi
  awk -v s="$SPEND" -v l="$LIMIT" 'BEGIN { if (s+0 > l+0) exit 0; else exit 1 }'
  if [ $? -eq 0 ]; then
    echo "KILL: cumulative \$$SPEND exceeds \$$LIMIT"
    pkill -f "python.*runtime.runner|python.*_resume_after_codegen" 2>/dev/null
    echo "killed runner"
    exit 0
  fi
  echo "[$(date +%H:%M:%S)] cumulative=\$$SPEND (limit \$$LIMIT)"
  sleep 30
done
