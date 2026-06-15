#!/usr/bin/env bash
LOG=/home/gdp/autoscientist/runs/run_78f82b18daef406481c1d80c6c199550/logs/run.jsonl
LAST_RESUME=$(grep -n "run.resume" "$LOG" | tail -1 | cut -d: -f1)
TOTAL=$(wc -l < "$LOG")
echo "Log lines: $TOTAL   last run.resume at line: $LAST_RESUME"
echo "=== key events since then ==="
tail -n +"$LAST_RESUME" "$LOG" \
  | jq -r 'select(.event | test("ollama.complete.done|execute.start|execute.done|tools.dispatch.ok|run.agent_done|tool_loop|run.end|run.checkpoint_opened"))
           | "\(.timestamp[11:19]) [\(.event)] ct=\(.completion_tokens // "") rc=\(.reasoning_chars // "") n_tc=\(.n_tool_calls // "") round=\(.round // "") cc=\(.content_chars // "") cmd=\((.cmd // ["",""])[1] | tostring[0:90])"' \
  2>/dev/null
echo
echo "=== latest assistant content/reasoning in DB ==="
cd /home/gdp/autoscientist
uv run python -c "
import sqlite3
conn = sqlite3.connect('autoscientist.db'); conn.row_factory = sqlite3.Row
row = conn.execute(\"SELECT rowid, role, agent_name, length(content) AS cc, length(reasoning) AS rc, completion_tokens FROM messages WHERE run_id='run_78f82b18daef406481c1d80c6c199550' AND agent_name='code_gen' ORDER BY rowid DESC LIMIT 1\").fetchone()
print(f'rowid={row[\"rowid\"]} role={row[\"role\"]} content_chars={row[\"cc\"]} reasoning_chars={row[\"rc\"]} ct={row[\"completion_tokens\"]}')
" 2>/dev/null
