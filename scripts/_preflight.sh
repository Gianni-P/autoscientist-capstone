#!/usr/bin/env bash
set -e
cd /home/gdp/autoscientist

echo "=== Kickoff payload ==="
head -3 projects/pneumonia-data-efficiency/kickoff_payload.json
echo ""

echo "=== Ollama models ==="
curl -s localhost:11434/api/tags | python3 -c "
import sys, json
tags = json.load(sys.stdin)
for m in tags.get('models', []):
    print(f'  {m[\"name\"]}')
" 2>&1 || echo "  ERROR: Ollama not reachable"

echo ""
echo "=== Dataset presence ==="
ls -la projects/pneumonia-data-efficiency/sandbox/data 2>/dev/null || echo "  no sandbox/data symlink"
ls -la projects/pneumonia-data-efficiency/datasets 2>/dev/null || echo "  no datasets symlink"

echo ""
echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

echo ""
echo "=== ANTHROPIC_API_KEY set? ==="
[ -n "$ANTHROPIC_API_KEY" ] && echo "  yes" || echo "  NOT SET"
