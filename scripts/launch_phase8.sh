#!/usr/bin/env bash
# Launch Phase 8 runner with the pneumonia-data-efficiency kickoff payload.
# Resolves $(cat ...) inside the WSL shell so harness shells don't pre-expand it.
set -euo pipefail
cd /home/gdp/autoscientist
PAYLOAD=$(cat projects/pneumonia-data-efficiency/kickoff_payload.json)
echo "payload bytes: ${#PAYLOAD}" >&2
exec uv run python -m autoscientist.runtime.runner \
    --agent lit_review \
    --project pneumonia-data-efficiency \
    --payload "$PAYLOAD"
