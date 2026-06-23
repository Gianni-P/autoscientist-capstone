"""Launch a fresh run for a project, reading its kickoff payload from disk.

Avoids passing a large multi-line JSON payload through the shell (which mangles
it across the Git-Bash -> wsl.exe boundary).

Usage: python scripts/_launch.py <project_id> [starting_agent]
"""

import sys
import tomllib
from pathlib import Path

from autoscientist.runtime.config import load_config
from autoscientist.runtime.runner import run

project_id = sys.argv[1]
starting_agent = sys.argv[2] if len(sys.argv) > 2 else "lit_review"

proj = tomllib.load(open(Path("projects") / project_id / "config.toml", "rb"))
payload_path = proj["project"].get(
    "kickoff_payload_path", f"projects/{project_id}/kickoff_payload.json"
)
payload = Path(payload_path).read_text(encoding="utf-8")

cfg = load_config()
run_id = run(
    starting_agent=starting_agent,
    project_id=project_id,
    initial_payload=payload,
    enable_checkpoints=True,
    cfg=cfg,
)
print("RUN_ID", run_id)
