"""Detached ``runner --resume`` launcher.

Identical semantics to ``autoscientist.checkpoints.ui._resume_in_background``:
resuming inline would block the web server until every downstream agent
finished, and a daemon thread dies when the server restarts (wedging the
run in 'running'). So we spawn the documented standalone resume process
(``python -m autoscientist.runtime.runner --resume <run_id>``) in a new
session; it keeps driving the chain and updates the DB the console polls,
independent of this process's lifecycle. A duplicate launch is harmless —
the second ``resume_run`` sees status != 'paused' and exits.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from autoscientist.runtime.config import load_config


def _spawn_detached(cmd: list[str], *, what: str, **log_fields: Any) -> bool:
    """Spawn ``cmd`` as a detached process. Returns True on success."""
    popen_kwargs: dict[str, Any] = {
        "cwd": str(load_config().root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "posix":
        # New session so the child isn't in the server's process group and
        # survives the server going away.
        popen_kwargs["start_new_session"] = True
    try:
        subprocess.Popen(cmd, **popen_kwargs)
        return True
    except Exception as e:  # never let a launch failure crash a request
        import structlog

        structlog.get_logger("autoscientist.web").exception(
            f"web.{what}_launch_failed", error=str(e), **log_fields
        )
        return False


def resume_command(run_id: str) -> list[str]:
    return [sys.executable, "-m", "autoscientist.runtime.runner", "--resume", run_id]


def resume_in_background(run_id: str) -> bool:
    """Launch a detached ``runner --resume <run_id>``. True if spawned."""
    return _spawn_detached(resume_command(run_id), what="resume", run_id=run_id)


def start_command(*, project: str, agent: str, payload: str) -> list[str]:
    return [
        sys.executable, "-m", "autoscientist.runtime.runner",
        "--agent", agent, "--project", project, "--payload", payload,
    ]


def start_in_background(*, project: str, agent: str, payload: str) -> bool:
    """Launch a detached fresh run (``runner --agent ... --project ...``).

    The runner creates the run row immediately (``start_run``), so the console
    surfaces the new run on its next refresh even though we don't capture the
    printed run_id from the detached process.
    """
    return _spawn_detached(
        start_command(project=project, agent=agent, payload=payload),
        what="start", project=project, agent=agent,
    )
