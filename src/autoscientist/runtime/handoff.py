"""Handoff payload passed between agents in the run loop.

In Phase 1 each agent sees only its own conversation history. The
handoff is the *only* state carried forward: a string payload plus an
explicit ``to`` target. The runner inspects the agent's output for a
``HANDOFF: <target>`` directive on its own line and routes accordingly.

Real agents in Phase 2+ will emit structured JSON; this regex-driven
contract is the lowest-friction stub for exercising the runtime.

Special target ``DONE`` ends the run cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DONE = "DONE"

_HANDOFF_RE = re.compile(r"^\s*HANDOFF:\s*(\S+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Handoff:
    from_agent: str
    to_agent: str
    payload: str

    @property
    def is_terminal(self) -> bool:
        return self.to_agent == DONE


def parse_handoff(content: str, from_agent: str) -> Handoff | None:
    """Look for a ``HANDOFF: <target>`` directive on its own line.

    Returns ``None`` if no directive — the runner treats this as terminal.
    The payload is everything after the directive line.
    """
    m = _HANDOFF_RE.search(content)
    if not m:
        return None
    target = m.group(1).strip()
    payload = content[m.end():].strip()
    return Handoff(from_agent=from_agent, to_agent=target, payload=payload)
