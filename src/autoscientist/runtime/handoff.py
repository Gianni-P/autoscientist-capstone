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

# Capture only the target token at the start of a HANDOFF line. Deliberately
# does NOT anchor to end-of-line: the shipped prompts decorate the directive
# (``HANDOFF: code_gen   # if revise``, ``HANDOFF: <repo_publisher if accept…>``),
# and an end-anchored ``(\S+)$`` rejected those outright — returning None and
# silently dropping the handoff. Optional leading ``<`` absorbs the placeholder
# form; the rest of the line (comment / conditional prose) is ignored.
_HANDOFF_RE = re.compile(
    r"^[ \t]*HANDOFF:[ \t]*<?([A-Za-z_][A-Za-z0-9_]*|DONE)\b", re.MULTILINE
)


@dataclass(frozen=True)
class Handoff:
    from_agent: str
    to_agent: str
    payload: str

    @property
    def is_terminal(self) -> bool:
        return self.to_agent == DONE


def parse_handoff(content: str, from_agent: str) -> Handoff | None:
    """Look for a ``HANDOFF: <target>`` directive at the start of a line.

    When several directives appear (an illustrative one in the body plus the
    real one at the end, or a file body that literally contains the token), the
    LAST is operative: agents are instructed to end their turn with it and the
    runner appends the structured-handoff directive at the end. The payload is
    everything after the directive's line.

    Returns ``None`` if no directive — the runner treats this as terminal.
    """
    matches = list(_HANDOFF_RE.finditer(content))
    if not matches:
        return None
    m = matches[-1]
    target = m.group(1).strip()
    line_end = content.find("\n", m.end())
    payload = content[line_end + 1:].strip() if line_end != -1 else ""
    return Handoff(from_agent=from_agent, to_agent=target, payload=payload)
