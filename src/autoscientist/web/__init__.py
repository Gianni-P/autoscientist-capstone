"""Snappy operator console (Starlette + Server-Sent Events).

A zero-extra-dependency replacement for the Streamlit console
(``autoscientist.checkpoints.ui``). The Streamlit page is kept as a
fallback; this package is the new default.

Why a second console
~~~~~~~~~~~~~~~~~~~~~
Streamlit re-runs the whole script on every interaction and tails the
DB by polling every 2-3 s, so it never feels instant. This console
serves a static single-page app and *pushes* state changes over SSE the
moment they land in ``autoscientist.db``; every operator action is a
plain ``fetch`` with an optimistic UI update. The backend reuses the
exact same Python logic the Streamlit page does — ``cp_manager`` for
checkpoint resolution, ``runtime.control`` for pause/resume, and a
detached ``runner --resume`` subprocess to drive the chain — so the two
consoles are behaviourally identical; only the presentation differs.

Run it::

    uv run uvicorn autoscientist.web.app:app --port 8650
    # or:
    uv run python -m autoscientist.web

The ASGI stack (starlette / uvicorn / sse-starlette) is already present
in the environment (the ``mcp`` SDK depends on it), so no new package is
required.
"""

from __future__ import annotations
