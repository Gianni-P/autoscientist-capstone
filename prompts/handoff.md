---
temperature: 0.0
max_tokens: 256
mock: true
---

You are the Phase 1 handoff stub.

When you receive ``COUNT N``:

  * if ``N > 1``, emit ``HANDOFF: echo`` with ``COUNT N-1``
  * if ``N <= 1``, emit ``HANDOFF: DONE``

This stub is driven by the deterministic mock provider in
``clients/mock.py``; the system prompt is recorded for traceability
but the response is hardcoded.
