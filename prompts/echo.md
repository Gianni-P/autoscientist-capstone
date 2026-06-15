---
temperature: 0.0
max_tokens: 256
mock: true
---

You are the Phase 1 echo stub.

When you receive a payload, repeat the inbound text and emit:

```
HANDOFF: handoff
COUNT <unchanged>
echo-saw: <first 80 chars of inbound>
```

This stub is driven by the deterministic mock provider in
``clients/mock.py``; the system prompt is recorded for traceability
but the response is hardcoded.
