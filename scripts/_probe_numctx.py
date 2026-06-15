"""Read-only probe: does Ollama's /v1 (OpenAI-compat) endpoint honor
options.num_ctx? Sends a tiny chat request asking for a large context, then
the caller runs `ollama ps` to read the CONTEXT actually allocated.

Safe to delete. Run from WSL:
    /home/gdp/autoscientist/.venv/bin/python scripts/_probe_numctx.py
"""
import json
import urllib.request

REQ_CTX = 65536
body = json.dumps({
    "model": "qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    "max_tokens": 5,
    "options": {"num_ctx": REQ_CTX},
}).encode()

req = urllib.request.Request(
    "http://localhost:11434/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
print(f"[probe] requested num_ctx={REQ_CTX}")
resp = urllib.request.urlopen(req, timeout=300).read().decode()
print("[probe] response (truncated):", resp[:240])
