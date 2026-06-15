"""Load qwen3-coder-30b-64k via the /v1 endpoint and print the reply.
Caller then runs `ollama ps` to confirm CONTEXT=65536 and GPU fit. Safe to delete.
"""
import json
import urllib.request

body = json.dumps({
    "model": "qwen3-coder-30b-64k",
    "messages": [{"role": "user", "content": "reply with the single word: ok"}],
    "max_tokens": 5,
}).encode()
req = urllib.request.Request(
    "http://localhost:11434/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
print("[verify] loading qwen3-coder-30b-64k ...")
print("[verify] reply:", urllib.request.urlopen(req, timeout=300).read().decode()[:160])
