"""Phase 8 pre-flight: one real Claude API call through the router.

Validates the live Anthropic wiring before committing to an end-to-end
pneumonia run. What this proves and how:

  1. ``ANTHROPIC_API_KEY`` is set and accepted by Anthropic.
  2. ``router.route`` dispatches to ``claude_haiku``, gets a real
     response, and writes a non-zero charge to ``budget_ledger``.
  3. Cache hit on a second identical call charges zero (cache before
     budget per KICKOFF Section 10).
  4. The hard budget cap actually refuses calls -- we set
     ``AUTOSCIENTIST_MONTHLY_CAP_USD=0.0001`` in a side connection and
     confirm ``BudgetExceeded`` is raised.
  5. (Optional) Ollama qwen3.6:27b answers a one-token prompt locally,
     to confirm the OpenAI-compat shim works end-to-end too.

Cost: well under one cent. We use ``claude_haiku`` with a tiny
prompt and ``max_tokens=64``, so the worst case is roughly:
  prompt: ~150 tokens * $1/Mtok  = ~$0.00015
  output:  ~64 tokens * $5/Mtok  = ~$0.00032
  total:                        ~ $0.0005

    uv run python scripts/dry_run_phase8.py            # Claude only
    uv run python scripts/dry_run_phase8.py --ollama   # also check Qwen
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Use a dedicated DB so the dry-run doesn't collide with the operator's
# real autoscientist.db (or its budget ledger).
_DB = _REPO / "dry_run_phase8.db"
os.environ["AUTOSCIENTIST_DB_PATH"] = str(_DB)


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def passed(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> str:
    print(f"  FAIL  {msg}")
    return msg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dry_run_phase8")
    parser.add_argument(
        "--ollama", action="store_true",
        help="Also exercise Ollama qwen_27b. Slower; requires ollama running.",
    )
    parser.add_argument(
        "--keep-db", action="store_true",
        help="Keep dry_run_phase8.db at exit instead of deleting it.",
    )
    args = parser.parse_args(argv)

    # Fresh DB each run -- the cache-hit assertion needs an empty cache.
    if _DB.exists():
        _DB.unlink()

    failures: list[str] = []

    # Lazy imports so missing deps surface as test failures, not import errors.
    from autoscientist.clients.router import route
    from autoscientist.runtime.budget import BudgetExceeded
    from autoscientist.runtime.config import load_config
    from autoscientist.state.db import open_db

    cfg = load_config(reload=True)

    # ------------------------------------------------------------------
    # 0) ANTHROPIC_API_KEY present.
    # ------------------------------------------------------------------
    section("Credentials: ANTHROPIC_API_KEY loaded")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        failures.append(fail("ANTHROPIC_API_KEY is unset; cannot make a real call"))
        return _finish(failures, args.keep_db)
    if not key.startswith("sk-ant-"):
        failures.append(fail(f"ANTHROPIC_API_KEY does not look like an Anthropic key "
                             f"(starts {key[:7]!r}, len={len(key)})"))
        return _finish(failures, args.keep_db)
    passed(f"ANTHROPIC_API_KEY present (len={len(key)})")

    # ------------------------------------------------------------------
    # 1) One real haiku call through the router. Tiny payload.
    # ------------------------------------------------------------------
    section("Live call: claude_haiku roundtrip via router")
    conn = open_db(_DB)
    try:
        prompt_text = (
            "Reply with exactly the string OK. Do not say anything else."
        )
        try:
            result = route(
                conn=conn,
                agent_name="lit_review",  # routed to claude_haiku
                system="You are a smoke-test responder. Follow instructions exactly.",
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=16,
                temperature=0.0,
                cfg=cfg,
            )
        except Exception as e:
            failures.append(fail(f"router.route raised {type(e).__name__}: {e}"))
            return _finish(failures, args.keep_db, conn=conn)

        passed(f"provider={result.provider} model={result.model}")
        passed(f"prompt_tokens={result.prompt_tokens} "
               f"completion_tokens={result.completion_tokens}")
        if not result.content:
            failures.append(fail(f"empty content (finish_reason={result.finish_reason!r})"))
        else:
            passed(f"content (truncated): {result.content[:80]!r}")

        # Verify charge landed in budget_ledger.
        row = conn.execute(
            "SELECT cost_usd, cache_hit, prompt_tokens, completion_tokens "
            "FROM budget_ledger ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            failures.append(fail("no row in budget_ledger after live call"))
        else:
            cost = float(row["cost_usd"])
            cache_hit = bool(row["cache_hit"])
            if cache_hit:
                failures.append(fail("first call recorded as cache_hit (impossible on fresh DB)"))
            elif cost <= 0.0:
                failures.append(fail(f"first call recorded cost_usd=${cost} (expected > 0)"))
            else:
                passed(f"budget_ledger row: cost_usd=${cost:.6f}, cache_hit={cache_hit}")

        # ------------------------------------------------------------------
        # 2) Identical second call -> cache hit, zero charge.
        # ------------------------------------------------------------------
        section("Cache: identical call hits the cache and charges $0")
        result2 = route(
            conn=conn,
            agent_name="lit_review",
            system="You are a smoke-test responder. Follow instructions exactly.",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=16,
            temperature=0.0,
            cfg=cfg,
        )
        if result2.content != result.content:
            failures.append(fail(
                f"cache returned different content: {result2.content!r} vs {result.content!r}"
            ))
        else:
            passed("cached content matches the first call")
        row2 = conn.execute(
            "SELECT cost_usd, cache_hit FROM budget_ledger ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if not (row2 and bool(row2["cache_hit"]) and float(row2["cost_usd"]) == 0.0):
            failures.append(fail(
                f"cache hit not recorded with $0 charge: row={dict(row2) if row2 else None}"
            ))
        else:
            passed("cache hit recorded with cost_usd=$0.0")

        # ------------------------------------------------------------------
        # 3) Total spend stays under one cent.
        # ------------------------------------------------------------------
        section("Spend invariant: this dry-run cost less than $0.01")
        total = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger "
            "WHERE cache_hit = 0"
        ).fetchone()["s"]
        if total >= 0.01:
            failures.append(fail(f"spent ${total:.6f}, which exceeds the $0.01 dry-run ceiling"))
        else:
            passed(f"total real spend: ${total:.6f}")
        conn.commit()
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 4) Budget enforcer fires when cap is set tiny.
    # ------------------------------------------------------------------
    section("Budget enforcer: refuses calls when monthly cap is exhausted")
    # Reload config with a near-zero cap; the env override takes precedence.
    os.environ["AUTOSCIENTIST_MONTHLY_CAP_USD"] = "0.0001"
    os.environ["AUTOSCIENTIST_BUDGET_BUFFER_USD"] = "0.0"
    try:
        cfg_tight = load_config(reload=True)
        conn2 = open_db(_DB)
        try:
            try:
                route(
                    conn=conn2,
                    agent_name="idea_gen",  # claude_sonnet -> larger estimate, cap blows
                    system="You are responding to a budget-cap test.",
                    messages=[{"role": "user", "content": "Respond with one sentence."}],
                    max_tokens=128,
                    temperature=0.0,
                    cfg=cfg_tight,
                )
                failures.append(fail("budget enforcer did NOT raise on a $0.0001 monthly cap"))
            except BudgetExceeded as e:
                passed(f"BudgetExceeded raised as expected: {e}")
        finally:
            conn2.close()
    finally:
        # Always restore the env so subsequent tooling sees the real cap.
        os.environ.pop("AUTOSCIENTIST_MONTHLY_CAP_USD", None)
        os.environ.pop("AUTOSCIENTIST_BUDGET_BUFFER_USD", None)
        load_config(reload=True)

    # ------------------------------------------------------------------
    # 5) (Optional) Ollama qwen_27b smoke.
    # ------------------------------------------------------------------
    if args.ollama:
        section("Live call: Ollama qwen_27b (local, free)")
        cfg2 = load_config(reload=True)
        conn3 = open_db(_DB)
        try:
            try:
                qresult = route(
                    conn=conn3,
                    agent_name="code_gen",  # routed to qwen_27b
                    system="Reply with exactly the string OK and nothing else.",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=16,
                    temperature=0.0,
                    cfg=cfg2,
                )
                passed(f"qwen provider={qresult.provider} model={qresult.model}")
                passed(f"content={qresult.content[:80]!r}")
            except Exception as e:
                failures.append(fail(f"Ollama call raised {type(e).__name__}: {e}"))
        finally:
            conn3.close()

    return _finish(failures, args.keep_db)


def _finish(failures: list[str], keep_db: bool, conn=None) -> int:
    if conn is not None:
        conn.close()
    print()
    if failures:
        print(f"*** Dry run FAILED: {len(failures)} issue(s) ***")
        for f in failures:
            print(f"  - {f}")
        rc = 1
    else:
        print("*** Dry run PASSED -- live wiring is good. ***")
        rc = 0
    if keep_db:
        print(f"  DB kept at: {_DB}")
    elif _DB.exists():
        try:
            _DB.unlink()
        except OSError:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
