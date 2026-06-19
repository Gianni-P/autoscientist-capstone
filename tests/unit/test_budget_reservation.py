"""Tests for the H13 reserve-before-call budget fix.

The old check (assert_can_spend) read SUM(cost_usd) then the caller made the LLM
call and recorded the charge separately — two concurrent callers could both read
the same stale total, both pass, and overshoot the monthly cap. reserve_charge
checks-and-writes atomically under a single BEGIN IMMEDIATE write lock; these
tests pin both the single-process settlement and the concurrent-process race.
"""

from __future__ import annotations

import threading
from contextlib import closing

import pytest

from autoscientist.runtime.budget import (
    BudgetConfig,
    BudgetExceeded,
    monthly_spent,
    project_spent,
    reconcile_charge,
    record_charge,
    release_reservation,
    reserve_charge,
)
from autoscientist.state.db import open_db, start_run


def _cfg(cap: float = 10.0, buffer: float = 0.0) -> BudgetConfig:
    return BudgetConfig(monthly_cap_usd=cap, hard_floor_buffer_usd=buffer)


def test_reserve_then_reconcile_settles_to_actual(tmp_path):
    with closing(open_db(tmp_path / "b.db")) as conn:
        lid = reserve_charge(
            conn, run_id="r", agent_name="a", provider="claude", model="m",
            estimated_cost_usd=2.0, cfg=_cfg(),
        )
        # Reservation counts toward spend immediately (so a concurrent caller sees it).
        assert monthly_spent(conn) == pytest.approx(2.0)
        reconcile_charge(conn, lid, cost_usd=0.5, prompt_tokens=10, completion_tokens=5)
        conn.commit()
        assert monthly_spent(conn) == pytest.approx(0.5)
        rows = conn.execute(
            "SELECT cost_usd, prompt_tokens, completion_tokens FROM budget_ledger"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["cost_usd"] == pytest.approx(0.5)
        assert rows[0]["prompt_tokens"] == 10
        assert rows[0]["completion_tokens"] == 5


def test_reserve_over_cap_raises_and_writes_nothing(tmp_path):
    with closing(open_db(tmp_path / "b.db")) as conn:
        record_charge(
            conn, run_id="r", agent_name="seed", provider="c", model="m",
            prompt_tokens=0, completion_tokens=0, cost_usd=9.5, cache_hit=False,
        )
        conn.commit()
        with pytest.raises(BudgetExceeded):
            reserve_charge(
                conn, run_id="r", agent_name="a", provider="c", model="m",
                estimated_cost_usd=1.0, cfg=_cfg(),  # 9.5 + 1.0 > 10 threshold
            )
        # Refused reservations leave the ledger untouched.
        assert monthly_spent(conn) == pytest.approx(9.5)
        assert conn.execute("SELECT COUNT(*) AS n FROM budget_ledger").fetchone()["n"] == 1


def test_release_reservation_removes_row(tmp_path):
    with closing(open_db(tmp_path / "b.db")) as conn:
        lid = reserve_charge(
            conn, run_id="r", agent_name="a", provider="c", model="m",
            estimated_cost_usd=3.0, cfg=_cfg(),
        )
        assert monthly_spent(conn) == pytest.approx(3.0)
        release_reservation(conn, lid)
        conn.commit()
        assert monthly_spent(conn) == pytest.approx(0.0)


def test_reserve_enforces_project_soft_cap(tmp_path):
    with closing(open_db(tmp_path / "b.db")) as conn:
        run_id = start_run(conn, project_id="proj")
        conn.commit()
        record_charge(
            conn, run_id=run_id, agent_name="a", provider="c", model="m",
            prompt_tokens=0, completion_tokens=0, cost_usd=4.0, cache_hit=False,
        )
        conn.commit()
        # Under the (high) monthly cap but over the $5 project soft cap.
        with pytest.raises(BudgetExceeded):
            reserve_charge(
                conn, run_id=run_id, agent_name="a", provider="c", model="m",
                estimated_cost_usd=2.0, cfg=_cfg(cap=1000.0),
                project_id="proj", project_soft_cap_usd=5.0,
            )
        assert project_spent(conn, "proj") == pytest.approx(4.0)


def test_concurrent_reservations_cannot_overshoot_cap(tmp_path):
    """The race the fix exists to close: two processes, one slot left."""
    db = tmp_path / "b.db"
    # Seed $9 spend with a $10 cap → only ONE more $0.6 reservation fits.
    with closing(open_db(db)) as conn:
        record_charge(
            conn, run_id="seed", agent_name="seed", provider="c", model="m",
            prompt_tokens=0, completion_tokens=0, cost_usd=9.0, cache_hit=False,
        )
        conn.commit()

    barrier = threading.Barrier(2)
    results: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        conn = open_db(db)
        try:
            barrier.wait()  # maximise contention — both hit reserve together
            try:
                reserve_charge(
                    conn, run_id=f"r{i}", agent_name="a", provider="c", model="m",
                    estimated_cost_usd=0.6, cfg=_cfg(),
                )
                with lock:
                    results.append("ok")
            except BudgetExceeded:
                with lock:
                    results.append("refused")
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one reservation succeeds; the other is refused — never both.
    assert sorted(results) == ["ok", "refused"], results
    with closing(open_db(db)) as conn:
        spent = monthly_spent(conn)
        assert spent == pytest.approx(9.6)
        assert spent <= 10.0  # cap never overshot
