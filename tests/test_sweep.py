"""Unit tests for the periodic forget sweep (no DB — the service is stubbed).

The timer itself is trivial asyncio; what matters is that a pass covers every
scope with apply=true and server-default thresholds, and that one failing
pass doesn't kill the loop.
"""

from __future__ import annotations

import asyncio

from hypermnesia.config import Settings
from hypermnesia.server import forget_sweep_loop, run_forget_sweep_once


class StubService:
    def __init__(self, fail_times: int = 0):
        self.forget_calls: list[dict] = []
        self.fail_times = fail_times

    async def distinct_scopes(self):
        return ["project:a", "shared"]

    async def forget(self, scopes, *, older_than_days, importance_floor, apply, limit=100):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("db hiccup")
        call = {
            "scopes": scopes, "older_than_days": older_than_days,
            "importance_floor": importance_floor, "apply": apply,
        }
        self.forget_calls.append(call)
        return {"dry_run": False, "scopes": scopes, "matched": 0,
                "truncated": False, "memories": []}


async def test_sweep_once_applies_defaults_over_all_scopes():
    svc = StubService()
    settings = Settings(forget_after_days=42.0, forget_importance_floor=0.7)
    result = await run_forget_sweep_once(svc, settings)
    assert result["dry_run"] is False
    (call,) = svc.forget_calls
    assert call == {
        "scopes": ["project:a", "shared"],
        "older_than_days": 42.0,
        "importance_floor": 0.7,
        "apply": True,
    }


async def test_sweep_loop_ticks_and_survives_a_failing_pass():
    svc = StubService(fail_times=1)  # first pass raises, later ones succeed
    settings = Settings(forget_sweep_hours=0.01 / 3600)  # ~10ms interval
    task = asyncio.create_task(forget_sweep_loop(svc, settings))
    try:
        async with asyncio.timeout(2):
            while len(svc.forget_calls) < 2:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
    # it kept ticking after the failure, always applying
    assert all(c["apply"] is True for c in svc.forget_calls)
