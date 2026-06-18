"""Unit tests for reciprocal-rank fusion + blended ranking (no DB / embeddings).

`_fuse` is pure given `self.settings`, so we drive it with a service whose pool
and embedder are unused.
"""

from __future__ import annotations

from hypermnesia.config import Settings
from hypermnesia.service import MemoryService


def _svc(**overrides) -> MemoryService:
    return MemoryService(pool=None, embedder=None, settings=Settings(**overrides))


def _row(mid: str, similarity: float, **kw) -> dict:
    base = {
        "id": mid,
        "owner_id": "o",
        "scope": "s",
        "content": f"content {mid}",
        "description": f"desc {mid}",
        "similarity": similarity,
        "last_accessed_at": None,  # recency term -> 0, keeps scores deterministic
    }
    base.update(kw)
    return base


def test_lexical_match_bypasses_similarity_floor():
    svc = _svc()
    # b is semantically weak (sim 0.1, below floor) but a lexical match.
    vrows = [_row("a", 0.9), _row("b", 0.1)]
    lrows = [_row("b", 0.1)]
    hits = svc._fuse(vrows, lrows, floor=0.4, k=10)
    ids = [h.id for h in hits]
    assert ids == ["b", "a"]  # b wins: ranked in both lists
    assert hits[0].similarity == 0.1  # raw cosine still reported


def test_vector_only_below_floor_is_dropped():
    svc = _svc()
    vrows = [_row("a", 0.1)]  # weak semantic, no lexical match
    hits = svc._fuse(vrows, [], floor=0.4, k=10)
    assert hits == []


def test_item_ranked_high_in_both_lists_wins():
    svc = _svc()
    vrows = [_row("a", 0.7), _row("b", 0.6)]
    lrows = [_row("a", 0.7), _row("b", 0.6)]
    hits = svc._fuse(vrows, lrows, floor=0.0, k=10)
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score


def test_pure_vector_when_no_lexical_results():
    svc = _svc()
    vrows = [_row("a", 0.8), _row("b", 0.5)]
    hits = svc._fuse(vrows, [], floor=0.0, k=10)
    assert [h.id for h in hits] == ["a", "b"]  # preserves vector order


def test_empty_inputs_return_empty():
    svc = _svc()
    assert svc._fuse([], [], floor=0.0, k=10) == []


def test_k_truncates_results():
    svc = _svc()
    vrows = [_row(str(i), 0.9 - i * 0.01) for i in range(10)]
    hits = svc._fuse(vrows, [], floor=0.0, k=3)
    assert len(hits) == 3


def test_forget_with_no_scopes_is_a_noop():
    # No accessible scopes -> never touches the pool, returns an empty dry-run.
    import asyncio

    svc = _svc()
    res = asyncio.run(svc.forget([], older_than_days=180, importance_floor=1.0))
    assert res == {"dry_run": True, "scopes": [], "matched": 0, "memories": []}


def test_importance_breaks_ties_within_a_rank():
    # Same rank in the same single list -> relevance ties; importance decides.
    svc = _svc()
    vrows = [_row("low", 0.8, importance=0.0), _row("high", 0.8, importance=2.0)]
    # both are vector rank 1 and 2; give them equal vector rank by using two lists
    # that cross so rrf is symmetric, isolating importance.
    lrows = [_row("high", 0.8, importance=2.0), _row("low", 0.8, importance=0.0)]
    hits = svc._fuse(vrows, lrows, floor=0.0, k=10)
    assert hits[0].id == "high"
