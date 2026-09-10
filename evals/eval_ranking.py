"""Score candidate fusion strategies against one fixed set of retrieval candidates.

Seeds `evals.corpus` into a scratch database, pulls the vector and lexical
candidate lists once per query via the production `_fetch_candidates`, then
replays every strategy over those cached candidates. Because the candidates are
identical across strategies, any difference in the reported accuracy is caused
by fusion alone — no embedding noise, no retrieval variance.

    python -m evals.eval_ranking --database-url postgresql://...

Reports top-1 accuracy and MRR split by query kind, because a fix that helps
conversational queries by discarding the lexical signal would show up as an
exact-token regression rather than as a win.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Callable

from psycopg.rows import dict_row

from hypermnesia.config import Settings
from hypermnesia.service import MemoryService

from .corpus import CORPUS, QUERIES

SCOPE = "eval"
OWNER = "eval"

# A strategy maps (similarity, vector rank or None, lexical rank or None, settings)
# to a relevance in [0,1]; the recency/importance blend is then applied on top,
# exactly as in production. `blend` names the weights that blend runs with, so
# the two independent causes of a wrong top hit — how relevance is fused, and
# how much recency and importance are allowed to move it — can be varied on
# separate axes and attributed separately.
Strategy = Callable[[float, int | None, int | None, Settings], float]

# Relevance must be *spread* over the blend's weights to matter. bge-small puts
# every plausible hit in a narrow cosine band (unrelated ~0.30-0.45, relevant
# ~0.55-0.73), so a raw-similarity relevance varies by ~0.18 across a candidate
# set while recency alone can move a score by the full 0.25. Rescaling the band
# to [0,1] restores the documented intent — relevance leads, recency and
# importance break ties — without hand-tuning weights per embedding model.
def _rescale(sim: float, floor: float) -> float:
    return min(max((sim - floor) / (1.0 - floor), 0.0), 1.0) if floor < 1.0 else sim


# The pre-fix constants, pinned here rather than read from Settings so this
# baseline keeps reproducing the old behaviour after the fix ships and those
# settings are gone. Without them the eval would silently start comparing the
# new ranking against itself.
_OLD_RRF_K, _OLD_WV, _OLD_WL = 60, 1.0, 1.0


def _rrf_norm(vrank: int | None, lrank: int | None, s: Settings) -> float:
    """The pre-fix relevance: pure RRF normalised against rank 1 in both lists."""
    kk, wv, wl = _OLD_RRF_K, _OLD_WV, _OLD_WL
    rrf = (wv / (kk + vrank) if vrank else 0.0) + (wl / (kk + lrank) if lrank else 0.0)
    return rrf / (wv / (kk + 1) + wl / (kk + 1))


def current(sim: float, vrank: int | None, lrank: int | None, s: Settings) -> float:
    return _rrf_norm(vrank, lrank, s)


def similarity_only(sim: float, vrank: int | None, lrank: int | None, s: Settings) -> float:
    """Hybrid disabled in all but name: lexical membership changes nothing."""
    return sim


def max_sim_rrf(sim: float, vrank: int | None, lrank: int | None, s: Settings) -> float:
    return max(sim, _rrf_norm(vrank, lrank, s))


def _mix(alpha: float) -> Strategy:
    """Weighted blend of calibrated similarity with the rank-only RRF score."""

    def f(sim: float, vrank: int | None, lrank: int | None, s: Settings) -> float:
        return alpha * sim + (1.0 - alpha) * _rrf_norm(vrank, lrank, s)

    return f


def _boost(amount: float) -> Strategy:
    """Similarity is the relevance; a lexical hit adds a bounded, rank-decayed bonus.

    Keeps the calibrated magnitude that RRF throws away, while still letting an
    exact-token match lift a candidate over near-ties — but never over a
    genuinely better semantic match, because the bonus is capped at `amount`.
    """

    def f(sim: float, vrank: int | None, lrank: int | None, s: Settings) -> float:
        kk = s.hybrid_rank_decay_k
        bonus = amount * ((kk + 1) / (kk + lrank)) if lrank else 0.0
        return min(1.0, sim + bonus)

    return f


def _rescaled_boost(amount: float) -> Strategy:
    """The proposed fix: rescaled similarity, plus a bounded lexical bonus."""

    def f(sim: float, vrank: int | None, lrank: int | None, s: Settings) -> float:
        kk = s.hybrid_rank_decay_k
        bonus = amount * ((kk + 1) / (kk + lrank)) if lrank else 0.0
        return min(1.0, _rescale(sim, s.search_min_similarity) + bonus)

    return f


# Blend weights to try, as (recency, importance). "shipped" is the pre-fix pair,
# pinned so the baseline stays reproducible; "damped" is what now ships.
BLENDS: dict[str, tuple[float, float]] = {
    "shipped": (0.25, 0.15),
    "damped": (0.10, 0.06),
}

# (strategy name, relevance fn, blend name). Not a full cross product — the
# combinations here are the ones that isolate one variable at a time.
STRATEGIES: dict[str, tuple[Strategy, str]] = {
    "current (pure RRF)": (current, "shipped"),
    "similarity only": (similarity_only, "shipped"),
    "max(sim, rrf)": (max_sim_rrf, "shipped"),
    **{f"mix a={a:.2f}": (_mix(a), "shipped") for a in (0.5, 0.7, 0.85)},
    **{f"boost +{b:.2f}": (_boost(b), "shipped") for b in (0.05, 0.10, 0.20)},
    # Blend damping alone, without touching fusion.
    "current + damped": (current, "damped"),
    "boost +0.10 damped": (_boost(0.10), "damped"),
    # Rescaling alone, then rescaling with a lexical bonus.
    "rescaled sim only": (_rescaled_boost(0.0), "shipped"),
    **{f"rescaled+boost {b:.2f}": (_rescaled_boost(b), "shipped")
       for b in (0.05, 0.10, 0.15, 0.20, 0.30)},
    **{f"rescaled+boost {b:.2f} damped": (_rescaled_boost(b), "damped")
       for b in (0.10, 0.15, 0.20, 0.30)},
    **{f"boost +{b:.2f} damped": (_boost(b), "damped") for b in (0.05, 0.15, 0.20)},
}


async def seed(svc: MemoryService) -> dict[str, str]:
    async with svc.pool.connection() as conn:
        await conn.execute("DELETE FROM memories WHERE scope = %s", (SCOPE,))
    ids: dict[str, str] = {}
    for key, desc, content, tags, importance in CORPUS:
        mem, created, _ = await svc.save(
            owner_id=OWNER, scope=SCOPE, content=content, description=desc,
            type="fact", tags=tags, importance=importance,
        )
        if not created:
            raise SystemExit(f"corpus entry {key!r} deduped into another entry — corpus is degenerate")
        ids[key] = mem.id
    return ids


async def collect(svc: MemoryService, k: int) -> list[dict[str, Any]]:
    """Pull the candidate lists once per query, so every strategy sees the same input."""
    out = []
    async with svc.pool.connection() as conn:
        conn.row_factory = dict_row
        for query, expected, kind in QUERIES:
            vrows, lrows = await svc._fetch_candidates(
                conn, query=query, scopes=[SCOPE], tags=None, k=k
            )
            out.append({"query": query, "expected": expected, "kind": kind,
                        "vrows": vrows, "lrows": lrows})
    return out


def rank(svc: MemoryService, case: dict[str, Any], strategy: Strategy,
         blend: str, k: int) -> list[str]:
    """Re-run _fuse's gating and blending with `strategy` supplying the relevance."""
    s = svc.settings
    vrank = {r["id"]: i for i, r in enumerate(case["vrows"], 1)}
    lrank = {r["id"]: i for i, r in enumerate(case["lrows"], 1)}
    rows: dict[str, dict[str, Any]] = {}
    for r in (*case["vrows"], *case["lrows"]):
        rows.setdefault(r["id"], r)

    floor, cutoff = s.search_min_similarity, s.search_relative_cutoff
    above = [r["similarity"] for r in rows.values() if r["similarity"] >= floor]
    rel_floor = max(above) - cutoff if above and cutoff > 0 else float("-inf")

    scored = []
    for mid, row in rows.items():
        if mid not in lrank and (row["similarity"] < floor or row["similarity"] < rel_floor):
            continue
        relevance = strategy(row["similarity"], vrank.get(mid), lrank.get(mid), s)
        w_rec, w_imp = BLENDS[blend]
        blended = Settings(
            database_url=s.database_url,
            score_weight_recency=w_rec,
            score_weight_importance=w_imp,
        )
        svc.settings, saved = blended, s
        try:
            score = svc._rank_score(relevance, row["importance"], row["last_accessed_at"])
        finally:
            svc.settings = saved
        scored.append((score, mid))
    scored.sort(reverse=True)
    return [mid for _, mid in scored[:k]]


def shipped(svc: MemoryService, cases, ids, k):
    """Score the real `_fuse` with the real defaults, on the same cached candidates.

    `rank()` above is a replica: it has to be, so retired strategies stay
    runnable after their settings are deleted. A replica can drift from the code
    it models, so this closes the loop — if the shipped implementation stops
    matching the strategy that was chosen, that shows up here as a gap.
    """
    buckets: dict[str, list[float]] = {}
    for case in cases:
        hits = svc._fuse(case["vrows"], case["lrows"], svc.settings.search_min_similarity,
                         k, relative_cutoff=svc.settings.search_relative_cutoff)
        ranked = [h.id for h in hits]
        want = ids[case["expected"]]
        rr = 1.0 / (ranked.index(want) + 1) if want in ranked else 0.0
        buckets.setdefault(case["kind"], []).append(rr)
        buckets.setdefault("all", []).append(rr)
    return {
        "top1": {kd: sum(1 for v in vs if v == 1.0) / len(vs) for kd, vs in buckets.items()},
        "mrr": {kd: sum(vs) / len(vs) for kd, vs in buckets.items()},
        "n": {kd: len(vs) for kd, vs in buckets.items()},
        "misses": [],
    }


def evaluate(svc, cases, ids, k):
    by_key = {v: kk for kk, v in ids.items()}
    results = {}
    for name, (strategy, blend) in STRATEGIES.items():
        buckets: dict[str, list[float]] = {}
        misses: list[tuple[str, str, str]] = []
        for case in cases:
            ranked = rank(svc, case, strategy, blend, k)
            want = ids[case["expected"]]
            rr = 1.0 / (ranked.index(want) + 1) if want in ranked else 0.0
            buckets.setdefault(case["kind"], []).append(rr)
            buckets.setdefault("all", []).append(rr)
            if rr != 1.0:
                got = by_key.get(ranked[0], "—") if ranked else "—"
                misses.append((case["query"], case["expected"], got))
        results[name] = {
            "top1": {kd: sum(1 for v in vs if v == 1.0) / len(vs) for kd, vs in buckets.items()},
            "mrr": {kd: sum(vs) / len(vs) for kd, vs in buckets.items()},
            "n": {kd: len(vs) for kd, vs in buckets.items()},
            "misses": misses,
        }
    return results


def report(results) -> None:
    kinds = ["conversational", "exact_token", "all"]
    n = next(iter(results.values()))["n"]
    head = f"{'strategy':<20}" + "".join(f"{kd.split('_')[0][:6]:>18}" for kd in kinds)
    print(head)
    print(f"{'':<20}" + "".join(f"{'top1 / mrr':>18}" for _ in kinds))
    print(f"{'':<20}" + "".join(f"{'(n=' + str(n[kd]) + ')':>18}" for kd in kinds))
    print("-" * len(head))
    for name, r in results.items():
        row = f"{name:<20}"
        for kd in kinds:
            row += f"{r['top1'][kd]:>10.0%} /{r['mrr'][kd]:>6.2f}"
        print(row)

    print("\nRemaining misses per strategy (query -> wanted / got):")
    for name, r in results.items():
        if not r["misses"]:
            print(f"  {name}: none")
            continue
        print(f"  {name}: {len(r['misses'])}")
        for q, want, got in r["misses"]:
            print(f"      {q!r} -> want {want}, got {got}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", required=True)
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    settings = Settings(database_url=args.database_url)
    svc = await MemoryService.create(settings)
    try:
        ids = await seed(svc)
        cases = await collect(svc, args.k)
        lex = sum(1 for c in cases if c["lrows"])
        print(f"corpus={len(ids)} queries={len(cases)} "
              f"(lexical candidates on {lex}/{len(cases)})\n")
        results = evaluate(svc, cases, ids, args.k)
        results["SHIPPED (real _fuse)"] = shipped(svc, cases, ids, args.k)
        report(results)
    finally:
        await svc.aclose()


if __name__ == "__main__":
    asyncio.run(main())
