"""Guards the ranking eval's labels against rot.

The eval only means anything if every query points at a memory that is actually
in the corpus, and if both query regimes stay represented — a corpus that
quietly lost its exact-token queries would report a fusion change as a clean win
while hiding the recall it broke. None of this needs a database.
"""

from __future__ import annotations

from evals.corpus import CORPUS, QUERIES

KINDS = {"conversational", "exact_token"}


def test_corpus_keys_are_unique():
    keys = [c[0] for c in CORPUS]
    assert len(keys) == len(set(keys))


def test_every_query_expects_a_real_corpus_entry():
    keys = {c[0] for c in CORPUS}
    unknown = {expected for _, expected, _ in QUERIES if expected not in keys}
    assert not unknown, f"queries expect memories that aren't in the corpus: {unknown}"


def test_query_kinds_are_known_and_both_populated():
    kinds = [kind for _, _, kind in QUERIES]
    assert set(kinds) <= KINDS, f"unknown query kinds: {set(kinds) - KINDS}"
    for kind in KINDS:
        assert kinds.count(kind) >= 5, f"too few {kind} queries to measure that regime"


def test_exact_token_queries_actually_appear_in_their_target():
    """An exact-token query that no longer matches its memory silently becomes a
    semantic query, and the lexical half of the table stops testing anything."""
    body = {key: f"{desc} {content}".lower() for key, desc, content, _, _ in CORPUS}
    for query, expected, kind in QUERIES:
        if kind == "exact_token":
            assert query.lower() in body[expected], (
                f"exact-token query {query!r} no longer appears in {expected!r}"
            )
