# Ranking eval

A labelled recall set for judging changes to how search ranks results, so
ranking is tuned from measurements rather than from intuition about what a
scoring formula "should" do.

```bash
docker run -d --name hm-eval-db -e POSTGRES_USER=hypermnesia \
  -e POSTGRES_PASSWORD=hypermnesia -e POSTGRES_DB=hypermnesia \
  -p 55432:5432 pgvector/pgvector:pg16

python -m evals.eval_ranking \
  --database-url postgresql://hypermnesia:hypermnesia@localhost:55432/hypermnesia
```

It seeds `corpus.py` into that scratch database, pulls the vector and lexical
candidate lists **once** per query through the production `_fetch_candidates`,
then replays every candidate strategy over those cached candidates. Identical
input for every strategy means a difference in the numbers is caused by ranking
alone — no embedding noise, no retrieval variance between runs.

Read the two columns together. `conversational` queries are natural-language
questions where meaning carries the query; `exact_token` queries are rare
identifiers where the embedding is weak and the keyword match is the whole
point. A change that improves one by sacrificing the other is not an
improvement, and splitting them is what makes that visible.

The last row, `SHIPPED (real _fuse)`, runs the actual implementation with the
actual defaults. It should match whichever modelled strategy was chosen; the
rest of the table is a replica that can drift from the code, so this is the row
that says the shipped thing is the thing that was measured.

## What it found

The strategies that look redundant are kept deliberately — they are the record
of what was ruled out. Two independent defects turned up, and only the split
table shows they are independent:

| | conversational top-1 | exact-token top-1 | MRR |
|---|---|---|---|
| pre-fix | 55% | 100% | 0.77 |
| bounded lexical bonus only | 60% | 100% | 0.82 |
| damped blend weights only | 60% | 100% | 0.81 |
| both (shipped) | 70% | 100% | 0.86 |

1. **Reciprocal-rank fusion discarded similarity magnitude.** Normalising the
   fused score against "rank 1 in both lists" gave any keyword match a flat
   ~0.5 relevance jump — larger than the entire spread of similarity scores.
   Replaced with cosine similarity plus a bounded, rank-decayed bonus.
2. **Recency and importance outweighed relevance.** Their weights were set
   against relevance's nominal `[0,1]` range, but embedding cosine is
   compressed: every plausible hit lands in ~0.55-0.73, a spread of ~0.18,
   while recency alone could move a score by 0.25. Damped to `0.10`/`0.06`.

The chosen lexical bonus sits on a plateau — anything from 0.05 to 0.30 scores
the same — so it is not a knife-edge constant fitted to this corpus.

## Caveats

Twenty memories and twenty-nine queries is small; roughly three queries account
for the conversational gain. The corpus is also written by the same hand that
wrote the fix, so it is better evidence for *ruling strategies out* than for the
absolute accuracy numbers. Several labels are simply beyond the embedding and
miss under every strategy — they dilute the percentages but do not favour any
strategy over another. Grow the corpus before treating a small delta here as
real.
