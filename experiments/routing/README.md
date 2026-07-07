# Query-aware model routing

Route a new query to the model that best mimics a target model, using a
behavioral geometry localized to the query.

Given n models with cached responses to m anchor queries, and a new query q*,
model-model distances are the DKPS response distances with each anchor query's
contribution weighted by a query kernel k_Q(q_j, q*) — the PKPS product-kernel
construction, centered on the new query. We route q* to the candidate model
nearest the target model in this geometry. As the query-kernel bandwidth grows
the geometry collapses to static DKPS, so the method and the static baseline
live on one continuum.

## Evaluation (offline, cached responses)

Suite: jailbreak-dkps — 82 models x 2666 queries, responses and queries
embedded with nomic-embed-text-v1.5. Queries are split into anchors (the
cached history) and held-out evaluation queries (the "new" queries). For each
evaluation query and each target model, the routed model's cached response is
scored against the target's cached response in embedding space:

    error(r; t, q*) = ||x_{r,q*} - x_{t,q*}||

Methods and baselines:

1. `qa-{f}x` — nearest to target in the query-weighted geometry (RBF query
   kernel, sigma = f x median pairwise anchor-query distance)
2. `static`  — nearest to target in the uniform-weight (static DKPS) geometry
3. `random`  — expected error of a uniform-random candidate
4. `oracle`  — hindsight-best candidate (floor for any router)
5. the target itself is the metric's zero by construction

The key computational trick: for a paired suite the per-query pairwise
squared-distance tensor P (m, n, n) is small (72 MB here), and every weighted
geometry — any bandwidth, static included — is a single tensordot over it, so
the full sweep (all targets x all held-out queries x bandwidth grid x seeds)
runs in seconds.

## Porting to another suite (e.g. full HELM)

`geometry.py`, `router.py`, and `evaluate.py` are dataset-agnostic; only
`data.py` knows about a specific suite. A new suite needs three arrays:

- `X` — (n_models, m_queries, d) response embeddings (paired design)
- `query_emb` — (m_queries, d_q) query embeddings (only used inside k_Q)
- `categories` — per-query task/scenario labels (used for stratified splits
  and per-task analysis)

For HELM, the adapter in `examples/helm` plus `dkps/embed.py` already pools
responses and queries.

## Pipeline

From the repo root:

```bash
python -m experiments.routing.embed_queries   # query embeddings (cached to data/)
python -m experiments.routing.precompute      # per-query distance tensor (m, n, n)
python -m experiments.routing.run_eval        # -> results/routing_errors.parquet, summary.csv
```

Smoke test (seed 0, 82 models, 533 held-out queries, every model as target):
oracle 6.48, qa-0.25x 9.92, static 10.09, random 14.80 — query-aware routing
beats static at moderate bandwidth, and qa-2x reproduces static, as the
continuum predicts.
