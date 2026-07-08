# Query-aware model routing

Standalone paper (separate from the PKPS paper; shares the `dkps` machinery
and the suites prepared in `examples/helm/`).

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

For each held-out evaluation query and each target model, the routed model's
cached response is scored against the target's cached response in embedding
space ("mimicry error" — behavioral fidelity, not correctness):

    error(r; t, q*) = ||x_{r,q*} - x_{t,q*}||

**Leave-query-out, asserted in code:** no model's responses to an evaluation
query may enter any geometry, score estimate, or overlap index. Splits are
stratified by task; every table is bracketed by `random` and `oracle`
(hindsight-best — a floor for any router, inflated by extreme-value selection
when candidate pools are large).

Suites (loaded via `examples/helm/pipeline/loaders.py`; Gemini embeddings):

- **HELM paired core** — 93 models x 1529 queries (18 tasks), fully paired.
  The per-query pairwise squared-distance tensor P (m, n, n) is small (53 MB)
  and every weighted geometry — any bandwidth, static included — is a single
  tensordot over it.
- **EEE** — 45 models x 6808 queries (16 tasks), unpaired by construction
  (~10% incidental overlap). No paired tensor exists; the rank-one localized
  product kernel factorizes under a linear response kernel into distances
  between each model's k_Q(·,q*)-weighted mean of its OWN pool
  (`batched_localized_stats`) — no shared queries needed. Evaluation on
  queries with >= 3 responders.
- **Combined pool** — the union as ONE unlabeled task: 138 models, 8457
  queries, 34 hidden tasks; joint PCA of the shared raw query space;
  block-padded response spaces (within-suite errors unchanged).

Methods and baselines:

1. `qa-{f}x`   — nearest to target in the query-weighted geometry (RBF query
   kernel, sigma = f x median pairwise anchor-query distance; selectable
   leak-free by anchor pseudo-evaluation CV, `bandwidth_cv.py`)
2. `task*`     — nearest in the task-indicator geometry (uses hidden labels;
   reported as a reference ceiling, not a deployable router)
3. `static`    — nearest in the uniform-weight (static DKPS) geometry
4. `profile`   — score-profile routing: nearest per-task anchor score profile
   (the real no-embedding baseline — what a leaderboard alone can do)
5. `random` / `oracle` — brackets; the target itself is the metric's zero
6. tabular outcome-regression *ablation* (`run_helm_rf.py`, `rf-*` in
   `run_eee.py`): kernel regression of realized pairwise errors on the query
   embedding. Identical to `qa` on paired data by algebraic identity; starves
   on unpaired overlap. An estimation-structure ablation, not a baseline —
   its labels derive from response embeddings.

## Modules

```
geometry.py             per-query distance tensor; weighted geometries; RBF weights
router.py               routing rules; mimicry / random / oracle errors
evaluate.py             paired-suite protocol (splits, sigma grid, task baseline)
bandwidth_cv.py         leak-free CV bandwidth selection (global & per-query)
data.py                 jailbreak-suite loader (original scoping; data on the a100)

run_helm.py             HELM paired core
run_helm_cv.py          + CV bandwidths, finer sigma grid
run_helm_rf.py          outcome-regression ablation (paired identity check)
run_eee.py              EEE: mimicry + score routing (batched; ~12 s/seed)
run_combined.py         HELM+EEE union as one unlabeled pool (headline)
run_profile_baseline.py score-profile baseline
dim_sweep.py            response/query dimension sweep (metric fixed full-dim)
query_dim_fine.py       low-d query PCA hypothesis test (per-dim bandwidths)
run_*_scorediff.py      side-study: |score difference| objective
```

All runnable from the repo root: `pixi run python -m experiments.routing.<name>`.
Outputs land in `results/` here (git-ignored): per-decision parquets + summaries.

## Results (mimicry error, lower is better; 5 seeds)

Combined pool, all task/suite labels hidden from routers:

| method                     | overall | HELM side | EEE side |
|----------------------------|--------:|----------:|---------:|
| oracle (hindsight)         |   0.183 |     0.083 |    0.400 |
| **qa-0.25x (label-free)**  | **0.378** | **0.308** |    0.531 |
| task* (hidden labels)      |   0.384 |     0.316 |  **0.531** |
| static DKPS                |   0.421 |     0.359 |    0.553 |
| random                     |   0.548 |     0.491 |    0.671 |

Standalone suites: HELM qa 0.295 / task 0.299 / static 0.342 / profile 0.402 /
random 0.519, oracle 0.089; EEE qa 0.532 / task 0.531 / static 0.555 /
profile 0.617 / random 0.670, oracle 0.400. Score routing on EEE (realized
score of pick, higher better): oracle 0.978, task 0.904, qa 0.900,
static 0.898, random 0.808.

## Findings

1. **Label-free routing matches — and on the merged pool beats — task-labeled
   routing.** The kernel discovers task granularity and its soft boundaries
   with no metadata; merging heterogeneous caches costs nothing (the
   factorized geometry is per-model and cannot be cross-contaminated).
2. **Leaderboards cannot pick behavioral surrogates.** The score-profile
   baseline loses badly on both suites: models with similar score profiles do
   not behave similarly. Response-level information carries the entire gap.
3. **The pairing story replays at the routing level.** Paired, the qa
   geometry *is* kernel regression of squared mimicry error (verified
   identity); unpaired, per-pair regression starves (~27 shared anchors/pair)
   while the per-model factorization keeps the full gain.
4. **The oracle gap is structural, not tunable.** Task-mean error is the
   entire predictable component: after removing task means, error surfaces of
   embedding-adjacent queries are uncorrelated (~0.01 at every query
   dimension, 2 -> 3072). Bandwidth (flat optimum, CV lands on it), response
   dimension (8 -> full: 0.308 -> 0.301), and query dimension (low-d loses
   even with per-dimension bandwidths) are all saturated.
5. **Boundary of the factorization** (score-difference side-study): score
   *agreement* is irreducibly pairwise — E|Δs| depends on correctness
   co-occurrence that marginal means cannot express (Jensen) — so there the
   pair's historical disagreement rate routes best. Behavioral surrogacy
   factorizes; score-agreement surrogacy does not.

## Recommended configuration

Query dim 12–40 (PCA of the shared query space), one global sigma ~0.25x the
median anchor distance (or CV-selected), nearest-candidate pick. A full suite
evaluation runs in about a minute (batched per-model matmuls, validated to
<= 5e-5 against the reference implementation).

## Original scoping (jailbreak suite)

`data.py` / `embed_queries.py` / `precompute.py` / `run_eval.py` target the
jailbreak-dkps suite (82 models x 2666 attacks, nomic embeddings; data lives
on the a100 box). `geometry.py`, `router.py`, `evaluate.py` are
dataset-agnostic — a new paired suite needs only `X` (n, m, d) responses,
`query_emb` (m, d_q), and per-query `categories`.
