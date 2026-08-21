# dkps — Perspective Spaces for Black-Box Model Comparison

This repository contains the **DKPS/PKPS method code** and the experiments for
two papers:

1. **PKPS paper** — *Predicting Benchmark Scores from Sparse and Unpaired
   Cached Responses* (`paper/main.tex`): the Product Kernel Perspective Space,
   a generalization of DKPS to unpaired evaluation data, with query-efficiency
   and matrix-completion results on two benchmark suites.
2. **Routing paper** (in progress, `experiments/routing/`): query-aware model
   routing from cached, unpaired, unlabeled evaluation history — selecting a
   behavioral surrogate for an unavailable model.

## The package

`dkps` exposes every method in the PKPS paper as an estimator with the same
three-call API:

| class | estimates a (model, task) score from | paper role |
|---|---|---|
| `PKPS` | the model's cached responses (any queries) + other models' scores | the method |
| `DKPS` | same, but only responses to *identical* queries | paired baseline |
| `SampleScore` | the mean of the cell's own scored responses | score-only baseline |
| `IRT` | 1PL item-response model on binary per-response scores | query-efficiency baseline |
| `LRMC` | low-rank completion of the sample-score matrix | completion baseline |
| `Ensemble` | a learned blend of any two of the above | the reported ensemble |

```python
est = PKPS(...).fit(records)        # embed (if needed), reduce, build the perspective space
est.predict(records)                # -> [{'model_id', 'task_id', 'score_hat'}, ...]
est.update(records)                 # add models / responses; only affected pairs recomputed
```

**Records** are JSON-friendly tables -- a DataFrame, a list of dicts, a dict of
lists, or a JSON string of either -- with one row per cached response:

| column | required | meaning |
|---|---|---|
| `model_id`, `task_id`, `query_id` | yes | which model answered which query of which task |
| `response_embedding` *or* `response` | yes | precomputed vector, or raw text to embed |
| `query_embedding` *or* `query` | PKPS only | precomputed vector, or raw text (DKPS needs neither) |
| `score` | for prediction | the per-response score |
| `reference_score` | optional | a known full-benchmark score for the cell, used as the regression target for *other* models |
| `sample_score` | optional | overrides the per-cell mean of `score` (cells scored on more responses than were embedded) |

Raw text is embedded through `dkps.embed` (`embedding_kwargs=dict(provider=
'google', model=None, api_key=None)`; the key falls back to the provider's
environment variable, `GEMINI_API_KEY` for Google).

### PKPS in one paragraph

For models `i, i'` with response embeddings `x` to queries `u`,

```
A_{ii'} = sum_{j,l} k_Q(u_j, u_l) k_R(x_ij, x_i'l) / sum_{j,l} k_Q(u_j, u_l)
D^2_{ii'} = A_ii + A_i'i' - 2 A_ii'
```

The query kernel `k_Q` decides which response pairs are comparable and how
much; the self-normalization keeps models with different numbers of queries
comparable. `k_Q = delta` (identical queries only) recovers paired DKPS, which is
what `DKPS` is. Classical MDS of `D` gives each model a coordinate; a per-task
k-NN regressor from those coordinates onto the other models' scores predicts the
target cell. Every knob is a constructor argument:

```python
PKPS(
    query_kwargs=dict(kernel='rbf',            # 'rbf' | 'delta' | 'linear' | 'cosine' | callable
                      bandwidth='cv',          # 'median' | 'cv' (leave-one-model-out grid search) | float
                      bandwidth_ref=None,      # reference scale (None = median query distance)
                      pca_dim='elbow'),        # 'elbow' (Zhu-Ghodsi) | int | None
    response_kwargs=dict(kernel='linear', pca_dim='elbow'),
    mds_kwargs=dict(dim=8),                    # int, or None for Zhu-Ghodsi selection
)
est.predict(cells, k=5,
            holdout='model',                   # or 'family': leave-one-family-out (query-efficiency protocol)
            whiten=False)                      # True: logit/standardize/bias residual regression (completion protocol)
```

### Quick start (synthetic data, runs anywhere)

```python
import numpy as np
from dkps import PKPS, LRMC, Ensemble, generate_benchmark_data

data, scores, observed, _, _ = generate_benchmark_data(
    d_latent=5, d_obs=20, n_models=40, n_tasks=8, n_queries_per_task=16,
    obs_prob=0.6, random_state=0)                       # rows only for observed cells
mi = data.model_id.str[6:].astype(int); ti = data.task_id.str[5:].astype(int)
data['score'] = 1 / (1 + np.exp(-scores[mi, ti]))       # per-response scores in (0, 1)

pkps = PKPS().fit(data)
lrmc = LRMC().fit(data)
ens = Ensemble([pkps, lrmc], mode='cv', holdout=None,
               predict_kwargs=[{'whiten': True}, {}]).fit(data)
missing = ens.predict()                                  # every cell without an observed score
```

### Reproducing a paper result

`examples/helm/example_table1.py` reproduces one cell of Table 1 end to end with
the classes above -- HELM suite, query-efficient evaluation at `m = 1` query per
cell, seed 0:

```bash
cd examples/helm
pixi run python example_table1.py --m 1 --seed 0
#  sample score  0.302
#  IRT           0.360   (binary tasks only)
#  DKPS          0.161
#  PKPS          0.137
#  Ensemble      0.128
```

(Table 1 reports the 16-seed means: 0.292 / 0.348 / 0.168 / 0.136 / 0.125.)
The script is ~60 lines: load the suite, sample `m` responses per cell, build
records, fit the five estimators, score against the full-benchmark scores. The
two protocol choices it makes -- `bandwidth='cv'` around the within-domain
median query distance, and `holdout='family'` -- are the paper's.

`examples/helm/validate_package.py` does this for **every** Table 1 operating
point on both suites (query efficiency at `m ∈ {1, 8}`; completion at the three
cohort/coverage settings), 16 seeds each, replaying the experiment scripts'
sampling and comparing per-(seed, task, method) MAEs to the result CSVs those
scripts produced. All rows agree to 0.

### Tests

```bash
pixi run pytest tests            # 14 tests: API, incremental update == fresh fit,
                                 # equality with the pipeline functions, leakage
```

## Repository layout

```
dkps/                    the package (see above)
tests/                   package tests (pixi run pytest tests)
paper/                   PKPS paper LaTeX + final figure PDFs (pdflatex main.tex)
examples/helm/           PKPS paper: real-data pipeline for BOTH suites
                         (HELM 18 tasks x 93 models; EEE 16 tasks x 45 models)
                         -> has its own README with layout + reproduce steps;
                         example_table1.py and validate_package.py live here
experiments/unpaired/    PKPS paper: synthetic study (paper Fig. 2)
experiments/routing/     Routing paper: all experiments -> has its own README
```

Conventions used throughout:

- **Results stay local.** `results*/`, `exports/`, embeddings, and raw data are
  git-ignored; only code and the paper's final figure PDFs are committed.
- **Leak-free evaluation.** All hyperparameters (incl. the PKPS query
  bandwidth) are selected by cross-validation on observed/anchor data only;
  `examples/helm/tests/test_leakage.py` verifies this by corrupting held-out
  scores. The routing code asserts the analogous leave-query-out invariant.
- Environment: [pixi](https://pixi.sh) — `pixi install`, then run scripts with
  `pixi run python ...`. Loaders in `examples/helm` use paths relative to that
  directory; run them from there (the routing scripts handle this themselves).

## Quick starts

```bash
pixi install

# PKPS paper, real data (from examples/helm/):
bash run_experiments.sh            # all sweeps + figures, 16 seeds

# PKPS paper, synthetic study (from repo root):
pixi run python -m experiments.unpaired.run_block1

# Routing paper (from repo root):
pixi run python -m experiments.routing.run_helm       # paired-suite routing
pixi run python -m experiments.routing.run_combined   # combined unlabeled pool
```

## Data requirements

The pipelines read cached artifacts that are not in git (large / paid):

- `examples/helm/exports/` — response & query embedding parquets for both
  suites (Gemini `gemini-embedding-001`, plus one-hot blocks for
  multiple-choice datasets). Built by `examples/helm/data/` scripts from the
  raw HELM dumps / EEE datastore; requires `GEMINI_API_KEY` to rebuild.
- `examples/helm/data/` — raw downloads (see `data/download/*.sh` and
  `data/eee.py` for the EEE datastore).

If you are on the existing machine, everything is already in place; otherwise
budget ~2 GB of embeddings (cached, resumable) to rebuild `exports/`.
