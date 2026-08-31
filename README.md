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

## For Every Eval Ever users

The [EEE datastore](https://huggingface.co/datasets/evaleval/EEE_datastore) already
holds what PKPS runs on: raw model responses with per-item scores. `dkps.eee` reads
the store's native `*_samples.jsonl` runs directly -- your uploaded run is already in
the right format, and no preprocessing pipeline is needed.

```bash
git clone <this repo> && cd dkps && pip install -e .    # or: pixi install
export GEMINI_API_KEY=...                               # embeds query/response text
```

`fit` itself does not embed anything: rows that carry raw `query`/`response` text are
sent through the configured embedding backend (`embedding_kwargs=dict(provider=,
model=, api_key=)`; default Gemini via `GEMINI_API_KEY`, results disk-cached), and
`provider='sentence-transformers'` embeds locally with no API key (requires the
`sentence-transformers` package). Rows that already carry
`response_embedding`/`query_embedding` vectors skip the backend entirely.

**Score a model from a handful of queries.** Fit on cached runs across several
benchmarks; every model in the fit gets a prediction for every task, using only the
responses it happens to have:

```python
from dkps import PKPS
from dkps.eee import fetch_samples, load_records

paths = fetch_samples(['gsm-mc', 'gpqa-diamond', 'math-mc'])   # newest run per model, cached
records = load_records(paths)          # model/task/query/text/score rows, 32 queries per cell
est = PKPS().fit(records)              # embeds the text, builds the perspective space
est.predict([{'model_id': 'openai/gpt-oss-20b'}])       # -> score_hat for every task
```

**Complete benchmarks a model never ran.** A model that answered *none* of a task's
queries is still embedded from its responses elsewhere; `est.predict()` with no
arguments returns every (model, task) cell that has no observed score. For the
paper's strongest completion numbers, blend with the matrix-completion baseline:

```python
from dkps import LRMC, Ensemble
ens = Ensemble([PKPS(), LRMC()], mode='cv', holdout=None,
               predict_kwargs=[{'whiten': True}, {}]).fit(records)
missing = ens.predict()
```

**Add your own run.** Point `load_records` at your local `*_samples.jsonl` files (or
directories in the store's `benchmark/developer/model/` layout), and fold new runs
into a fitted space without re-embedding the cache:

```python
mine = load_records(['my_model_run_samples.jsonl'], model_from='record')
est.update(mine)                       # only affinities involving your model recompute
est.predict([{'model_id': mine.model_id.iloc[0]}])
```

Practical notes:

- **Fit across several benchmarks.** The method borrows strength across models and
  tasks; a single benchmark with few models predicts poorly. The adapter emits a
  `suite` column and `PKPS.fit` then reduces responses per benchmark into disjoint
  unit-normalized blocks (the paper's construction) automatically.
- **Or use the whole store.** `fetch_samples('all')` pulls every benchmark, and the
  adapter handles the store's full heterogeneity: single-turn `output.raw` and
  multi-turn transcripts (arena and agent runs -- the model's own turns become the
  response), per-benchmark `evaluation_id` conventions, and repeat uploads. On the
  complete store (896 runs, 36 benchmarks) this yields 164 models x 46 tasks and a
  fit in about a minute; every unevaluated (model, task) cell gets a prediction.
  Keep expectations calibrated to coverage: 59% of model pairs share no benchmark,
  and for those pairs the response channel carries no direct signal -- predictions
  travel through the models that connect them.
- **Cost.** Embedding is the only paid step and is cached on disk: the paper's full
  five-benchmark suite (~38k texts, ~11M tokens) cost under \$2 and ~13 minutes
  through `gemini-embedding-001`; other providers plug in via
  `PKPS(embedding_kwargs=dict(provider=..., api_key=...))`.
- **Query pools.** `load_records` caps each (model, task) cell at 32 queries (the
  paper's setting) because PKPS cost grows with the product of two models' per-cell
  counts; pass `max_queries_per_cell=None` to keep full depth.
- **Naming.** The store's directory layout carries normalized model ids
  (`cohere/c4ai-command-a-03-2025`) while in-file `model_id` is the raw repo name
  (`CohereLabs/...`); the adapter prefers the path form so runs of the same model
  line up. It also parses the store's string scores, `is_correct` fallback, and list
  outputs, and resolves duplicate runs to the newest.
- **Sanity check on the live store**: a joint fit over five benchmarks (69 models
  x 16 tasks at 8 queries per cell) predicts held-out-family sample scores at MAE
  0.120 versus 0.153 for the per-task mean. The paper's Table 1 reports the full
  evaluation on a 45-model, 16-task submatrix.

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
