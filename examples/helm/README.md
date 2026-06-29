# PKPS on the HELM suite

Real-data experiments for **Product Kernel Perspective Spaces (PKPS)** on a
heterogeneous 18-task, 93-model suite (MATH, WMT-14, MedQA, LegalBench). This
directory reproduces the two real-data results in the paper: query-efficient
evaluation and matrix completion. The PKPS method itself lives in the top-level
`dkps/` package (`dkps.unpaired_dkps.ProductKernelPerspectiveSpace`); this is its
application to HELM.

## Layout

```
data/          data acquisition and preparation
  download/      download-*.sh — pull the four HELM datasets
  parsers/       per-dataset response/score parsers
  extract.py     parse raw HELM dumps into per-task score + response tables
  embed_*.py     embed responses / queries (Gemini, one-hot)
  export_wmt.py  score WMT translations
pipeline/      core library (imported, not run directly)
  loaders.py       load the (model x task) score + embedding tensors
  perspective.py   PKPS embedding + matrix-completion crossfit
  crossval.py      leak-free joint-observation CV loader
  query_select.py  query-efficiency helpers (dense block, leave-one-out regression)
  baselines.py     IRT (1PL Rasch) and APW baselines
experiments/   runnable drivers
  query_efficiency.py          Result 1: predict the full score from m queries/cell
  completion.py                Result 2: predict never-run (model, task) cells
  dump_*_conditional.py        per-cell dumps for the conditional figures
figures/       paper-figure scripts (read result CSVs, write PDF/PNG)
tests/         test_leakage.py — leak-free check on the estimators
```

## Reproduce

Run everything from this directory (`examples/helm/`):

```bash
bash install.sh            # pixi environment + dependencies
bash run_experiments.sh    # all sweeps + per-cell dumps + figures (16 seeds)
```

`run_experiments.sh` writes result CSVs to `results-pkps-rd1/` (query efficiency)
and `results-pkps-unified/` (completion), then regenerates the figures. Individual
steps can be run directly, e.g.

```bash
python experiments/query_efficiency.py --sweep budget --n_seeds 16
python figures/query_efficiency.py
```

The synthetic study (paper Figure 2) is generated separately from
`experiments/unpaired/` at the repo root. Raw data, results, and embeddings are
large and git-ignored; see `data/download/` to fetch the HELM datasets.

---

## Algorithm card (DKPS)

**Team Name:** JHU &nbsp;·&nbsp; **Algorithm Name:** Data Kernel Perspective Space
(DKPS) — Performance Estimation &nbsp;·&nbsp; **Last Updated:** 2025-08-12

This algorithm predicts the performance of a new model `m_new` on a dataset in a
query-efficient way. We use the (precomputed) scored outputs from a large set of
models plus _unscored_ outputs from a new model to learn a regressor in DKPS space
that predicts the new model's performance on the entire dataset.

```python
def run_dkps_performance_estimation(
  models_old : list[LLM],
  dataset    : list[str],
  metric     : Metric,              # scoring metric for dataset
  model_new  : LLM,
  budget     : int,                 # query budget for new model
  embedder   : DenseEmbeddingModel, # embeds strings -> list[float]
  regressor  : Regressor,           # sklearn-style regressor
):
  # aggregate scores of old models on dataset (precomputed / pulled from HELM here)
  model_old_outputs       = [[model(d) for d in dataset] for model in models_old]
  model_old_output_scores = [[metric(o) for o in outs] for outs in model_old_outputs]
  model_old_agg_scores    = [np.mean(scores) for scores in model_old_output_scores]

  # run model_new on a budget-sized subset of dataset
  sel                      = np.random.choice(len(dataset), size=budget, replace=False)
  model_new_subset_outputs = [model_new(dataset[i]) for i in sel]

  # embed outputs of old models and model_new
  model_old_output_embs = [[embedder(o) for o in outs[sel]] for outs in model_old_outputs]
  model_new_output_embs = [embedder(o) for o in model_new_subset_outputs]

  # DKPS -> a low-dimensional embedding of each model
  model_embs = dkps(model_old_output_embs + model_new_output_embs)

  # regress agg scores on model embeddings, then predict model_new
  regressor.train(model_embs[:-1], model_old_agg_scores)
  return regressor.predict(model_embs[-1])
```

**Practical impact.** Useful where inference (running `model_new` on `dataset`) is
expensive, or scoring (`metric(o)`) is expensive or impossible (e.g. an expert
human judge).
