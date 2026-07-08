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

## Orientation for DKPS veterans

If you know the original DKPS code, the map from old to new:

- `DataKernelPerspectiveSpace` (paired DKPS) still lives in `dkps/dkps.py`,
  unchanged.
- The new method is `dkps.unpaired_dkps.ProductKernelPerspectiveSpace`
  (aliases: `PKPS`, and `DoubleKernelDKPS` for back-compat with early
  notebooks). It replaces DKPS's identity query-matching with a **product
  kernel**: the comparison between models `i, i'` is

  ```
  A_{ii'} = sum_{j,l} k_Q(q_j, q_l) k_R(x_ij, x_i'l) / sum_{j,l} k_Q(q_j, q_l)
  ```

  self-normalized per pair, so models that answered different (numbers of)
  queries remain comparable. `k_Q = δ` (or an RBF with σ→0) recovers paired
  DKPS exactly; distances `D² = A_ii + A_i'i' − 2A_ii'` feed classical MDS as
  always. The paper's instantiation is an RBF query kernel (bandwidth chosen
  per run by leak-free CV) with a linear response kernel.
- Baselines used by the paper are exposed from the package:
  `matrix_completion_predict` (logit-space rank-2 ALS, BenchPress-style) and
  the 1PL IRT trio (`irt_fit_difficulties`, `irt_estimate_ability`,
  `irt_predict`).
- `dkps/embed.py` is the Gemini embedding client (disk-cached, 429-retry);
  `dkps/synthetic.py` generates the synthetic benchmark.

## Repository layout

```
dkps/                    method package (see above)
paper/                   PKPS paper LaTeX + final figure PDFs (pdflatex main.tex)
examples/helm/           PKPS paper: real-data pipeline for BOTH suites
                         (HELM 18 tasks x 93 models; EEE 16 tasks x 45 models)
                         -> has its own README with layout + reproduce steps
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
