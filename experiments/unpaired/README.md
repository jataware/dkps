# Block I Synthetic Experiments

Run the paper's synthetic Block I experiments from the repo root:

```bash
bash experiments/unpaired/run.sh
```

For a faster smoke run:

```bash
bash experiments/unpaired/run.sh quick
```

Or run one experiment directly:

```bash
python -m experiments.unpaired.run_block1 --experiment s0 --preset paper
python -m experiments.unpaired.run_block1 --experiment s1 --preset paper
python -m experiments.unpaired.run_block1 --experiment s2 --preset paper
```

Outputs are written under `experiments/unpaired/results/<preset>/`:

- `s0_results.csv`
- `s1_summary.csv`
- `s1_search.csv`
- `s2_results.csv`
- `plots/`

For the exact-recovery sensitivity sweeps discussed alongside the paper runs:

```bash
python -m experiments.unpaired.block0 --preset quick
python -m experiments.unpaired.block0 --preset paper
```

Those outputs are written under `experiments/unpaired/results/block0/<preset>/`:

- `block0_results.csv`
- `plots/block0_mse.png`
- `plots/block0_max_abs_error.png`

For a 2D visual walkthrough of why the synthetic estimators behave the way they do:

```bash
python -m experiments.unpaired.walkthrough --preset default
python -m experiments.unpaired.walkthrough --preset quick
```

Those outputs are written under `experiments/unpaired/results/walkthrough/` by default:

- `walkthrough.md`
- `case_metrics.csv`
- `alpha_sweep_no_coverage.csv`
- `unpaired_all_vs_strict_nonshared.csv`
- `plots/`
