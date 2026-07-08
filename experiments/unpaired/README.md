# Synthetic study (PKPS paper, Figure 2)

Generates the synthetic benchmark of the PKPS paper and sweeps every lever of
the observation process: cohort size n, number of tasks T, task coverage
p_task, query depth p_query, cross-model overlap rho, and the denoising panel
(queries per cell at rho in {0, 1}). PKPS (RBF query kernel, CV bandwidth) vs
DKPS (identity kernel) vs the sample score.

```bash
# from the repo root
pixi run python -m experiments.unpaired.run_block1        # all panels, 50 seeds
```

Results and the figure land in `experiments/unpaired/results/paper/`
(`fig_synthetic.pdf` is copied into `paper/figures/`). To re-plot from the
saved CSVs without re-running:

```python
from pathlib import Path
import pandas as pd
from experiments.unpaired.plots import save_figure
d = Path('experiments/unpaired/results/paper')
save_figure(d, {p.stem: pd.read_csv(p) for p in d.glob('*.csv')})
```

`block1.py` holds the data-generating process and estimators; `plots.py` the
figure. The method itself is `dkps.unpaired_dkps.ProductKernelPerspectiveSpace`.
