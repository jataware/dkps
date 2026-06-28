import sys; sys.path.insert(0, '/home/paperspace/projects/dkps/examples/helm')
from joblib import Parallel, delayed
import helm_doublekernel as H, pandas as pd
from helm_rd1_suite import run_seed
data = H.load_suite()
jobs = []
for n in [10, 40, 93]:                       # cohort, at m=2, full coverage
    jobs += [delayed(run_seed)(data, 2, s, n_models=n, p_task=1.0, n_paired=None, dump_cells=True) for s in range(8)]
for p in [0.2, 0.5, 0.9]:                     # coverage, at m=2, full cohort
    jobs += [delayed(run_seed)(data, 2, s, n_models=None, p_task=p, n_paired=None, dump_cells=True) for s in range(8)]
rows = [r for sub in Parallel(n_jobs=-1)(jobs) if sub for r in sub]
pd.DataFrame(rows).to_csv('/home/paperspace/projects/dkps/examples/helm/results-pkps-rd1/qe_cells_levers.csv', index=False)
print('wrote qe_cells_levers.csv', len(rows), 'rows; n_models', sorted(set(r['n_models'] for r in rows)),
      'p_task', sorted(set(r['p_task'] for r in rows)))
