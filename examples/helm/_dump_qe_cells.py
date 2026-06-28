import sys; sys.path.insert(0, '/home/paperspace/projects/dkps/examples/helm')
from joblib import Parallel, delayed
import helm_doublekernel as H, pandas as pd
from helm_rd1_suite import run_seed
data = H.load_suite()
jobs = [delayed(run_seed)(data, m, s, n_models=None, p_task=1.0, n_paired=None, dump_cells=True)
        for m in [1, 2, 4, 8] for s in range(8)]
rows = [r for sub in Parallel(n_jobs=-1)(jobs) if sub for r in sub]
pd.DataFrame(rows).to_csv('/home/paperspace/projects/dkps/examples/helm/results-pkps-rd1/qe_cells.csv', index=False)
print('wrote qe_cells.csv', len(rows), 'rows')
