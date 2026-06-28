import sys; sys.path.insert(0, '/home/paperspace/projects/dkps/examples/helm')
from joblib import Parallel, delayed
import pandas as pd
from helm_rd2_cv import load
from helm_completion_suite import trial
data, qmed, suite = load('suite')
jobs = [delayed(trial)(data, qmed, 93, None, p, s, p_query=0.5, dump_cells=True)
        for p in [0.2, 0.5, 0.9] for s in range(8)]
rows = [r for sub in Parallel(n_jobs=-1)(jobs) if sub for r in sub]
pd.DataFrame(rows).to_csv('/home/paperspace/projects/dkps/examples/helm/results-pkps-unified/completion_cells_cov.csv', index=False)
print('wrote completion_cells_cov.csv', len(rows), 'rows; p_task', sorted(set(r['p_task'] for r in rows)))
