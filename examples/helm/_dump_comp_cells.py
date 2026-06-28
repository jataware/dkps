import sys; sys.path.insert(0, '/home/paperspace/projects/dkps/examples/helm')
from joblib import Parallel, delayed
import pandas as pd
from helm_rd2_cv import load
from helm_completion_suite import trial
data, qmed, suite = load('suite')
jobs = [delayed(trial)(data, qmed, n, None, 0.5, s, p_query=0.5, dump_cells=True)
        for n in [10, 40, 93] for s in range(8)]
rows = [r for sub in Parallel(n_jobs=-1)(jobs) if sub for r in sub]
pd.DataFrame(rows).to_csv('/home/paperspace/projects/dkps/examples/helm/results-pkps-unified/completion_cells.csv', index=False)
print('wrote completion_cells.csv', len(rows), 'rows')
