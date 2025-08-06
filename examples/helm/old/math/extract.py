#!/usr/bin/env python
"""
    math/extract.py
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from seaborn import pairplot

# --
# Parse args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',   type=str, default='./crfm-helm-public/lite/benchmark_output/runs/')
    parser.add_argument('--outpath', type=str, default='./math.tsv')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    
    return args

# --
# Run

args = parse_args()

inpaths = sorted(Path(args.indir).rglob('scenario_state.json'))

all_dfs = []
for inpath in tqdm(inpaths, desc='loading data'):
    run     = json.loads(inpath.read_text())
    _params = {    
        "dataset" : inpath.parent.name.split(',')[0],
        **dict([x.split('=') for x in inpath.parent.name.split(':')[1].split(',')])
    }
    
    dps      = json.loads((inpath.parent / 'display_predictions.json').read_text())
    id2score = {x['instance_id']: x['stats']['math_equiv_chain_of_thought'] for x in dps}
    
    df_tmp = pd.DataFrame([{
        "dataset"     : _params['dataset'],
        "model"       : _params['model'],
        "instance_id" : _params['dataset'] + '--' + x['instance']['id'],
        "query"       : x['request']['prompt'],
        "response"    : x['result']['completions'][0]['text'],
        "target"      : x['instance']['references'][0]['output']['text'],
        "score"       : id2score[x['instance']['id']],
    } for x in run['request_states']])
    
    all_dfs.append(df_tmp)


df = pd.concat(all_dfs)

# sometimes a record is missing some annotations - let's drop them for now
assert all(df.score.notna()), 'some instances are missing scores'
assert df.drop_duplicates(['dataset', 'model', 'instance_id']).shape[0] == df.shape[0]
df = df.reset_index(drop=True)

df.to_csv(args.outpath, index=False, sep='\t')

# with open('math.jsonl', 'w') as f:
#     for _, row in df.iterrows():
#         _ = f.write(json.dumps({
#             'text': row.response,
#         }) + '\n')