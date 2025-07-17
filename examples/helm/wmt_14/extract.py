#!/usr/bin/env python
"""
    wmt_14/extract.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rich import print as rprint

# --
# Parse args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',   type=str, default='./crfm-helm-public/lite/benchmark_output/runs/')
    parser.add_argument('--outpath', type=str, default='./wmt_14.tsv')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    
    return args

# --
# Helpers

args = parse_args()

inpaths = sorted(Path(args.indir).rglob('scenario_state.json'))

all_dfs = []
for inpath in inpaths:
    run     = json.loads(inpath.read_text())
    _params = {    
        "dataset" : inpath.parent.name.split(':')[0],
        **dict([x.split('=') for x in inpath.parent.name.split(':')[1].split(',')])
    }
    
    dps      = json.loads((inpath.parent / 'display_predictions.json').read_text())
    id2score = {x['instance_id']: x['stats']['bleu_4'] for x in dps}
    
    df_tmp = pd.DataFrame([{
        "dataset"     : _params['dataset'],
        "model"       : _params['model'],
        "instance_id" : _params['dataset'] + '--' + x['instance']['id'],
        "query"       : x['request']['prompt'],
        "response"    : x['result']['completions'][0]['text'],
        "score"       : id2score[x['instance']['id']],
    } for x in run['request_states']])
    
    all_dfs.append(df_tmp)

df = pd.concat(all_dfs)

# sometimes a record is missing some annotations - let's drop them for now
assert all(df.score.notna()), 'some instances are missing scores'
assert df.drop_duplicates(['dataset', 'model', 'instance_id']).shape[0] == df.shape[0]

# drop instance_ids with short responses
# !! For now, lets skip this
lens = df.groupby('instance_id').response.apply(lambda x: min([len(xx) for xx in x]))
bad  = lens[lens < 25]
df   = df[~df.instance_id.isin(bad.index)]

df = df.reset_index(drop=True)

df.to_csv(args.outpath, index=False, sep='\t')

# --------------------------------------------------------------------------------
# Gut check - are the scores on the two subtasks correlated?
# Answer    - yes

# tmp = df.groupby(['dataset', 'model']).score.mean().reset_index()
# tmp = tmp.pivot(index='model', columns='dataset', values='score')

# from matplotlib import pyplot as plt
# from rcode import *
# _ = plt.scatter(*tmp.values.T)
# _ = show_plot()