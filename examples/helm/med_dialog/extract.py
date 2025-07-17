#!/usr/bin/env python
"""
    extract-med_dialog.py
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
    parser.add_argument('--indir',   type=str, default='./crfm-helm-public/medhelm/benchmark_output/runs/v2.0.0/')
    parser.add_argument('--outpath', type=str, default='./med_dialog.tsv')
    args   = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    
    return args

# --
# Helpers

def _extract_score(x):
    try:
        judges = ['gpt', 'llama', 'claude']
        scores = [[xx['score'] for xx in x['annotations']['med_dialog'][judge].values()] for judge in judges]
        return np.mean(np.concatenate(scores))
    except:
        # rprint(f'[red] !! error @ _extract_score : {x['instance']['id']} [/red]')
        return None


args = parse_args()

inpaths = sorted(Path(args.indir).rglob('scenario_state.json'))

all_dfs = []
for inpath in inpaths:
    run     = json.loads(inpath.read_text())
    _params = {    
        "dataset" : inpath.parent.name.split(':')[0],
        **dict([x.split('=') for x in inpath.parent.name.split(':')[1].split(',')])
    }
    
    df_tmp = pd.DataFrame([{
        "dataset"     : _params['dataset'],
        "model"       : _params['model'],
        "instance_id" : _params['dataset'] + '--' + x['instance']['id'],
        "query"       : x['request']['prompt'],
        "response"    : x['result']['completions'][0]['text'],
        "score"       : _extract_score(x),
    } for x in run['request_states']])

    all_dfs.append(df_tmp)

df = pd.concat(all_dfs)

# sometimes a record is missing some annotations - let's drop them for now
bad_ids = set(df.instance_id[df.score.isna()].values)
rprint(f'[red]!! dropping {len(bad_ids)} instances because they are missing gpt and llama annotations for >= 1 run[/red]')
df      = df[~df.instance_id.isin(bad_ids)]

# bunch of duplicate entries in icliniq ... opened issue on HELM.  not a big deal for our purposes really.
df = df.sort_values(['dataset', 'model', 'instance_id']).drop_duplicates(['dataset', 'model', 'query'])

# drop instance_ids with short responses
bad = df.groupby('instance_id').response.apply(lambda x: min([len(xx) for xx in x]))
bad = bad[bad < 50]
df  = df[~df.instance_id.isin(bad.index)]

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