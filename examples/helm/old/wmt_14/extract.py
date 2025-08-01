#!/usr/bin/env python
"""
    wmt_14/extract.py
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from seaborn import pairplot

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')

from sacrebleu.metrics import CHRF
# --
# Helpers

def bleu_4(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0, 0, 0, 1))

def bleu_flat(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0.25, 0.25, 0.25, 0.25))

def chrf(gold: str, pred: str) -> float:
    metric = CHRF(word_order=2) # chrF++; set word_order=0 for plain chrF
    return metric.sentence_score(pred, [gold]).score

def meteor(gold: str, pred: str) -> float:
    return meteor_score([word_tokenize(gold)], word_tokenize(pred))   # returns 0‑1; multiply by 100 if you prefer %

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
# Run

args = parse_args()

inpaths = sorted(Path(args.indir).rglob('scenario_state.json'))

all_dfs = []
for inpath in tqdm(inpaths, desc='loading data'):
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
        "target"      : x['instance']['references'][0]['output']['text'],
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

def _compute_scores(target, response):
    return {
        "bleu_4"     : bleu_4(target, response),
        "bleu_flat" : bleu_flat(target, response),
        "chrf"      : chrf(target, response),
        "meteor"    : meteor(target, response),
    }

jobs    = [delayed(_compute_scores)(q, r) for q, r in tqdm(df[['target', 'response']].values, desc='computing scores')]
_scores = Parallel(n_jobs=-1, verbose=10)(jobs)

df[['bleu_4', 'bleu_flat', 'chrf', 'meteor']] = pd.DataFrame(_scores)

# _ = pairplot(df[['bleu_4', 'bleu_flat', 'chrf', 'meteor']])
# plt.show()

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