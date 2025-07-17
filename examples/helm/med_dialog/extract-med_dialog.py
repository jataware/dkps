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
    parser.add_argument('--outpath', type=str, default='./data/helm/medhelm/med_dialog.tsv')
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

TARGET = "Generate a one sentence summary of this patient-doctor conversation.\n\nPatient-Doctor: Patient: Hello doctor,I am trying to conceive but my husband and I did cocaine a week ago. How long should my husband and I wait to safely continue to try to get pregnant? How long until it is out of our system? How long does cocaine stay in sperm? Thanks in advance. Doctor: Hello, Wellcome to iclinq.com. There are few researches/studies on cocaine use by males and its effect on pregnancy. Few suggest that cocaine by itself has limited effects as most of the time it is taken along with other drugs or as a cocktail of alcohol and cigarette (tobacco). So, most of the people take not just cocaine but a combination of drugs. Cocaine narrows blood vessels (vasoconstriction). It can lead to erectile dysfunction. Few studies suggest that it has receptors on testicles and sperm. So, it can degenerate testicular tissues/or sperm quality, transfer from sperm to female egg and can lead to early miscarriage. Cocaine is a very fast acting drug which affects the nervous system and produces short-lived euphoric attitude for 15 minutes to an hour, but causes long-term damage to the body and brain like anxiety, depression, aggression, impairment of logic and critical thinking, heart problem, hypertension and decrease in bone density. Its half-life is an hour. So, it takes about an hour for half of the cocaine consumed to leave the body. But, with long-term use, the drug starts to accumulate in the blood and body tissues allowing certain tests to detect it in the system for an extended period of time. After a single use of cocaine, agents created by its metabolism can be detected in the urine for two to four days, and in chronic users, cocaine can be detected up to 12 days and highly concentrated cocaine can be detected in the urine up to three weeks. It can be detected in the saliva and blood for an average 12-48 hours. In hairs and sweat for an extended period of time, it can be detected. So, after all the above description, I do not know how your husband had cocaine, as a cocktail along with other drugs, or just cocaine? Secondly, how long he has been taking it? For you, have you been on alcohol, cocaine or smoking? If you had taken in the past, better to quit completely. My advice is, try to avoid drugs like cocaine, alcohol, ketamine, and MDMA completely for a couple of months before trying for pregnancy. Because, if the mother has an addiction, it has psychosocial effects on the fetus in addition to the drugs' side effects itself. Best of luck. For more information consult an obstetrician and gynaecologist online --> https://www.icliniq.com/ask-a-doctor-online/obstetrician-and-gynaecologist\nSummary:"

all_dfs = []
bad_ids = set()
for inpath in inpaths:
    run     = json.loads(inpath.read_text())
    _params = {    
        "dataset" : inpath.parent.name.split(':')[0],
        **dict([x.split('=') for x in inpath.parent.name.split(':')[1].split(',')])
    }
    
    counter = 0
    for x in run['request_states']:
        if x['request']['prompt'] == TARGET:
            counter += 1
    
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