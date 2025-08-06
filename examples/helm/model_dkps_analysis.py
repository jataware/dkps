import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rich import print as rprint

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='math:subject=algebra')
    parser.add_argument('--score_col', type=str, default='score')
    args = parser.parse_args()
    
    args.tsv_path = Path('data') / f'{args.dataset}-{args.score_col}-res.tsv'
    args.plot_dir = Path('plots')
    
    args.plot_dir.mkdir(parents=True, exist_ok=True)
        
    return args

args = parse_args()

df_res = pd.read_csv(args.tsv_path, sep='\t')

# --
# Plot

# COLORS = ['black', 'blue', 'red', 'green', 'orange']
cnames = [c for c in df_res.columns if 'e_' in c]

df_avg = df_res.groupby(['mode', 'n_samples']).agg({
    'y_act' : lambda x: np.mean(x),
    **{c: lambda x: np.mean(x) for c in cnames},
}).reset_index()

for i,c in enumerate(cnames):
    _ = plt.plot(df_avg.n_samples, df_avg[c], label=c)#, c=COLORS[i])


# # <<
# # Add error bars to show 95% CI of mean
# # [TODO] double check this
# df_ci = df_res.groupby(['mode', 'n_samples']).agg({
#     **{c: lambda x: 1.96 * np.std(x) / np.sqrt(len(x)) for c in cnames},  # 95% CI = 1.96 * SE
# }).reset_index()

# # Merge with averages to get the CI values
# df_plot = pd.merge(df_avg, df_ci, on=['mode', 'n_samples'], suffixes=('', '_ci'))

# # Plot with error bars
# plt.figure(figsize=(10, 6))
# for i, c in enumerate(cnames):
#     plt.errorbar(
#         df_plot.n_samples, 
#         df_plot[c], 
#         yerr=df_plot[f"{c}_ci"],
#         label=c, 
#         # c=COLORS[i],
#         capsize=4,
#         marker='o',
#         markersize=5,
#         linewidth=2,
#         elinewidth=1,
#     )
# # >> 

_ = plt.legend()
_ = plt.grid('both', alpha=0.25, c='gray')
_ = plt.xscale('log')
_ = plt.ylabel(f'error (mean over {n_replicates} runs x {len(model_names)} models)')
_ = plt.xlabel('n_samples')
_ = plt.title(f'{args.dataset} - {args.score_col}')
_ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err-big.png')
_ = plt.close()

# --
# # plot gain over null, per model
# fine, but I don't really care

# df_per_model = df_res.groupby(['target_model', 'mode', 'n_samples']).agg({
#     'y_act'       : 'mean', # noop - they're all the same
#     'e_null'      : 'mean',
#     'e_sample'    : 'mean',
#     'e_dkps2'     : 'mean',
#     'e_knn_dkps2' : 'mean',
#     'e_sknn'      : 'mean',
# }).reset_index()

# df_per_model['dkps2_gain']  = df_per_model.e_dkps2 - df_per_model.e_null
# df_per_model['sample_gain'] = df_per_model.e_sample - df_per_model.e_null
# df_per_model['knn_gain']    = df_per_model.e_knn_dkps2 - df_per_model.e_null
# df_per_model['sknn_gain']   = df_per_model.e_sknn - df_per_model.e_null

# for model in model_names:
#     sub = df_per_model[df_per_model.target_model == model]
#     _ = plt.plot(sub.n_samples, sub.dkps2_gain, c='red', alpha=0.1)
#     _ = plt.plot(sub.n_samples, sub.sample_gain, c='blue', alpha=0.1)
#     _ = plt.plot(sub.n_samples, sub.knn_gain, c='green', alpha=0.1)
#     _ = plt.plot(sub.n_samples, sub.sknn_gain, c='orange', alpha=0.1)

# _ = plt.plot(df_per_model.groupby('n_samples').dkps2_gain.median(), label='dkps2', c='red', linewidth=5)
# _ = plt.plot(df_per_model.groupby('n_samples').sample_gain.median(), label='sample', c='blue', linewidth=5)
# _ = plt.plot(df_per_model.groupby('n_samples').knn_gain.median(), label='knn', c='green', linewidth=5)
# _ = plt.plot(df_per_model.groupby('n_samples').sknn_gain.median(), label='sknn', c='orange', linewidth=5)

# _ = plt.legend()
# _ = plt.ylim(-0.2, 0.2)
# _ = plt.axhline(0, c='black')
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.xscale('log')
# _ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-err-by-model.png')
# _ = plt.close()


# # if you're trying to determine whether the new model is in the top 10% of models, how well do you do vs sampling?
# fine, can revisit later

# t = np.percentile(list(y_acts.values()), 90)

# tmp = []
# for n_samples in df_res.n_samples.unique():
#     sub = df_res[df_res.n_samples == n_samples]
#     tmp.append({
#         'n_samples'     : n_samples,
#         'auc_null'      : metrics.roc_auc_score(sub.y_act > t, sub.p_null),
#         'auc_sample'    : metrics.roc_auc_score(sub.y_act > t, sub.p_sample),
#         'auc_dkps2'     : metrics.roc_auc_score(sub.y_act > t, sub.p_lr_dkps2),
#         'auc_knn_dkps2' : metrics.roc_auc_score(sub.y_act > t, sub.p_3nn_dkps2),
#         'auc_sknn'      : metrics.roc_auc_score(sub.y_act > t, sub.p_3nn_score),
#     })

# df_f1 = pd.DataFrame(tmp)

# _ = plt.plot(df_f1.n_samples, df_f1.auc_null, label='null', c='black')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_sample, label='sample', c='blue')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_dkps2, label='dkps2', c='red')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_knn_dkps2, label='knn', c='green')
# _ = plt.plot(df_f1.n_samples, df_f1.auc_sknn, label='sknn', c='orange')

# _ = plt.legend()
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.xscale('log')
# _ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-auc.png')
# _ = plt.close()


# if you're trying to determine which of two models is better
# only makes sense to do within a family
# [TODO] this makes sense but need to double check

# from tqdm import tqdm

# tmp = []
# for seed in df_res.seed.unique():
#     for n_samples in df_res.n_samples.unique():
#         sub = df_res[(df_res.seed == seed) & (df_res.n_samples == n_samples)]
        
#         o_act       = sub.y_act.values[None,] > sub.y_act.values[:,None]
#         o_sample    = sub.p_sample.values[None,] > sub.p_sample.values[:,None]
#         o_dkps2     = sub.p_lr_dkps2.values[None,] > sub.p_lr_dkps2.values[:,None]
#         o_knn_dkps2 = sub.p_3nn_dkps2.values[None,] > sub.p_3nn_dkps2.values[:,None]
#         o_sknn      = sub.p_3nn_score.values[None,] > sub.p_3nn_score.values[:,None]
        
#         family = np.array([model2family(m) for m in sub.target_model.values])
#         mask   = family[None,] == family[:,None]
        
#         c_sample    = (o_act == o_sample)[mask].sum() / mask.sum()
#         c_dkps2     = (o_act == o_dkps2)[mask].sum() / mask.sum()
#         c_knn_dkps2 = (o_act == o_knn_dkps2)[mask].sum() / mask.sum()
#         c_sknn      = (o_act == o_sknn)[mask].sum() / mask.sum()
        
#         tmp.append({
#             'seed'        : seed,
#             'n_samples'   : n_samples,
#             'c_sample'    : c_sample,
#             'c_dkps2'     : c_dkps2,
#             'c_knn_dkps2' : c_knn_dkps2,
#             'c_sknn'      : c_sknn,
#         })


# tmp = pd.DataFrame(tmp)

# tmp = tmp.groupby('n_samples').agg({
#     'c_sample'    : lambda x: np.mean(x),
#     'c_dkps2'     : lambda x: np.mean(x),
#     'c_knn_dkps2' : lambda x: np.mean(x),
#     'c_sknn'      : lambda x: np.mean(x),
# }).reset_index()

# _ = plt.plot(tmp.n_samples, tmp.c_sample, label='sample', c='blue')
# _ = plt.plot(tmp.n_samples, tmp.c_dkps2, label='dkps2', c='red')
# _ = plt.plot(tmp.n_samples, tmp.c_knn_dkps2, label='knn', c='green')
# _ = plt.plot(tmp.n_samples, tmp.c_sknn, label='sknn', c='orange')
# _ = plt.legend()
# _ = plt.grid('both', alpha=0.25, c='gray')
# _ = plt.xscale('log')
# _ = plt.savefig(f'plots/{args.dataset}-{args.score_col}-win.png')
# _ = plt.close()