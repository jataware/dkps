"""
    runners/dkps.py - Standard DKPS prediction (leave-one-out by model/family)
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from utils import make_embedding_dict
from dkps.dkps import DataKernelPerspectiveSpace as DKPS


def model2family(model):
    return model.split('_')[0]


def setup(df, model_names, args):
    """Setup function called before running jobs"""
    return {}


def run_one(df_sample, n_samples, mode, seed, y_acts, pred_null, **kwargs):
    out = []
    model_names = df_sample.model.unique()

    embedding_dict = make_embedding_dict(df_sample)

    for target_model in model_names:

        # split data
        assert mode in ['model', 'family']
        if mode == 'model':
            train_models = np.array([m for m in model_names if m != target_model])
        elif mode == 'family':
            target_family = model2family(target_model)
            train_models  = np.array([m for m in model_names if model2family(m) != target_family])

        y_test  = y_acts[target_model]

        # average score over the `n_samples` evaluated
        p_sample = df_sample[df_sample.model == target_model].score.mean()

        # lr on DKPS embeddings of varying dimension
        res = {}
        for n_components_cmds in [8]:
            for n_models in [20, 50, len(train_models)]:
                if n_models != len(train_models):
                    _suffix  = f'_dkps__n_components_cmds={n_components_cmds}__n_models={n_models}'
                else:
                    _suffix  = f'_dkps__n_components_cmds={n_components_cmds}__n_models=ALL'

                _train_models = np.random.choice(train_models, size=n_models, replace=False)

                # --
                # dkps w/ target model

                _embedding_dict = {k:embedding_dict[k] for k in (set(_train_models) | set([target_model]))}
                P = DKPS(n_components_cmds=n_components_cmds)
                P = P.fit_transform(_embedding_dict, return_dict=True)

                _X_train = np.vstack([P[m] for m in _train_models])
                _y_train = np.array([y_acts[m] for m in _train_models])
                _X_test  = np.vstack([P[target_model]])

                # linear regression on DKPS embeddings
                lr = LinearRegression().fit(_X_train, _y_train)
                res['p_lr' + _suffix] = float(lr.predict(_X_test)[0])

                # # knn regression on DKPS embeddings
                # knn = KNeighborsRegressor(n_neighbors=5).fit(_X_train, _y_train)
                # res['p_knn5' + _suffix] = float(knn.predict(_X_test)[0])

        out.append({
            "seed"         : seed,
            "n_samples"    : n_samples,
            "mode"         : mode,
            "target_model" : target_model,

            "y_act"        : y_test,
            "p_null"       : pred_null[mode][target_model],
            "p_sample"     : p_sample,

            **res,
        })

    return out
