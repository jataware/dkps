"""
    runners/qselect.py - DKPS prediction with query selection (holdout family validation)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

from utils import make_embedding_dict
from dkps.dkps import DataKernelPerspectiveSpace as DKPS


def model2family(model):
    return model.split('_')[0]


def knn_predict(X_train, y_train, X_valid=None, n_neighbors=[2, 3, 4]):
    """
        custom knn function
    """

    # predict
    y_preds, r2_scores = {}, {}

    if X_valid is not None:
        dists = cdist(X_valid, X_train)
        neibs = np.argsort(dists, axis=1)
        for k in n_neighbors:
            neibs_k    = neibs[:, :k]
            y_preds[k] = np.mean(y_train[neibs_k], axis=1)

    # compute r2 on training data
    dists = cdist(X_train, X_train)
    np.fill_diagonal(dists, np.inf)
    neibs = np.argsort(dists, axis=1)

    for k in n_neighbors:
        neibs_k      = neibs[:, :k]
        y_pred_train = np.mean(y_train[neibs_k], axis=1)
        r2_scores[k] = r2_score(y_train, y_pred_train)

    return y_preds, r2_scores


def setup(df, model_names, args):
    """Setup function called before running jobs - creates train/valid family split"""
    FAMILIES = list(set([model2family(target_model) for target_model in model_names]))
    TRAIN_FAMILIES, VALID_FAMILIES = train_test_split(FAMILIES, test_size=0.33, random_state=args.seed)
    return {
        'TRAIN_FAMILIES': TRAIN_FAMILIES,
        'VALID_FAMILIES': VALID_FAMILIES,
    }


def run_one(df_sample, n_samples, mode, seed, y_acts, pred_null, TRAIN_FAMILIES=None, VALID_FAMILIES=None, **kwargs):
    out = []
    model_names = df_sample.model.unique()

    embedding_dict = make_embedding_dict(df_sample)

    for target_model in model_names:
        # Only keep target models in VALID_FAMILIES
        if model2family(target_model) not in VALID_FAMILIES:
            continue

        # split data
        assert mode in ['model', 'family']
        if mode == 'model':
            train_models = np.array([m for m in model_names if m != target_model])
        elif mode == 'family':
            target_family = model2family(target_model)
            train_models  = np.array([m for m in model_names if model2family(m) != target_family])

        # only keep models in TRAIN_FAMILIES
        train_models = np.array([m for m in model_names if model2family(m) in TRAIN_FAMILIES])

        y_test  = y_acts[target_model]

        # average score over the `n_samples` evaluated
        p_sample = df_sample[df_sample.model == target_model].score.mean()

        # lr on DKPS embeddings of varying dimension
        res = {}
        for n_components_cmds in [8]:
            for n_models in [20, 50, len(train_models)]:
                if n_models > len(train_models):
                    continue

                if n_models != len(train_models):
                    _suffix  = f'_dkps__n_components_cmds={n_components_cmds}__n_models={n_models}'
                else:
                    _suffix  = f'_dkps__n_components_cmds={n_components_cmds}__n_models=ALL'

                _train_models = np.random.choice(train_models, size=n_models, replace=False)

                # --
                # dkps w/o target model - for GOF metrics only

                _embedding_dict0 = {k:embedding_dict[k] for k in set(_train_models)}
                P0 = DKPS(n_components_cmds=n_components_cmds)
                P0 = P0.fit_transform(_embedding_dict0, return_dict=True)

                _X_train0 = np.vstack([P0[m] for m in _train_models])
                _y_train0 = np.array([y_acts[m] for m in _train_models])

                # linear regression on DKPS embeddings
                lr0 = LinearRegression().fit(_X_train0, _y_train0)

                # goodness of fit metrics
                lr_pred_train0 = lr0.predict(_X_train0)
                res['r2_lr' + _suffix] = r2_score(_y_train0, lr_pred_train0)

                # # knn regression on DKPS embeddings
                # _, r2_knns = knn_predict(_X_train0, _y_train0)
                # for k, r2_knn in r2_knns.items():
                #     res[f'r2_knn{k}' + _suffix] = float(r2_knn)

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
                # p_knns, _ = knn_predict(_X_train, _y_train, _X_test)
                # for k, p_knn in p_knns.items():
                #     res[f'p_knn{k}' + _suffix] = float(p_knn[0])

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
