from pathlib import Path
from dkps.helm import onehot_embedding, make_embedding_dict, dkps_df

def make_experiment_path(embed_provider, embed_model, dataset, score_col, n_replicates=None):
    if embed_model == 'onehot':
        embed_provider = 'local'

    _embed_str = 'embed-' + embed_provider + ('-' + embed_model if embed_model else '')
    out = (
        Path(_embed_str)          /
        dataset.replace(':', '-') /
        score_col
    )

    if n_replicates:
        out = out / str(n_replicates)

    return out
