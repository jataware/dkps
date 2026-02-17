from pathlib import Path
from dkps.helm import onehot_embedding, compute_embeddings, make_embedding_dict, dkps_df, uses_onehot, DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL

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
