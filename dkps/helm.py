import numpy as np
from .dkps import DataKernelPerspectiveSpace


def onehot_embedding(df, dataset):
    if dataset == 'med_qa':
        lookup = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3}

        embeddings = np.zeros((len(df), 4))
        for i, xx in enumerate(df.response.values):
            xx = xx.strip().upper()
            if xx in lookup:
                embeddings[i, lookup[xx]] = 1

        df['embedding'] = embeddings.tolist()

    elif 'legalbench' in dataset:
        # Map response strings to integer indices; unrecognized values get index 0
        unique_responses = sorted(set(
            r.strip().lower() for r in df.response.values if isinstance(r, str)
        ))
        lookup = {r: i + 1 for i, r in enumerate(unique_responses)}
        n_levels = len(lookup) + 1  # +1 for the 0 (unrecognized) bucket

        embeddings = np.zeros((len(df), n_levels))
        for i, xx in enumerate(df.response.values):
            idx = lookup.get(xx.strip().lower(), 0) if isinstance(xx, str) else xx
            embeddings[i, idx] = 1

        df['embedding'] = embeddings.tolist()
    else:
        raise ValueError(f'{dataset} is not supported for onehot embeddings')

    return df


def make_embedding_dict(df):
    model_names  = df.model.unique()
    instance_ids = df.instance_id.unique()

    embedding_dict = {}
    for model_name in model_names:
        sub = df[df.model == model_name]
        assert (sub.instance_id.values == instance_ids).all(), f'instance_ids are not the same for model {model_name}'
        embedding_dict[model_name] = np.vstack(sub.embedding.values)

    embedding_dict = {k:v[:,None] for k,v in embedding_dict.items()}

    return embedding_dict


ONEHOT_DATASETS = ('med_qa', 'legalbench')

DEFAULT_EMBED_PROVIDER = 'google'
DEFAULT_EMBED_MODEL    = 'gemini-embedding-001'


def uses_onehot(dataset):
    return any(dataset.startswith(d) for d in ONEHOT_DATASETS)


LEGALBENCH_UNRECOGNIZED = '__unrecognized__'


def clean_legalbench_answer(text):
    """Normalize a free-text legalbench answer: lowercase, strip whitespace and a
    trailing period, and drop a leading 'function:'/'answer:'/'label:' prefix."""
    text = str(text).lower().strip().rstrip('.')
    for prefix in ('function: ', 'answer: ', 'label: '):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def prepare_responses(df, dataset, references=None):
    """Dataset-specific normalization of the `response` column prior to embedding.

    legalbench: collapse free-text model answers onto the gold label class space.
    `references` is a sequence of HELM instance `references` entries; the gold
    labels are their distinct correct-answer strings. A response that does not
    clean to a gold label collapses to a single 'unrecognized' class. Without
    this, onehot_embedding builds a class space from every distinct raw response.

    Other datasets: returned unchanged.
    """
    if not dataset.startswith('legalbench'):
        return df

    if references is None:
        raise ValueError("legalbench response prep requires `references`")

    gold = {str(refs[0]['output']['text']).strip().lower() for refs in references}

    def to_class(response):
        cleaned = clean_legalbench_answer(response)
        return cleaned if cleaned in gold else LEGALBENCH_UNRECOGNIZED

    df = df.copy()
    df['response'] = df['response'].map(to_class)
    return df


def compute_embeddings(df, dataset, embed_provider=None, embed_model=None):
    if uses_onehot(dataset):
        return onehot_embedding(df, dataset)
    else:
        from dkps.embed import embed_api
        if embed_provider is None:
            embed_provider = DEFAULT_EMBED_PROVIDER
        if embed_model is None:
            embed_model = DEFAULT_EMBED_MODEL
        df['embedding'] = list(embed_api(
            provider=embed_provider,
            input_strs=[str(xx) for xx in df.response.values],
            model=embed_model
        ))
        return df


def dkps_df(df, **kwargs):
    embedding_dict = make_embedding_dict(df)
    return DataKernelPerspectiveSpace(**kwargs).fit_transform(embedding_dict, return_dict=True)
