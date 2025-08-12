import numpy as np
from dkps.dkps import DataKernelPerspectiveSpace

def dkps_df(df, oos=None, **kwargs):
    if oos is None:
        oos = []
    
    model_names  = df.model.unique()
    instance_ids = df.instance_id.unique()
    
    embedding_dim = df.embedding.values[0].shape[0]
    
    embedding_dict = {}
    for model_name in model_names:
        sub = df[df.model == model_name]
        if set(sub.instance_id.values) == set(instance_ids):
            embedding_dict[model_name] = np.row_stack(sub.embedding.values)
        
        else:
            assert model_name in oos, 'if model is not in oos, all instance_ids must be present'
            _tmp = np.zeros((len(instance_ids), embedding_dim)) + np.nan
            for i, iid in enumerate(sub.instance_id.values):
                _tmp[instance_ids == iid] = sub.embedding.values[i]
            embedding_dict[model_name] = _tmp
    
    # <<
    # Adding extra dimension because we only have one replicate
    embedding_dict = {k:v[:,None] for k,v in embedding_dict.items()}
    # >>
    
    is_embedding_dict = {k:v for k,v in embedding_dict.items() if k not in oos}
    
    dkps     = DataKernelPerspectiveSpace(**kwargs)
    dkps_emb = dkps.fit_transform(is_embedding_dict, return_dict=True)
    
    for model_name in oos:
        dkps_emb[model_name] = dkps.transform(embedding_dict[model_name])
    
    return dkps_emb

