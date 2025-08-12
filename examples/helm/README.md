# AIQ Algorithm Card

**Team Name:** JHU

**Algorithm Name:** Data Kernel Perspective Space (DKPS) - Performance Estimation

**Last Updated:** 2025-08-12

**Algorithm Description**
This algorithm predicts the performance of a new model `m_new` on a dataset in a query-efficient way.  
We use the (precomputed) scored outputs from a (large) set of models plus _unscored_ outputs from a new model to learn a regressor in DKPS space that predicts the performance of the new model on the entire dataset.

**Pseudocode:**
```
def run_dkps_performance_estimation(
  models_old : list[LLM],
  dataset    : list[str],
  metric     : Metric,              # scoring metric for dataset
  model_new  : LLM,
  budget     : int,                 # query budget for new model
  embedder   : DenseEmbeddingModel, # neural network that embeds strings -> list[float]
  regressor  : Regressor,           # sklearn-style regressor that takes a list[float] and predicts a float
):
  # compute aggregate scores of models on dataset
  # Note: These can be precomputed / amortized.  In this repo, we are actually pulling these from HELM.
  model_old_outputs        : list[list[str]]   = [[model(d) for d in dataset] for model in models_old]
  model_old_output_scores  : list[list[float]] = [[metric(o) for o in outputs] for outputs in model_old_outputs]
  model_old_agg_scores     : list[float]       = [np.mean(scores) for scores in model_old_output_scores]
  
  # run `model_new` on a `b`-sized subset of `dataset`
  sel                      : list[int] = np.random.choice(len(dataset), size=budget, replace=False)
  model_new_subset_outputs : list[str] = [model_new(dataset[i]) for i in sel]
  
  # embed outputs of `models` and `model_new`
  model_old_output_embs : list[list[list[float]]] = [[embedder(o) for o in outputs[sel]] for outputs in model_old_outputs]
  model_new_output_embs : list[list[float]]       = [embedder(o) for o in model_new_subset_outputs]
  
  # run dkps on output of `models` and `model_new`on `dataset[sel]`
  # this yields a low-dimension embedding of each _model_
  model_embs: list[list[float]] = dkps(model_old_output_embs + model_new_output_embs)
  
  # train a regressor to predict model_old_agg_scores using model_embs
  regressor.train(model_embs[:-1], model_old_agg_scores)
  
  # predict score of `model_new` on `dataset`
  model_new_agg_score = regressor.predict(model_embs[-1])
  
  return model_new_agg_score
```

**Practical Impact**

These methods could be used to estimate model performance in settings where:
  - inference (running `new_model` on elements of `dataset`) is expensive
  - scoring (computing / determining `metric(o)`) is expensive or impossible (e.g. in the case where `metric` is actually an expert human judge)

