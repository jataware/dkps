Setting is:
    - You've run `n_models` models on a benchmark with `n_instances` instances.
    - You get a new model `new_model`, and you want to estimate it's benchmark score `b(new_model)`.
    - But you're cheap, so you only want to run `new_model` on `budget << n_instances` questions.

What are some naive ways to estimate `new_model`'s score?
 - `p_null`
  - Guess `new_model`'s score is the mean of all the other models.
  - E.g.: `b(new_model) = mean([b(m) for m in models])`
  - *Note*: This only requires access to the overall scores.
  - English: "Predict benchmark score is close to the mean of all the other models."

 - `p_sample`
  - Sample `budget` instances from the benchmark.
  - Run + score `new_model` on the sampled instances.
  - Guess `new_model`'s score is the mean of the scores on the `budget` sampled instances.
  - E.g.: `b(new_model) = mean([score(new_model(i), i) for i in sample(benchmark, budget)])`
  - *Note*: This only requires access to the scoring function.
  - English: "Predict benchmark score is close to the score on a random sample of instances."

Another baseline is:
 - `s_3nn_score`
  - Sample `budget` instances from the benchmark.
  - Run + score `new_model` on the sampled instances.
  - Find the 3 models that have the most similar scores to `new_model` on the `budget` sampled instances (in Euclidean space).
  - Guess `new_model`'s score is the mean of the overall scores those three models.
  - *Note*: This requires access to the scoring function and individual instances scores.
  - English: "Predict benchmark score is close to scores of models that produce similar _scoring_ outputs."

DKPS method:
 - `p_{lr,3nn}_dkps2`
  - Sample `budget` instances from the benchmark.
  - Run `new_model` on the sampled instances.
  - Compute DKPS embeddings for `[original models, new_model]`
  - Train {linear regression, 3-NN regression} `score ~ DKPS embeddings` on the original models.
  - Predict `new_model`'s score using the regression model.
  - *Note*: This requires access to outputs of the original models.
  - English: "Predict benchmark score is close to scores of models that produce similar outputs."

--

Datasets:
 - ~ 90 models
 - 7 splits of [MATH](https://arxiv.org/pdf/2103.03874) dataset
    - Examples: 
        - https://crfm.stanford.edu/helm/lite/latest/#/runs/math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model=openai_gpt-4o-2024-05-13
        - https://crfm.stanford.edu/helm/lite/latest/#/runs/math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,model=openai_gpt-4o-2024-05-13
        - ... etc ...

 - Given a `new_model`, exclude all models trained by the same company from the set of original models.  This is to prevent leakage, since some models are finetuned versions of others.  However, it is probably _too_ aggressive depending on real-world setting.
 
--

Parameters / details:
 - Sweep budget across [2, 4, 8, 16, 32, ...]
 - 32 replicates at each budget
 - Embedding model: `jina-embedding-v3`
 - DKPS dimension: 2
 - Error: mean absolute error `abs(act - pred)`
    - HELM MATH evaluation is 0/1 correct/incorrect, so benchmark scores mean percentage correct
 - Plots show error averaged over all models and all replicates `np.mean(x)`.  Error bars are `1.96 * np.std(x) / np.sqrt(len(x))`
 
Code:
 - https://github.com/jataware/dkps/blob/bkj/examples/helm/math/run.sh




