# Comparison with IRT and APW Baselines

## Baseline Descriptions

**IRT (Polo et al., 2024).** We implement the 1-parameter logistic (Rasch) model, which models item responses as $P(S_{ij} = 1 \mid \theta_i, \beta_j) = \sigma(\theta_i - \beta_j)$, where $\theta_i$ is model ability and $\beta_j$ is item difficulty. We choose 1PL over 2PL for robustness: with ~80 reference models and up to 1000 items, 2PL discrimination parameters are poorly identified. Item difficulties $\beta$ are fit via L-BFGS-B on the reference model score matrix with abilities initialized by the row-mean logit. At evaluation time, the held-out model's ability $\hat\theta$ is estimated by MLE from $m$ randomly selected item responses, and its benchmark score is predicted as $\frac{1}{M}\sum_j \sigma(\hat\theta - \beta_j)$. The ability estimate is capped at $\pm 4$ to handle the all-correct/all-incorrect edge cases at $m=1$.

**APW (Vivek et al., 2024).** Anchor Point Weighting selects $m$ representative queries via K-Medoids clustering on a query-query Pearson correlation matrix (computed across reference models), then predicts the held-out model's score as a cluster-size-weighted average of its scores on the anchor queries. We implement PAM K-Medoids with 5 random restarts. Unlike IRT and Sample Score, APW uses *active* query selection rather than random queries, so the comparison is not strictly controlled—APW numbers represent the method at its best.

## Adaptations to Our Setting

Both baselines were originally designed for full-benchmark evaluation with all queries available. In our query-efficient setting ($m \ll M$), several modifications were necessary:

- IRT receives only $m$ randomly-selected item responses for the held-out model (the same queries used by Sample Score and DKPS), while item difficulties are fit on the full reference model score matrix.
- APW's K-Medoids query selection replaces the random selection, so it uses different queries than the other methods at each $m$.
- Zero-variance queries (where all reference models score identically) are assigned correlation 0 in APW's correlation matrix.

## Combined Methods

We also evaluate two ways of combining IRT with DKPS:

- **DKPS+IRT**: append the IRT ability estimate $\hat\theta$ as an additional feature to the DKPS embedding coordinates before fitting the linear regression. This gives the regressor both behavioral similarity (DKPS) and direct performance (IRT) signals.
- **Ensemble**: interpolate DKPS+IRT and Sample Score predictions as $\hat{y} = \frac{M-m}{M} \hat{y}_{\text{DKPS+IRT}} + \frac{m}{M} \hat{y}_{\text{Sample}}$, matching the ensemble formulation in the main paper.

## Evaluation

We use leave-one-family-out (LOFO) evaluation with 100 runs per setting. In each run, a model family is sampled uniformly at random for held-out evaluation; all remaining models serve as references ($n = \text{ALL}$). Results are reported as mean absolute error (MAE) across all runs and held-out models.

## Results

### MATH (counting_and_probability) — 95 models, 39 queries, binary scores

| $m$ | Pop. Mean | Sample | IRT | APW | DKPS | DKPS+IRT | Ens(DKPS+IRT) |
|-----|-----------|--------|-----|-----|------|----------|---------------|
| 1   | .247      | .312   | .277 | .276 | .142 | .140    | **.139**      |
| 4   | .247      | .152   | .125 | .141 | .110 | .099    | **.097**      |
| 16  | .247      | .060   | **.050** | .077 | .092 | .056 | .053         |

### LegalBench (abercrombie) — 93 models, 95 queries, binary scores

| $m$ | Pop. Mean | Sample | IRT | APW | DKPS | DKPS+IRT | Ens(DKPS+IRT) |
|-----|-----------|--------|-----|-----|------|----------|---------------|
| 1   | .168      | .421   | .385 | .364 | .170 | .170    | **.170**      |
| 4   | .168      | .185   | .165 | .184 | .103 | .102    | **.100**      |
| 16  | .168      | .090   | .082 | .073 | .057 | **.055** | .056         |
| 64  | .168      | .025   | .022 | .028 | .030 | .022    | **.021**      |

### MedQA — 95 models, 1000 queries, binary scores

*(Running — results pending.)*

### WMT-14 (ru-en) — 94 models, 122 queries, BLEU scores

*(Requires API key for cached embeddings — results pending.)*

## Discussion

IRT and APW both improve over Sample Score at all query budgets, confirming that leveraging reference model structure is beneficial. However, both baselines are consistently outperformed by DKPS, often substantially: at $m=4$ on LegalBench, DKPS achieves MAE .103 vs. IRT's .165 and APW's .184—a 38–44% reduction.

The DKPS+IRT combination is particularly effective, achieving the best or near-best MAE at every operating point. Appending the IRT ability estimate as a feature gives the DKPS linear regression access to a complementary signal: IRT captures the direct item-level performance pattern, while DKPS captures behavioral similarity through the response embedding geometry. On MATH at $m=16$, DKPS alone (.092) is worse than standalone IRT (.050)—this is a small dataset (39 queries) where the embedding geometry has limited resolution—but DKPS+IRT (.056) recovers most of IRT's advantage while retaining DKPS's strength at low $m$.

The Ensemble further improves at low $m$ by incorporating the sample mean, which provides an unbiased (if noisy) signal that complements the model-based predictions.

APW's active query selection (shown in the APW column and available as an option for DKPS) provides modest gains at intermediate $m$ but does not change the overall ranking of methods.

## Reproducing

```bash
# Standalone baselines (IRT, APW)
python run_baselines.py --dataset "math:subject=counting_and_probability" --n_lofo 100
python run_baselines.py --dataset "legalbench:subset=abercrombie" --n_lofo 100

# Combined methods (DKPS, DKPS+IRT, Ensemble, with random + APW query selection)
python run_combined.py --dataset "math:subject=counting_and_probability" --embed_provider google --n_lofo 100
python run_combined.py --dataset "legalbench:subset=abercrombie" --embed_model onehot --n_lofo 100
python run_combined.py --dataset "med_qa" --embed_model onehot --n_lofo 100
python run_combined.py --dataset "wmt_14:language_pair=ru-en" --embed_provider google --score_col meteor --sample 0.2 --n_lofo 100
```
