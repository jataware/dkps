# Paper Plan: Unpaired DKPS
## *Embedding Black-Box Generative Models from Unpaired Responses*

Target venue: ICLR 2026

---

## 1. Paper Narrative

Standard DKPS requires every model to answer every query — a controlled experimental design that is unavailable when studying models "in the wild." This paper extends DKPS to the setting where queries are not shared across models, or only partially shared. We make four contributions:

1. **Method.** A simple estimator of model distance from unpaired observations based on linear MMD (mean embedding distance). The estimator is the direct analogue of DKPS with pairing removed, and reduces to standard DKPS when queries are fully shared.

2. **Identifiability.** We show that without correction, the unpaired estimator conflates two distinct sources of apparent model difference: genuine behavioral difference and query distribution difference. We introduce a coverage imputation principle that separates these, and show it is the minimal condition required for the estimator to identify model similarity rather than query-distribution similarity. Coverage weights are estimated via KDE in a low-dimensional PCA space derived from pooled response embeddings, with dimensionality selected by the scree plot of eigenvalues.

3. **Unified framework for missing queries.** When queries are unobserved, we impute them using a language model inverter — a model prompted to recover the query that would elicit each observed response. This unifies the query-available and query-unavailable regimes under a single method; the only difference is the noise level of the coverage weights. We demonstrate robustness to the choice of inverter across a diverse suite of imputer models.

4. **Theory.** Under mild assumptions, we prove that the combined paired+unpaired estimator achieves strictly lower asymptotic variance than either alone, and characterize the exchange rate between paired and unpaired observations as a function of query distribution overlap.

**Applications.** We demonstrate the method on two tasks: predicting model benchmark scores from a small set of held-out queries, and auditing models for training data exposure — detecting whether a model has been trained on or given retrieval access to a specific document corpus.

---

## 2. Method

### 2.1 Background: Paired DKPS

*[Standard exposition — refer to Helm et al. 2025.]*

The paired DKPS distance between models i and k is:

$$D_{\text{paired}}(i,k) = \|\bar{X}_i - \bar{X}_k\|_F$$

where $\bar{X}_i \in \mathbb{R}^{m \times p}$ is the matrix of per-query mean embedded responses for model i, with $(\bar{X}_i)_j = \mathbb{E}[g(f_i(q_j))]$ estimated from replicates. This requires all n models to share the same m queries.

### 2.2 General Kernel Framework

Let $\mathcal{K}: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$ be a positive definite kernel on the response embedding space and $\kappa: \mathcal{Q} \times \mathcal{Q} \to \mathbb{R}_{\geq 0}$ a weighting function on query pairs. The kernel-weighted distance between models $i$ and $k$ is:

$$\hat{D}^2_\kappa(i,k) = \sum_{j \in \mathcal{J}_i,\, j' \in \mathcal{J}_k} \kappa(q_j, q_{j'}) \left[ \mathcal{K}(x_{ij}, x_{kj'}) - \tfrac{1}{2}\mathcal{K}(x_{ij}, x_{ij'}) - \tfrac{1}{2}\mathcal{K}(x_{kj}, x_{kj'}) \right]$$

where $\mathcal{J}_i$ denotes the index set of queries posed to model $i$. The choice of $\kappa$ and $\mathcal{K}$ determines how query pairs contribute to the distance estimate. With appropriate choices of $\kappa$ and $\mathcal{K}$ this framework recovers paired DKPS — in particular, $\kappa(q, q') = \mathbf{1}[q = q']$ with $\mathcal{K}(x,y) = x^\top y$ and shared queries $\mathcal{J}_i = \mathcal{J}_k$ reduces to $\|\bar{X}_i - \bar{X}_k\|_F$. The proposed estimator, defined in Section 2.3, uses a $\kappa$ that composes coverage adjustment with a paired/unpaired mixture.

Enriching to a nonlinear response kernel $\mathcal{K}$ is a natural extension if first-moment comparisons prove insufficient; all theoretical results carry over to any bounded positive definite $\mathcal{K}$.

### 2.3 The Proposed Estimator

The proposed $\kappa$ serves two distinct purposes that must not be conflated:

**Purpose 1: Identifiability (coverage adjustment).** When two models are queried in different regions of $\mathcal{Q}$, $\hat{D}^2_\kappa$ with $\kappa \equiv 1$ conflates two sources of apparent difference:
- *Genuine model difference:* the models would respond differently to the same query.
- *Query distribution difference:* the models were simply asked different questions.

These are not separately identified without additional structure. Coverage adjustment restores identifiability by downweighting query pairs in regions where either model lacks coverage. Let $\hat{p}_i(q)$ denote the estimated query density of model $i$ near $q$. Define:

$$w(i,k,q,q') = \frac{2\,\hat{p}_i(q)\,\hat{p}_k(q')}{\hat{p}_i(q) + \hat{p}_k(q')}$$

(the harmonic mean of model $i$'s density at $q$ and model $k$'s density at $q'$). This is zero when either model lacks coverage in its respective query region, and one when both are equally represented. When $w(q,q') = 0$ — one side of the query pair is uncovered — that pair contributes nothing to the distance rather than spurious dissimilarity. In such regions the estimator implicitly imputes the neutral baseline $\bar{\delta}(i,k)$ (the average distance over well-covered pairs). Coverage adjustment is *necessary for identifiability*: without it, query distribution differences are conflated with model differences.

**Purpose 2: Variance reduction (paired queries).** Paired queries — where both models answer the same query — cancel the query's contribution to the response, leaving only the model-specific signal. Incorporating paired queries when available reduces estimator variance. This is orthogonal to coverage adjustment: it improves efficiency regardless of whether query distributions match.

**The composed estimator.** When $\alpha \in [0,1]$ fraction of queries are shared (paired) and the remainder are unpaired, the two purposes are composed into a single query kernel:

$$\kappa_{\alpha,w}(q, q') = \alpha \cdot \mathbf{1}[q = q'] + (1-\alpha) \cdot w(i,k,q,q')$$

The $\mathbf{1}[q=q']$ term extracts paired comparisons for variance reduction; the $w$ term applies coverage adjustment to the unpaired comparisons for identifiability. At $\alpha = 1$ this recovers paired DKPS; at $\alpha = 0$ it recovers the coverage-adjusted unpaired estimator. Setting $w \equiv 1$ gives the combined estimator without coverage adjustment. The optimal $\alpha^*$ is derived in Section 3.

### 2.4 Estimating Coverage Weights

**Low-dimensional PCA space.** Embedded responses $x = g(f(q)) \in \mathbb{R}^p$ are high-dimensional (p = 768 for all-mpnet-base-v2), making density estimation unreliable. We project all pooled embedded responses into a $d_{\text{PCA}}$-dimensional subspace via PCA, with $d_{\text{PCA}}$ selected by the scree plot of eigenvalues (profile likelihood of Zhu & Ghodsi, 2006). All density estimation is performed in this space.

**Coverage weights when queries are observed.** Project query embeddings into the PCA response space. Estimate $\hat{p}_i(q)$ via KDE and compute $w(i,k,q,q')$ as above.

**Coverage weights when queries are unobserved.** Impute queries using a modern language model inverter prompted with each observed response $f(q)$ to produce $\hat{q}$. The imputed queries are projected into the PCA response space and used to estimate $w$. Errors in imputation increase noise in $w$ but do not bias $\hat{D}$, since the distance computation uses only embedded responses $x^i_j$. The method degrades gracefully: noisier coverage weights, unchanged distance estimator.

**Robustness to inverter choice.** Evaluated across a suite of modern language models in Experiment S3; KL divergence between imputed and true query distributions in PCA space serves as a selection criterion.

---

## 3. Theory

### 3.1 Setup

Let $D^*(i,k)$ be the true model distance under a shared reference query distribution $Q^*$. Let $\mathcal{K}$ be a bounded positive definite kernel on $\mathbb{R}^p$ and $\kappa$ a positive definite kernel on $\mathcal{Q}$. Denote by $\hat{D}^2_{\kappa,m_p}$ the kernel-weighted estimator from $m_p$ paired queries and $\hat{D}^2_{1,m_u}$ the unpaired estimator (constant $\kappa = 1$) from $m_u$ responses per model. Assume:

- **(A1) Query overlap.** Each model's query distribution $Q_i$ has density bounded below by $\varepsilon > 0$ on the support of $Q^*$.
- **(A2) Bounded embeddings.** $\|g(f(q))\| \leq B$ almost surely.
- **(A3) Smoothness.** The conditional mean embedded response $\mathbb{E}[g(f_i(Q)) \mid Q=q]$ is $L$-Lipschitz in $q$ in the PCA subspace.
- **(A4) PCA density fidelity.** The $d_{\text{PCA}}$ principal components retain sufficient variance that KDE in the projected space is consistent for the marginal query density in the relevant directions. For any fixed $d_{\text{PCA}}$ retaining $(1-\delta)$ of total variance, the approximation error is $O(\delta)$. This follows from standard KDE consistency results (Devroye & Györfi); the scree plot criterion ensures $\delta$ is small.

### 3.2 General Kernel Results

**Proposition 1 (Bias of uncorrected estimator).** Under (A1)–(A2), $\hat{D}_{1,m_u} \to_p D^*(i,k) + \text{bias}(\varepsilon)$ as $m_u \to \infty$, where $\text{bias}(\varepsilon) \to 0$ as $\varepsilon \to 1$ and $\text{bias}(\varepsilon) > 0$ whenever query distributions differ. The bias is determined by the mismatch between the query distribution used to elicit responses and $Q^*$, and holds for any bounded kernel $\mathcal{K}$.

**Proposition 2 (Consistency of coverage-adjusted estimator).** Under (A1)–(A4), the coverage-adjusted estimator $\hat{D}_{\text{adj}} \to_p D^*(i,k)$ as $m_u \to \infty$, regardless of $\varepsilon$, for any bounded kernel $\mathcal{K}$. Assumption (A4) is a standard regularity condition on variance concentration in the PCA projection; it is satisfied whenever $d_{\text{PCA}}$ is chosen by the scree plot criterion.

**Theorem 1 (Variance reduction).** Under (A1)–(A4), for any bounded kernel $\mathcal{K}$, the combined estimator $\hat{D}(\alpha^*)$ with optimally chosen $\alpha^*$ satisfies:

$$\text{Var}[\hat{D}(\alpha^*)] \leq \min\bigl(\text{Var}[\hat{D}^{\text{paired}}],\, \text{Var}[\hat{D}^{\text{unpaired}}]\bigr)$$

with strict inequality whenever both $m_p > 0$ and $m_u > 0$.

**Theorem 2 (Exchange rate).** Under (A1)–(A4), one paired observation is asymptotically worth $1/\varepsilon^2$ unpaired observations in terms of variance reduction. When query distributions are identical ($\varepsilon = 1$), the combined estimator with $m_p$ paired and $m_u$ unpaired queries achieves the same variance as $m_p + m_u$ paired queries.

**Corollary (Small paired regime).** The combined estimator weakly dominates the paired-only estimator for all $m_p, m_u \geq 0$, with strict improvement whenever $m_u > 0$ and $\varepsilon > 0$.

### 3.3 Optimal Query Kernel for Linear MMD

For the linear response kernel $\mathcal{K}(x,y) = x^\top y$, we derive the optimal query kernel $\kappa^*$ that minimizes the variance of the estimator subject to unbiasedness.

**Setup.** Under the synthetic generative model (Section 4.1) with $x_{ij} = q_j + v_i + \varepsilon_{ij}$, the estimator $\hat{D}^2_\kappa(i,k)$ is unbiased for $\|v_i - v_k\|^2$ if and only if $\kappa$ satisfies:

$$\int \kappa(q, q') \, dQ_i(q) \, dQ_k(q') = \int \kappa(q, q') \, dQ^*(q) \, dQ^*(q')$$

i.e., the $\kappa$-weighted cross-covariance is the same under the observed and reference query distributions. Minimizing variance of the resulting estimator over the space of such $\kappa$ yields:

**Theorem 3 (Optimal query kernel for linear MMD).** Under the Gaussian synthetic model with $Q_i = Q_k = Q^*$ (no distribution mismatch), the variance-minimizing unbiased query kernel is:

$$\kappa^*(q, q') = \exp\!\left(-\frac{\|q - q'\|^2}{2\sigma^{*2}}\right)$$

where $\sigma^{*2} = (s^{-2} + t^{-2}) / \|v_i - v_k\|^2$ is determined by the signal-to-noise ratio of the generative model. This is an RBF kernel on queries. In the limit $\sigma^* \to 0$, $\kappa^*$ concentrates on identical queries and recovers the paired DKPS estimator. As $\sigma^* \to \infty$, $\kappa^*$ becomes constant and recovers linear MMD.

**Interpretation.** The optimal query kernel weights query pairs by their similarity — similar queries invoke similar responses (by A3) and thus provide a more informative comparison than dissimilar pairs. The optimal bandwidth $\sigma^*$ is determined by the coupling strength $s$: when coupling is strong (large $s$), similar queries are highly informative and $\sigma^*$ is small (kernel concentrates); when coupling is weak, query similarity is less informative and $\sigma^*$ is large. The constant kernel ($\sigma^* \to \infty$, i.e., linear MMD) is optimal only when queries carry no information about model identity relative to observation noise — i.e., when A3 is poorly satisfied. This gives a formal characterization of the regime in which linear MMD is the right choice.

**Corollary (Optimal combination weight for linear $\mathcal{K}$).** With the coverage-adjusted estimator, both the paired and unpaired components are unbiased, and the optimal combination weight reduces to inverse-variance weighting:

$$\alpha^* = \frac{m_p}{m_p + m_u}$$

This is fully estimable from data with no knowledge of $\varepsilon$ or $\sigma^*$. The coverage adjustment therefore has a second benefit beyond identifiability: it makes the optimal combination rule data-adaptive, replacing an inestimable bias-dependent expression with a simple ratio of observable sample sizes.

*Proof sketches.* Proposition 1 follows from bias of kernel mean embeddings under mismatched query distributions. Proposition 2 follows from consistency of the coverage-adjusted estimator under (A1)–(A4). Theorems 1–2 follow from the bias-variance decomposition of the combined estimator. Theorem 3 follows from minimizing the variance of the $U$-statistic estimator of $\hat{D}^2_\kappa$ over the RKHS of symmetric kernels on $\mathcal{Q}$, with the unbiasedness constraint enforced via Lagrange multipliers; under the Gaussian generative model the solution is the RBF kernel with the stated bandwidth. Full proofs in appendix.

---

## 4. Experiments

Experiments are organized into three blocks: synthetic studies establishing core statistical behavior, and two real-data applications.

### Block I: Synthetic Experiments

#### 4.1 Unified Synthetic Setup

**Queries.** For $j = 1, \ldots, m$, queries are drawn i.i.d. from a mixture of two Gaussians:
$$q_j \sim \pi_1 \, \mathcal{N}(\mu_1,\, I_{d_{\text{act}}}) + \pi_2 \, \mathcal{N}(\mu_2,\, I_{d_{\text{act}}})$$
with $\mu_1 = [d/2, 0, \ldots]$ and $\mu_2 = [-d/2, 0, \ldots]$, so that the scalar $d = \|\mu_1 - \mu_2\|$ controls component separation. At $d = 0$ the mixture collapses to a single Gaussian; as $d$ grows the two components become increasingly distinct. Paired and unpaired queries are drawn with potentially different mixture weights $\pi^{\text{paired}}$ and $\pi^{\text{unpaired}}$, enabling controlled mismatch: the default is $\pi^{\text{paired}} = [1, 0]$ (paired queries from component 1 only) and $\pi^{\text{unpaired}} = [0, 1]$ (unpaired queries from component 2 only). Setting both to $[0.5, 0.5]$ gives no mismatch.

**Model offsets.** For $i = 1, \ldots, n$, drawn once and fixed across all queries:
$$v_i \sim \mathcal{N}(0,\, I_{d_{\text{act}}})$$

**Latent mean response.** For model $i$, query $j$:
$$\mu_{ij} \sim \mathcal{N}(q_j + v_i,\, s^{-2}\, I_{d_{\text{act}}})$$

**Observed response.** Observation noise $t^{-2}$ is added, and $d_{\text{obs}} - d_{\text{act}}$ pure noise dimensions are appended:
$$r_{ij} \sim \mathcal{N}(\mu_{ij},\, t^{-2}\, I_{d_{\text{act}}}), \qquad \varepsilon_{ij} \sim \mathcal{N}(0,\, I_{d_{\text{obs}} - d_{\text{act}}})$$
$$x_{ij} = [r_{ij},\; \varepsilon_{ij}] \in \mathbb{R}^{d_{\text{obs}}}$$

**Ground truth distance.**
$$D^*(i,k) = \|v_i - v_k\|$$

**Marginal structure.** Integrating out $\mu_{ij}$:
$$r_{ij} \mid q_j, v_i \sim \mathcal{N}(q_j + v_i,\; (s^{-2} + t^{-2})\, I_{d_{\text{act}}})$$

The population mean embedded response for model $i$, averaged over $P_Q$, is:
$$\mu_i = \mathbb{E}_{q \sim P_Q}[x_{ij}] = \left[\sum_c \pi_c \mu_c + v_i,\; 0\right]$$

Since $\sum_c \pi_c \mu_c$ is the same for all models, the population linear MMD distance is:
$$\|\mu_i - \mu_k\| = \|v_i - v_k\| = D^*(i,k)$$

Recovery is guaranteed in the large-$m$ limit regardless of $s$ and $t$, *provided* paired and unpaired queries are drawn from the same $P_Q$. When paired and unpaired queries are drawn from different mixture components, the per-component means $\mu_c$ no longer cancel and the uncorrected estimator is biased — this is precisely the coverage problem the paper addresses.

**Role of parameters.**
- **$s$** — query-response coupling. Large $s$: $\mu_{ij} \approx q_j + v_i$ tightly, responses are highly query-dependent, coverage of query mixture components matters, A3 is well-satisfied. Small $s$: responses are diffuse around $q_j + v_i$, query structure is washed out, coverage is irrelevant, A3 is poorly satisfied.
- **$t$** — observation noise. Controls within-query response variance independently of coupling.
- **$d_{\text{obs}} - d_{\text{act}}$** — noise dimensions. Tests whether PCA correctly recovers the $d_{\text{act}}$-dimensional signal subspace and discards noise.
- **$d$** — component separation $\|\mu_1 - \mu_2\|$. Controls the severity of query distribution mismatch when paired and unpaired queries are drawn from different components. At $d=0$ there is no mismatch; at large $d$ the two query distributions are nearly disjoint.
- **$\pi^{\text{paired}}, \pi^{\text{unpaired}}$** — mixture weights for paired and unpaired queries. Setting these to different distributions induces mismatch; setting them equal eliminates it. Controls query overlap $\varepsilon$ for Experiment S1.

**Default parameters.** $n = 20$, $d_{\text{act}} = 10$, $d_{\text{obs}} = 50$, $d = 2$, $s = 2$, $t = 2$, $m_{\text{total}} = 1000$, $\pi^{\text{paired}} = [1, 0]$, $\pi^{\text{unpaired}} = [0, 1]$.

**Pairing structure.** A fraction $\alpha$ of $m_{\text{total}}$ queries are shared (paired) across all models; the remaining $(1-\alpha)$ are drawn independently per model from $P_Q$. Query distribution mismatch in Experiment S2 is induced by drawing paired and unpaired queries from different mixture component weightings $\{\pi_c\}$.

**Experimental parameter grid.**

| Experiment | Fixed | Varied |
|---|---|---|
| S0 (kernel expressiveness) | C=2, s, t, α=0 | component separation d |
| S1 (exchange rate) | C=2, d=1, s, t | ε (paired/unpaired mixture weights), m_p |
| S2 (query kernel) | C=2, α=0, κ_w | component separation d |

**Implementation.**

```python
def generate_data(d_act, d_obs, n_models, n_queries, alpha, s, t,
                  d_sep=2.0, pi_paired=None, pi_unpaired=None):
    """
    Parameters
    ----------
    d_act       : int    — dimensionality of active (signal) subspace
    d_obs       : int    — total observed dimensionality (d_act + noise dims)
    n_models    : int    — number of models
    n_queries   : int    — total query budget m_total
    alpha       : float  — fraction of queries that are paired (shared across models)
    s           : float  — query-response coupling (inverse std of latent mean)
    t           : float  — observation noise (inverse std around latent mean)
    d_sep       : float  — separation between the two mixture component means
    pi_paired   : array  — mixture weights for paired queries; default [1, 0] (component 1 only)
    pi_unpaired : array  — mixture weights for unpaired queries; default [0, 1] (component 2 only)
                           set both to [0.5, 0.5] for no mismatch
    """
    # two mixture components separated by d_sep along the first axis
    mu_c    = np.array([[d_sep / 2] + [0] * (d_act - 1),
                        [-d_sep / 2] + [0] * (d_act - 1)], dtype=float)
    Sigma_c = [np.eye(d_act), np.eye(d_act)]

    if pi_paired is None:
        pi_paired   = np.array([1.0, 0.0])   # paired queries from component 1
    if pi_unpaired is None:
        pi_unpaired = np.array([0.0, 1.0])   # unpaired queries from component 2

    # model offsets — drawn once, fixed across queries
    model_offsets = np.random.normal(0, 1, (n_models, d_act))

    n_paired   = int(alpha * n_queries)
    n_unpaired = n_queries - n_paired

    data = []

    # paired queries: same query drawn for all models
    for query_idx in range(n_paired):
        c         = np.random.choice(2, p=pi_paired)
        query_vec = np.random.multivariate_normal(mu_c[c], Sigma_c[c])
        for model_idx in range(n_models):
            mean_response_vec  = np.random.normal(
                                     query_vec + model_offsets[model_idx], 1/s, d_act)
            model_response_vec = np.random.normal(mean_response_vec, 1/t, d_act)
            noise              = np.random.normal(0, 1, d_obs - d_act)
            data.append({
                "query_idx"          : query_idx,
                "model_idx"          : model_idx,
                "is_paired"          : True,
                "query_vec"          : query_vec,
                "model_response_vec" : model_response_vec,
                "model_observed_vec" : np.hstack([model_response_vec, noise]),
            })

    # unpaired queries: each model gets its own independent query draw
    for model_idx in range(n_models):
        for query_idx in range(n_paired, n_paired + n_unpaired):
            c         = np.random.choice(2, p=pi_unpaired)
            query_vec = np.random.multivariate_normal(mu_c[c], Sigma_c[c])
            mean_response_vec  = np.random.normal(
                                     query_vec + model_offsets[model_idx], 1/s, d_act)
            model_response_vec = np.random.normal(mean_response_vec, 1/t, d_act)
            noise              = np.random.normal(0, 1, d_obs - d_act)
            data.append({
                "query_idx"          : query_idx,
                "model_idx"          : model_idx,
                "is_paired"          : False,
                "query_vec"          : query_vec,
                "model_response_vec" : model_response_vec,
                "model_observed_vec" : np.hstack([model_response_vec, noise]),
            })

    data   = pd.DataFrame(data)
    dist_gt = squareform(pdist(model_offsets))

    return data, dist_gt
```

**Experiment instantiation.**

```python
# S0: paired vs unpaired vs combined — vary alpha, m_total as color, fixed d_sep=2
for m_total in [200, 500, 1000]:
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        data, dist_gt = generate_data(d_act=10, d_obs=50, n_models=20,
                                      n_queries=m_total, alpha=alpha,
                                      s=2, t=2, d_sep=2.0)

# S1: exchange rate — vary epsilon via pi_paired/pi_unpaired interpolation
for eps in np.linspace(0, 1, 10):
    pi_p = np.array([0.5 + eps/2, 0.5 - eps/2])
    pi_u = np.array([0.5 - eps/2, 0.5 + eps/2])
    data, dist_gt = generate_data(d_act=10, d_obs=50, n_models=20,
                                  n_queries=1000, alpha=0.1,
                                  s=2, t=2, d_sep=1.0,
                                  pi_paired=pi_p, pi_unpaired=pi_u)

# S2 (appendix): kernel expressiveness — vary d_sep, alpha as x-axis
for d_sep in [0, 1, 2, 4, 8]:
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        data, dist_gt = generate_data(d_act=10, d_obs=50, n_models=20,
                                      n_queries=1000, alpha=alpha,
                                      s=2, t=2, d_sep=d_sep)
```

#### 4.2 Experiment S0: Paired vs. Unpaired vs. Combined

**Question.** Does the combined estimator dominate both paired-only and unpaired-only across the full range of paired fractions, and does this hold across different query budgets?

**Design.** Fix $C = 2$, $d = 2$ (moderate mismatch), $s = 2$, $t = 2$. Vary paired fraction $\alpha \in \{0.0, 0.05, 0.1, 0.2, 0.5, 1.0\}$. Repeat for three total query set sizes $m_{\text{total}} \in \{200, 500, 1000\}$, shown as separate colors. Evaluate MSE against $D^*$ for three estimators:

- **Paired-only**: uses only the $\alpha \cdot m_{\text{total}}$ shared queries.
- **Unpaired-only**: uses only the $(1-\alpha) \cdot m_{\text{total}}$ unshared queries per model.
- **Combined** ($\kappa_{\alpha,w}$): uses all queries with the composed kernel.

Run 50 random seeds per condition for error bars.

**Key claims.** (i) Combined dominates both baselines for all $\alpha \in (0,1)$ at all query set sizes — the composed kernel always extracts more signal than either component alone. (ii) The gain from combining is largest at small $\alpha$ (few paired queries) — the regime most relevant in practice. (iii) At $\alpha = 0$ combined and unpaired-only coincide; at $\alpha = 1$ combined and paired-only coincide. (iv) The relative benefit of combining persists across all $m_{\text{total}}$ values, confirming it is not a finite-sample artifact.

#### 4.3 Experiment S1: Exchange Rate vs. Theory

**Question.** Does the empirical exchange rate between paired and unpaired observations match the theoretical prediction of Theorem 2, and does it provide actionable budget guidance?

**Design.** Fix $C = 2$, $d = 1$ (moderate mismatch), $s = 2$, $t = 2$. Vary query overlap $\varepsilon$ by interpolating the paired/unpaired mixture weights from fully shared ($\varepsilon = 1$) to disjoint components ($\varepsilon \approx 0$). For each $\varepsilon$, fix $m_p \in \{10, 25, 50, 100\}$ and find the smallest $m_u$ such that the combined estimator achieves the same MSE as paired-only with $m_p$. Report empirical exchange rate $m_u / m_p$ as a function of $\varepsilon$ and overlay the theoretical prediction $1/\varepsilon^2$.

**Key claims.** (i) Empirical exchange rate matches $1/\varepsilon^2$ closely across all $m_p$ values. (ii) Practitioners can use the theoretical rate to plan data collection: given a target MSE and knowledge of query overlap, the required unpaired budget is determined.

#### 4.4 Experiment S2: Nonlinear Query Kernel vs. Linear MMD

**Question.** When queries come from a mixture, does the RBF query kernel ($\kappa^*$ from Theorem 3) outperform the constant kernel (linear MMD), and does the gap grow with component separation?

**Design.** Fix $C = 2$, $\alpha = 0$ (fully unpaired), coverage-adjusted $\kappa_w$ applied to both methods. Vary component separation $d \in \{0, 0.5, 1, 2, 4, 8\}$. Compare:

- **Linear MMD** ($\kappa \equiv 1$): constant query kernel, marginalizes over queries.
- **RBF query kernel** ($\kappa^* = \exp(-\|q-q'\|^2/2\sigma^{*2})$): bandwidth set to theoretical optimum $\sigma^{*2} = (s^{-2} + t^{-2})/\|v_i - v_k\|^2$, estimated from data.

**Key claims.** (i) At $d = 0$ (single Gaussian), RBF and linear MMD are equivalent — query structure carries no information. (ii) As $d$ grows, RBF outperforms linear MMD — similar queries invoke similar responses (A3), and upweighting similar query pairs extracts more signal. (iii) The gap matches the theoretical prediction from Theorem 3 that linear MMD is optimal only when A3 is poorly satisfied.

### Block II: Benchmark Score Prediction

#### 4.6 Experiment R1: Prediction from Partial Queries

**Objective.** Show that adding unpaired observations improves benchmark score prediction over paired-only, especially when shared queries are few.

**Setup.** 25–30 models spanning Llama, Mistral, Qwen, Gemma, and Phi families. 500 queries drawn from MMLU. Vary α ∈ {0.05, 0.1, 0.2, 0.5, 1.0}. For each α, compute the following five methods: paired-only, unpaired-only, unpaired (imputation), combined, and combined (imputation). "Imputation" variants use LLM-imputed queries (best inverter from S3) in place of observed queries for coverage weight estimation; the distance computation itself is identical. Fit ridge regression from embedding coordinates to benchmark scores using a train/test split (80/20 over models, repeated 20 times with different random splits).

**Benchmarks.** MMLU, GSM8K, HumanEval, ARC-Challenge (scores from Open LLM Leaderboard).

**Evaluation.** Test-set R² and MAE for each (method, benchmark, α), averaged over random splits. Primary figure: R² vs. m_p/m for each method, with error bands over splits and random seeds. Additionally:

- *Effective sample size.* For each benchmark, find the number of paired-only queries m_p^* that matches the combined estimator's R² at each α. Report the effective sample size ratio m_p^* / m as a function of α, interpreted as the empirical exchange rate. Compare to the theoretical prediction from Theorem 2.
- *α* calibration.* At each α, compare the theoretically optimal weight α* = m_p/(m_p + m_u) against the α that minimizes empirical held-out error. Agreement validates the variance scaling assumptions; systematic deviation identifies where they fail.

#### 4.7 Experiment R2: Model Family Recovery

**Objective.** Confirm the embedding captures model family structure in the partially-paired regime.

**Setup.** Same model set as R1. ARI, NMI, and Silhouette for k-means clustering vs. ground-truth family labels, as a function of α.

### Block III: Model Auditing

#### 4.8 Experiment R3: Detecting Training Data Exposure

**Objective.** Show that the embedding detects whether a model has been fine-tuned on or given retrieval access to a specific document corpus, using only black-box query access.

**Framing.** Given a reference population of base models and a set of exposed models (fine-tuned or RAG-augmented on a target corpus), does the embedding separate exposed from unexposed? This is relevant for copyright compliance auditing, data contamination detection, and monitoring for unauthorized capability acquisition.

**Design.**

*Corpora.* Three target corpora with distinct topical signatures: (i) a domain-specific scientific corpus (PubMed abstracts from a specific subfield), (ii) an out-of-copyright literary corpus, (iii) a code corpus (a specific GitHub library).

*Models.* For each corpus: base model (Llama-3-8B), fine-tuned variant (1–3 epochs), RAG-augmented variant (retrieval access at inference time). Nine models total: 3 corpora × 3 conditions.

*Queries.* Two sets per corpus: **probe queries** whose answers require exposure to the corpus, and **control queries** unrelated to the corpus. Query sets generated independently of the models to avoid contamination.

*Methods.* Five methods as in R1: paired-only, unpaired-only, unpaired (imputation), combined, and combined (imputation). For the imputation variants, queries are withheld entirely for one corpus and imputed via M_inv.

*Inverter sensitivity.* For combined (imputation), repeat the full evaluation across a suite of modern language models used as inverters. Report AUC as a function of m_p/m for each inverter — this directly populates Figure 4 right column.

**Evaluation.**

- Primary: test-set AUC of a linear classifier distinguishing exposed from unexposed on the embedding, using an 80/20 train/test split over models, repeated 20 times.
- Ablation: AUC using probe queries only vs. control queries only. Claim: signal is in probe queries; control queries yield AUC ≈ 0.5.
- Sanity check: embedding distance between fine-tuned and base model as a function of fine-tuning epochs — should decrease monotonically.
- Fine-tuning vs. RAG: reported separately. If indistinguishable, frame as a feature for auditing (both detected) and limitation for interpretability (mechanism unknown).
- *Effective sample size.* As in R1, find the paired-only query count m_p^* that matches combined AUC at each α. Report the empirical exchange rate and compare to Theorem 2. Because probe and control queries differ in their informativeness for this task, we expect the exchange rate to vary by query type — probe queries from the same distribution should yield ε ≈ 1, while mixing probe and control queries should yield a lower ε.
- *α* calibration.* Compare theoretical α* = m_p/(m_p + m_u) against empirical optimum for AUC.

#### 4.9 Experiment R4: Probe Query Budget

**Objective.** How many probe queries are needed for reliable exposure detection?

**Design.** Using R3 setup, vary number of probe queries per model m ∈ {10, 25, 50, 100, 200, 500}. Report AUC and Procrustes similarity to the full-data embedding as a function of m. Repeat 20 times with different random subsamples.

**Expected result.** AUC > 0.9 with O(50–100) probe queries, demonstrating practical deployability.

---

## 5. Paper Structure

```
1. Introduction                                (~1 page)
   - Models in the wild: no controlled queries
   - The identifiability problem: model difference vs. query difference
   - Two applications: benchmark prediction, auditing
   - Contributions

2. Background                                  (~0.5 page)
   - Paired DKPS
   - Linear MMD and its connection to DKPS

3. Method                                      (~2 pages)
   3.1 Background: paired DKPS
   3.2 General kernel framework: κ on queries, 𝒦 on responses; special cases
   3.3 Coverage adjustment: κ_w as identifiability fix; combined kernel κ_α
   3.4 Estimating coverage weights: PCA + KDE
   3.5 LLM query imputation for the no-query regime

4. Theory                                      (~2 pages)
   4.1 General kernel results: bias (Prop. 1), consistency (Prop. 2),
       variance reduction (Thm. 1), exchange rate (Thm. 2), small-paired corollary
   4.2 Optimal query kernel for linear MMD (Thm. 3)
   4.3 Optimal combination weight + data-adaptive α corollary

5. Experiments                                 (~3 pages)
   5.1 Synthetic: kernel expressiveness (S0), exchange rate (S1), query kernel (S2)
       → Figure 2
   5.2 Benchmark prediction: five methods × benchmarks × embedding functions (R1, R2)
       → Figure 3
   5.3 Model auditing: five methods × corpora × inverters (R3, R4)
       → Figure 4

6. Related Work                                (~0.5 page)

7. Conclusion                                  (~0.25 page)

Appendix
   A. Proofs
   B. Experiment S3 (inverter robustness in synthetic setting)
   C. Full benchmark prediction tables
   D. Additional auditing results (fine-tuning vs. RAG, per-corpus breakdown)
   E. Inverter prompts and implementation details
   F. PCA dimensionality selection details
   G. 2-d PCA response visualizations colored by paired/unpaired status (synthetic, benchmark prediction, auditing)
   H. RBF query kernel vs. linear MMD: MSE vs. component separation d, both coverage-adjusted (S2, Theorem 3)
```

**Page budget.** ICLR allows 9 pages + references. Structure above targets ~8.75 pages.

**Figure plan.** All multi-panel figures are full width.

- **Figure 1** (schematic, 2 panels): *(A)* 2-d PCA projection of pooled responses colored by paired/unpaired status at a representative small α, with points shaped by model identity. Illustrates the coverage structure directly in response space — paired observations are a sparse subset of the full response cloud — motivating the coverage weight construction. *(B)* Three side-by-side 2-d MDS embeddings of the real model set (R1) at the same α, colored by model family — paired-only, combined, and combined (imputation). Paired-only fails to recover family structure; combined and combined (imputation) recover it.

- **Figure 2** (synthetic results, 3 panels): *(A)* MSE vs. paired fraction $\alpha$ for paired-only, unpaired-only, and combined estimators (S0), with total query set size $m$ as color — combined dominates both baselines at all $\alpha$ and $m$, with the largest gain at small $\alpha$; boundary conditions at $\alpha=0$ and $\alpha=1$ confirm the estimator reduces correctly. *(B)* Empirical exchange rate $m_u/m_p$ vs. query overlap $\varepsilon$ with theoretical prediction $1/\varepsilon^2$ overlaid (S1) — theory matches empiric, providing actionable budget guidance. *(C)* MSE vs. paired fraction $\alpha$ for linear MMD, coverage-adjusted linear, and coverage-adjusted kernel MMD, with component separation $d$ as color — at $d=0$ all methods coincide; as $d$ grows color separation reveals that more severe mismatch demands a more expressive kernel; coverage-adjusted kernel remains calibrated across all $\alpha$ and $d$.

- **Figure 3** (benchmark prediction, 3 panels): *(A)* R² vs. m_p/m for a single benchmark (MMLU) — paired-only, unpaired-only, and combined. *(B)* R² vs. m_p/m for combined across all benchmarks. *(C)* R² vs. m_p/m for combined across multiple sentence embedding functions g.

- **Figure 4** (model auditing, 2 rows × 3 panels): Rows correspond to the two exposure mechanisms: fine-tuning (top) and RAG (bottom). *(Left)* AUC vs. m_p/m for a single corpus — paired-only, unpaired-only, unpaired (imputation), combined, and combined (imputation). *(Middle)* AUC vs. m_p/m for combined and combined (imputation) across all corpora. *(Right)* AUC vs. m_p/m for combined (imputation) across a suite of modern language models used as inverters. The two-row layout directly answers whether the embedding detects exposure mechanism-agnostically or differentially for fine-tuning vs. RAG.

---

## 6. Key Claims

**C1.** The combined estimator achieves strictly lower error than either the paired-only or unpaired-only estimator whenever both types of data are available and query distributions overlap. *(Theorem 1, S1, R1)*

**C2.** Without coverage adjustment, the unpaired estimator is biased when query distributions differ, conflating query distribution difference with model difference. The coverage-adjusted estimator removes this bias. This is the minimal condition for identifying model similarity. *(Proposition 2, S2)*

**C3.** LLM query imputation makes the coverage adjustment applicable in the no-query regime, and downstream accuracy is robust to the choice of inverter model. *(S3, R3 no-query condition)*

**C4.** The embedding captures training data exposure, not just aggregate capability. Signal concentrates in probe queries and is absent in control queries. *(R3 ablation)*

---

## 7. Open Questions and Risks

- **Theory in high dimensions.** Asymptotic results are for fixed p; in practice p = 768 >> n (models) and often >> m (queries). We work in the d_PCA-dimensional PCA subspace, but theory needs to be stated in this space explicitly. Options: prove results in d_PCA dimensions and treat projection as preprocessing, or use effective dimension arguments. Note that (A4) reduces the high-dimensional density estimation problem to a standard KDE consistency result in d_PCA dimensions, so the main remaining question is how d_PCA scales with m_u for the asymptotic results to hold.

- **First-moment limitation of linear MMD.** If empirical results are weak — particularly for tasks where models differ primarily in response diversity or shape rather than conditional means — the method should be enriched to kernel MMD or energy distance. Both are straightforward extensions and the coverage imputation framework carries over unchanged; only the distance computation changes.

 An adversary who knows the auditing pipeline could fine-tune a model to produce responses that impute to misleading queries, evading coverage adjustment. State this explicitly as a limitation.

- **Fine-tuning vs. RAG distinguishability.** An open empirical question. Report separately in R3 and let the result stand.

- **Exchange rate theory vs. practice.** The theoretical rate 1/ε² may not match the empirical rate if (A3) is poorly satisfied. A gap is a potential extension, not a blocker.
