# Agentic-Traces DKPS: Experimental Findings Log

Working log of the agentic-traces extension (2026-08-10/11). Self-contained:
protocols, exact representation definitions, all headline numbers, and negative
results. Everything reproduces from the embedding caches + seeds below without
re-calling any API.

## 1. Data

**Small cohort** (`/home/ubuntu/20260729_traces.zip`, extract to `data/traces/`):
swe-agent on SWE-bench Verified, Langfuse span JSONL. 14 models x 12 instances
x 5 replicates = 839 traces (Nemotron-3-Nano missing replicate 5 of
sympy__sympy-14248). No correctness labels; `exit_status` is termination reason
only (7/14 models submit 100% -- saturated). Loader: `dkps.traces.langfuse`.

**Leaderboard corpus** (`data/leaderboard/verified/`, synced from
`s3://swe-bench-submissions/verified/<submission>/` with `--no-sign-request`;
note key prefix has NO `evaluation/`): 134 Verified submissions; 119 have
trajectories (mandatory since 2024-07); official per-instance resolved labels
harvested from the github.com/swe-bench/experiments git repo into
`data/leaderboard/verified_labels.json` (all 134). Loader:
`dkps.traces.leaderboard` (format-agnostic text render; handles dir-style
trajectories, `instance_` filename prefixes, .log/.yaml/.txt extensions).

**Panels** (fixed seeds; reproduce exactly):
- `q418` = sorted intersection of instance ids across the 107 gate-passing
  systems (>= 480 traj files + labels).
- `q150` = `np.random.default_rng(0).choice(q418, 150, replace=False)` sorted.
- `q20`  = `np.random.default_rng(1).choice(q150, 20, replace=False)` sorted.
- Head/tail embeddings exist for all 107 x q150 (+6 partial systems x q20/q150
  coverage); chunk embeddings and judge descriptions for 107(+6) x q20.

**Embedding caches** (gitignored, on this machine):
- `.dkps_cache_lb/<system>/<instance>.<cfg>.npz` where cfg =
  sha1(`openai/text-embedding-3-small|headtail8000`)[:8] holds `head`,`tail`
  (8K-token tiktoken slices); cfg for `|chunks4000` holds `chunks` (4K-char
  line-aligned chunks, full coverage). Embedder: OpenAI text-embedding-3-small
  (1536-d), token-bucket-paced (measured account limit: 5M TPM / 10K RPM).
- Judge descriptions: `data/judge/<judge-model>/<system>/<instance>.txt`
  (+`_emb.npz` for gpt-5.4-mini); prompt in `scripts/judge_describe.py`
  (describe behavior, not problem; report observed verification outcomes).
- Query vectors: `data/leaderboard/query_vecs_64.npz` = PCA-64 of embedded
  problem statements for q418.
- Small-cohort cache `.dkps_cache/` via `scripts/embed_traces.py`
  (nomic-embed-text-v1.5 local; NOTE this host's NVML is broken -- CUDA OOM
  asserts instead of recovering; embedder has small-batch long-text tier +
  halve-batch backoff for this).

## 2. Evaluation protocols

Target y = official resolve rate (|resolved|/500). Readouts:
- **kNN**: k=3, inverse-distance-weighted, on the DKPS distance matrix
  (`model_distance_matrix`: mean over replicates, flatten instances, pdist /
  sqrt(n_queries)).
- **ridge**: kernel-form ridge on flattened per-instance representations,
  lambda in {1,10,100}, best reported.

Reference protocols (who may serve as neighbors/training rows for a target):
- LOO: everyone else. Inflated by family leakage -- do not headline.
- **leave-one-LLM-out (STANDARD, user decision 2026-08-11)**: exclude systems
  sharing the target's underlying LLM (`model_display` tag from metadata.yaml).
  Same-scaffold references allowed (leave-family-out deemed too conservative;
  scaffold siblings cost only ~0.007 vs ~0.018 for LLM siblings).
- Ensemble alpha (blend with sample score) selected per target from its allowed
  references only (honest); grid alpha in linspace(0,1,21).

Context baselines (107 systems, LLM-out): predict-the-mean 0.136; date-3NN
0.114/rho .51; same-LLM-nearest-date 0.084 (requires LLM identity + sibling --
unavailable for genuinely new models). Oracle kNN floor (score-proximal
neighbors): 0.004 -- ceiling is not the constraint.

## 3. Best-known pipeline (as of 2026-08-11)

For a new system evaluated on m instances (references pre-cached):
1. Render trace to text (`load_leaderboard_trajectory`), take first & last 8K
   tokens, embed (text-embedding-3-small).
2. Judge-describe each trace (gpt-5.4-mini, behavior-only prompt), embed the
   description.
3. Per instance, subtract the reference pool's **median** vector (consensus
   centering), L2-normalize the residual; fuse [head+tail | judge] blocks at
   unit RMS each.
4. Distances via **PKPS** (product kernel: RBF over PCA-64 problem-statement
   embeddings x linear response kernel) so the target's m instances compare
   against references' full coverage. Paired DKPS when target coverage is full.
5. kNN(k=3) prediction over LLM-out references; blend with the target's sample
   score, alpha chosen on references.
6. Optional: evaluate on greedily selected informative tasks instead of random.

Numbers (q20 panel, 107 targets, LLM-out, 200 subsets):

| m | sample | pipeline (random tasks) | + selected tasks (held-out half) |
|---|--------|------------------------|----------------------------------|
| 1 | 0.438  | 0.095                  | 0.095                            |
| 3 | 0.208  | 0.088                  | 0.076                            |
| 5 | 0.157  | 0.082                  | 0.070                            |
| 10| 0.103  | 0.070                  | 0.057                            |
| 20| 0.061  | 0.050-0.053            | 0.057                            |

Layer ablations at m=1: naive paired geometry 0.107 -> judge-fused 0.105 ->
PKPS full-coverage refs 0.095. PKPS@m=1 ~= paired@m=20 for the geometry
component (0.098 vs 0.096). Selection halves the budget mid-range (m=10
selected ~= m=20 random); selection overfits its pool (in-pool 0.041-0.058 vs
held-out 0.057-0.070) -- always report held-out.

## 4. Findings (numbered for citation)

**F1. One trace nearly places a system.** Single-query model-distance matrices
correlate r~0.83 with full-panel distances; prediction error is nearly flat in
n_queries (small cohort AND 107-system corpus). Replicates do early work: 1
query x 1 rep 0.109 vs x5 reps 0.089 (small cohort, localization target).

**F2. Reference pool, not query budget, is the binding resource.** Error vs
n_models still declining at 13 (small cohort) and at ~100 (leaderboard).
Supervised channel weighting fails honestly at 13 refs (capacity exists:
in-sample 0.028 vs honest 0.083+) but ridge succeeds at ~97 refs.

**F3. Identity vs competence dissociation.** Action n-grams = best model
fingerprint (replicate-consistency 0.653); text/outcome surface = best
competence signal (Mantel r vs localization: whole 0.62, outcome 0.58, action
0.38). Small cohort.

**F4. Naive representations sit at the unsupervised ceiling.** On the honest
protocol, an equivalence class all lands ~0.078 ridge / ~0.095-0.10 kNN
(q150/q20 panels): head+tail embedding, chunk-mean, rubric sections (hard/soft,
any normalization), positional pyramid, MMD-RBF chunk distributions,
trajectory-dynamics features, displacement (tail-head), relational profiles,
per-instance PCs, global PCA (r>=16), hashed path footprint. Fusions of these
with the naive rep move nothing. Tail >> head (head barely beats date-only).

**F5. Consensus centering is the one geometry trick that matters.**
Per-instance median-centering + L2 of the residual direction: 0.086 -> 0.077
LOO (0.103 -> 0.095 LLM-out). Anchor must be global & robust: median > mean >
trimmed; local (kNN) anchors and PC-removal destroy signal monotonically --
the dominant modes of cross-system variation ARE the signal. Global PCA's top
components encode instance identity, not system behavior (r=2 collapses).

**F6. Symbolic footprint equivalence.** A 512-d hashed bag of file paths (no
embedding model) matches the 3072-d text embedding under both readouts.
Together with F4: what traces reveal is mostly where the system went and what
it produced, not the prose.

**F7. Gold-file localization is a label-free competence proxy.** Fraction of
gold-patch files touched (gold patches are public): spread 0.39-0.82 across
models, credible ranking, MAE 0.086 as a 1-dim/instance representation.
`dkps.traces.metrics`.

**F8. Ensemble with the sample score dominates at every budget.** Honest
alpha; see section 3 table. Even at full panel the geometric prior improves
the sample score (0.034-0.035 vs 0.039 on q150) -- shrinkage against sampling
noise. alpha* slides 0.09 -> 0.85 with m.

**F9. Judge-describe adds complementary signal, scaling with judge quality.**
Describe-then-embed (behavior-only prompt): gpt-4o-mini descriptions are worse
than naive standalone; gpt-5.4-mini's beat naive on kNN (0.097 vs 0.102), lose
on ridge (0.087 vs 0.078), and fused improve both (ridge 0.075; +2-4% through
the whole pipeline at every budget). Only source found whose value has a
quality gradient rather than flat equivalence. Frontier-judge rung untested.

**F10. PKPS earns its keep exactly in sparse coverage.** Target on m
instances, refs on full coverage: 10% geometry gain at m=1; converges to
paired as m grows; slight bias penalty at full coverage. On fully-paired data,
unpaired/PKPS ~= paired (parity, as theory predicts). Class name in code:
`UnpairedDKPS` (`dkps/unpaired_dkps.py`, from jataware/dkps@unpaired-20260401);
paper name: Product Kernel Perspective Space (PKPS).

**F11. Partial-coverage references are includable but not yet helpful.** +6
systems (14-19/20 coverage) via masked PKPS: slightly hurts geometry (coverage
mismatch biases their distances), ~wash for ensemble. Untried fix: the
`use_coverage=True` KDE adjustment in UnpairedDKPS; mismatch also shrinks on
larger panels.

**F12. Task selection transfers.** Greedy forward selection (objective:
ensemble MAE on half the systems) transfers to the held-out half: ~13-17%
better than random subsets at m=3-10; m=10 selected ~= m=20 random. Selection
order is django-heavy (densest instance population).

**F13. Family leakage inflates naive evaluations.** LOO -> LLM-out costs
~0.018 MAE (0.077 -> 0.095); scaffold siblings only ~0.007 more. Most of the
"performance" of metadata baselines is sibling score-copying; the strongest
(same-LLM nearest-date, 0.084) is unavailable for new models by construction.

**F14. Small-cohort results (localization target, pre-labels).** DKPS beats
direct eval below ~half the query budget; error-vs-queries flat (F1); channel
ablations (F3); exit_status is not a score.

**F15. Query-specific rubric extraction BEATS naive under both readouts (the
representation win).** Construction: per instance, LLM writes a 6-section
rubric from the problem statement (what understanding/localization/
reproduction/editing/verification/final_state specifically mean for THIS
issue; `data/judge/qspec_rubrics/`); a judge (gpt-5.4-mini) extracts 30-60
words per section from each trace against that rubric (JSON;
`data/judge/structured-qspec/`); sections embedded separately, per-
(instance,section) median-centered, L2, concatenated (6x1536). q20 panel,
LLM-out: knn 0.0768/0.804 vs naive 0.1022/0.613 (-25%); ridge 0.0758/0.821 vs
0.0779/0.780. Fused with naive: ridge 0.0708/0.837. Ensemble (PKPS + honest
alpha): new bests at every m>=3 (0.083/0.078/0.067/0.049 at m=3/5/10/20);
at m=1 generic reps transfer better (instance-calibrated contrasts don't
cross instances). Scale confirmation on q150 NOT yet run.

**F16. qspec audit & mechanism.** (a) Construction ranking, monotone in
information retained: descriptive qspec > verdict-style questions (0.083/
0.086) > fixed rubric (0.093/0.081) > free-form blob (0.095/0.083) -- forcing
YES/NO verdicts LOSES signal; the latent variable is graded behavioral
completeness, not a checklist. (b) Verbosity mediation, not leakage: system
mean extraction length correlates rho=.59 with score, but length-only is weak
(0.119/0.103), naive+length ~= naive, and residualizing length out of qspec
destroys it (over-control: length and semantics are surface forms of the same
behavioral facts). Keep the 6 length counts as a free auxiliary channel;
do not "correct" for them. (c) Sections individually redundant, collectively
robust; verification strongest for knn (0.091/0.74), editing for ridge
(0.082/0.80), understanding ~dead weight. (d) Rubric/extraction lengths well
controlled by prompt (27-57 / mean 39 words); parse failures ~5/2140.

**F17. Probe selection: CV-then-freeze, not CV-inside-search.** Greedy probe
selection (objective: ensemble MAE on half the systems; evaluated held-out)
improves the ensemble at every budget when hyperparameters are frozen first
(CV'd once, then fixed: sigma=med/16, k=5): m=5 0.074 -> 0.057 (-23%). Running
CV(sigma, k) inside every candidate evaluation halves the gains (m=5 0.070)
and destabilizes choices -- joint optimization over probe set x hyperparams
overfits a 53-system selection pool. Same degrees-of-freedom lesson as the
13-ref weight-learning failure. First frozen-greedy picks are interpretable:
medium-difficulty, high item-total-r instances from four repos, plus one
near-unsolvable instance (5% solve rate, item r=.03) picked for its trace
geometry -- selection exploits information item-response statistics cannot see.

**F18. Post-centering PCA is a real pipeline stage.** Global PCA on the
centered qubric/fusion tensors (fit on the stacked (systems x instances)
residuals) improves BOTH readouts at rank 64-128 (~5-6% flat-panel; budget
sweep: monotone gains growing with m, geometry 0.0745 -> 0.0690 and ensemble
0.0516 -> 0.0476 at m=20 -- pipeline under 0.05 for the first time). Works
here though it failed on raw embeddings (F4-era) because centering first
removes instance identity, so PCA extracts a shared cross-instance behavioral
basis instead of "which problem is this". Order matters: center -> PCA. Query-
side embeddings are also PCA'd (64-d) before the PKPS RBF kernel; sensitivity
check: kernel matrices stable for query dims 32 -> raw-1536 (r ~ .94-.96,
median-bandwidth self-adapts), materially distorted at 8-16d. Rank not yet in
the CV grid (chosen on-panel at 64; flag for the q150 confirmation).
ADDENDUM: PCA does NOT compound with greedy probe selection (greedy+PCA64
held-out: 0.083/0.064/0.057 at m=1/5/20 vs greedy full-d 0.078/0.057/0.053) --
selection and PCA spend the same slack; use PCA for random probes, full-d for
selected probes.

**F19. Similarity axes and the quotient ladder (intrinsic evaluation).**
Trace similarity decomposes into axes: task, provenance, lineage, format,
behavior, outcome. Trace-level nearest-neighbor metrics on q20 (chance rates:
task/provenance ~0.01, LLM 0.09, scaffold 0.06):

| axis            | raw   | centered | qubric-raw | qubric-cent |
|-----------------|-------|----------|------------|-------------|
| task NN         | 0.831 | 0.367    | 0.994      | 0.953       |
| provenance P@1  | 0.675 | 0.830    | 0.057      | 0.094       |
| same-LLM NN     | 0.215 | 0.201    | 0.128      | 0.117       |
| same-scaffold NN| 0.307 | 0.308    | 0.072      | 0.079       |
| outcome AUC     | 0.569 | 0.519    | 0.575      | 0.547       |

(a) The judge quotients the who-axes to ~chance; (b) it does NOT quotient the
task axis (rubric injects instance vocabulary; residuals stay instance-
conditioned, 0.95) -- the architecture handles task via within-instance
comparison + PKPS query kernel (also explains qubric's weak m=1-2 cross-
instance transfer); (c) centering alone AMPLIFIES provenance (0.68 -> 0.83);
(d) qubric's trace-level identity collapse is largely judge sampling noise:
10-instance aggregation recovers self-match 0.09 -> 0.52; (e) naive's
intrinsic scores are inflated by lineage leakage (sibling-NN 26%, 3x chance;
sibling-excluded score-gap advantage 0.014 -> 0.002). Intrinsic metrics must
be aggregation-aware and lineage-controlled.

## 5. Negative results (do not re-run without new ideas)

- Supervised channel-weight learning at 13 refs: five schemes all <= uniform.
- Chunk-level product kernels (position/section side kernels, any content
  centering): reproduce pooled special cases, never beat them.
- Local/neighborhood centering, per-dim z-scoring, PC-removal: hurt (F5).
- Dynamics/displacement/relational/conformity reps: redundant with naive (F4).
- gpt-4o-mini as judge: below naive standalone (F9).
- Rubric approaches: hard-assignment + centering TIES naive (LOO 0.083 both,
  q20 full panel); soft pooling loses; 6-dim mass profile within 0.015 of the
  3072-d baseline (remarkable per-dim, not a win). 150-instance confirmation
  deliberately skipped (cost/benefit).

## 6. Caveats

- q20-panel differences under ~0.01 are within noise; q150 differences under
  ~0.005 likewise. Selection/ensemble tables use paired subsets where possible.
- Judge + chunk experiments live on q20 only; head/tail on q150.
- The 20-instance selection table's held-out half is a different population
  than the all-107 random baseline (worth ~0.004; see m=20 row).
- Sample scores at m use official per-instance labels ("run the harness on m
  instances"); DKPS-only numbers require no harness at all.

## 7. Repo map (this branch)

- `dkps/traces/`: schema, langfuse + leaderboard loaders, canonicalize,
  channels (incl. rubric/whole), rubric.py, embedder (local + OpenAI, paced),
  metrics (localization), assemble (build_dkps_input, distances).
- `dkps/unpaired_dkps.py`: PKPS implementation (verbatim from jataware fork).
- `scripts/`: embed_traces, analyze_traces, smoke_test (small cohort);
  predict_localization, learn_channel_weights (small cohort experiments);
  leaderboard_baselines, leaderboard_chunks, leaderboard_reprs, rubric_viability,
  chunk_pkps, leaderboard_unpaired, judge_describe (leaderboard).
- `figures/`: perspective spaces + error curves (small cohort, share-ready).
- One-off experiment code (centering variants, ensemble sweeps, selection,
  PKPS budget sweeps) ran inline; recipes fully specified above.
