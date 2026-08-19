# Working narrative: Embedding Agentic Traces

Five-act narrative (HH, 2026-08-18), with the evidence map and outstanding
gaps. Numbers reference FINDINGS.md (F1-F19).

**Thesis line**: "A trace representation should be faithful to content and
invariant to authorship." Content = (task, behavior, outcome); authorship =
(system, LLM, scaffold, style). Radar orientation follows: task/behavior/
outcome/reliability spokes desirable-HIGH; provenance/LLM/scaffold spokes
desirable-at-CHANCE. Off-the-shelf embeddings represent who wrote the trace;
qubric represents what happened in it. (Centered-naive is the worst of both:
task fidelity destroyed, authorship amplified.)

## The narrative

**1. Agentic traces are a new evaluation modality.** Modern evaluation
produces not answers but *trajectories*: multi-step tool-call logs, tens of
thousands of tokens, produced by heterogeneous harnesses. The SWE-bench
Verified leaderboard archive alone holds ~119 systems x 500 instances of them,
in dozens of mutually incompatible formats. Nobody has a principled way to
compare them.

**2. They can be processed by off-the-shelf embedding models.** Rendered to
text, any trace fits the standard pipeline: tokenize, truncate or chunk,
embed. Feasibility is not the problem: an 8K-token slice of a median 40K-token
trace, or mean-pooled full-coverage chunks, both produce usable
representations -- and (importantly for the argument) all such variants
perform identically (F4: the naive equivalence class).

**3. But off-the-shelf embeddings have the wrong sensitivity profile.**
Trace similarity is not one relation; it decomposes into axes: task
(which problem), provenance (which system), lineage (which LLM/scaffold),
surface format, behavior (what it did), outcome (how well it did). Raw
embeddings are dominated by the nuisance axes: nearest neighbors are
same-instance 83%, same-system 68-83%, same-scaffold at 5x chance -- while
outcome coherence is barely above chance (AUC 0.52). Consequences are
measurable: evaluation numbers inflate ~25% through lineage leakage
(LOO vs leave-one-LLM-out, F13), and the classical fix (per-instance
centering) makes provenance dominance WORSE (0.68 -> 0.83), because removing
shared content spotlights style (F19).

**4. An LLM-based processing step re-shapes the sensitivity profile.**
The qubric construction: per instance, an LLM writes what each of six
behavioral roles specifically means for that problem (from the public problem
statement); a judge extracts, per trace, a short factual description per role;
sections are embedded separately, consensus-centered, concatenated.
- It quotients the who-axes almost exactly to chance (provenance 0.83 -> 0.09,
  scaffold 0.31 -> 0.08, LLM 0.20 -> 0.12) while raising outcome coherence
  (F19). Division of labor: the *representation* mods out who; the
  *perspective-space architecture* (within-instance comparison + PKPS query
  kernel) mods out which-problem, which the rubric text deliberately retains.
- The construction is a similarity-specification device: the rubric prompt is
  a natural-language definition of the equivalence relation the space should
  respect. Instance-conditioning is the active ingredient (qubric > generic
  rubric > verdict-style > free-form blob > raw, F15/F16); description quality
  scales with judge capability (4o-mini < 5.4-nano < 5.4-mini, F9 + nano test);
  the signal is graded behavioral completeness, mediated by (not confounded
  with) verbosity (F16).
- Costs: ~1 mini-judge call per trace + 1 rubric call per instance; per-trace
  measurement noise (single-trace retrieval is poor; aggregation over a few
  instances recovers it -- report reliability honestly).

**5. The re-shaped embedding is useful: query-efficient benchmarking.**
QUENCH-style validation with true leaderboard labels, leave-one-LLM-out:
- One probe trace ~ 8-10 scored probes; geometry nearly flat in probe count
  (F1); reference pool is the binding resource (F2).
- qubric beats every naive variant under both kNN and ridge (F15); fused with
  the trace-end slice it is best from 2 probes on.
- Full pipeline (fusion -> centering -> [PCA-64] -> CV'd PKPS -> alpha-blend
  with sample score -> optional greedy probes): 0.095 MAE at 1 probe, ~0.074
  at 5, 0.048-0.053 at 20, vs sample-score-alone 0.44/0.16/0.06. Every layer
  ablated (F17, F18); ensemble dominates the sample score at every budget.

## Figure/table plan (REVISED 2026-08-19 v2, per HH)

- Fig 1 (schematic, 3 panels): (a) anatomy of an agentic trace -- what is in
  it, which parts are authorship (harness boilerplate, tool syntax, model
  style) vs content (actions, findings, edits), annotated real excerpt;
  (b) qubric as the mod-out mechanism (problem -> rubric -> judge extraction
  -> per-section embed -> consensus center -> concat); (c) pillar taxonomy +
  independence evidence (heatmap moves here or to appendix with a pointer).
- Fig 2: sensitivity radars across embedding functions, qubric vs baselines
  (exists: radar_all.png; add bootstrap CIs; q150 scale-up API-blocked).
- Fig 3 (payoff): QUENCH-style analysis on SWE-bench AND Terminal-Bench.
  SWE-bench arm exists (fig4_quench.png). Terminal-Bench arm: trajectories
  ARE public -- HF dataset harborframework/terminal-bench-2-leaderboard
  (40GB, submissions/<agent>__<model>/ job dirs, >=5 trials/task, per-trial
  result.json) + abacusai/abacusai-terminal-bench-leaderboard (TB 1.0 logs).
  Data sync + loader + labels + local embedding: doable now. Rubric + judge
  calls: API-blocked. Bonus: >=5 trials/task = replicate axis for Fig 4 on a
  second benchmark.
- Fig 4: reliability (exists: fig5_reliability.png; judge test-retest
  API-blocked; TB trials add a second agent-stochasticity corpus).
- Fig 5 / Table 1: robustness to pipeline choices -- embedder (7, done),
  judge x construction matrix (judge_matrix.png moves here), centering (F5),
  PCA rank (F18), protocol LOO vs LLM-out (F13), CV-then-freeze (F17), plus
  computable-now sweeps: kNN k, PKPS sigma, alpha-blend, rubric-section
  ablation (sections separately embedded -- drop-one is cache-only), fusion
  weight.
- Baselines owed to related work (cache-only, from RELATED.md): LEACE-style
  authorship erasure on raw embeddings (Fig 2/appendix); CAPA
  correctness-only kernel (Fig 3 competitor).
- Table 1: pillar definitions + axis-ladder numbers.
- Table 2: headline budget table (sample / geometry / ensemble / +selection).
- Appendix: CV-then-freeze, PKPS-vs-paired twins, mediation controls,
  identity-beyond-type tests, independence checks, protocols.

## Gaps / risks (state of 2026-08-19)

**STANDING CONSTRAINT: no OpenAI API, permanently. All new LLM calls use
open-source models -- local vLLM on the A100-40GB (candidates: gpt-oss-20b,
Qwen3-14B, Qwen3-32B-AWQ) or OpenRouter (drop-in: scripts read
OPENAI_BASE_URL; needs an OpenRouter key). Cached gpt-5.4-mini/nano/4o-mini
judge outputs stay as comparison rows. This is a reproducibility UPGRADE for
the paper: the pipeline becomes fully open-weight.**

Migration plan (order matters):
1. Judge migration validation: re-run q20 extraction (2140 calls, existing
   mini-written rubrics, so only the extractor changes) with 1-2 open judges;
   score in the F21 matrix protocol (nomic embedder). Bar: beats naive both
   readouts, within noise of gpt-5.4-mini. Adds open rows to the judge matrix.
2. Open rubric writer (150 calls at q150 scale -- use a frontier open model
   via OpenRouter, e.g. DeepSeek/Kimi/Qwen3-235B; pennies): fills the
   rubric-writer axis of the matrix (mini-written vs open-written rubrics).
3. q150 scale-up with the validated open judge: ~16K calls x ~12K tokens.
   Local gpt-oss-20b overnight, or OpenRouter ($20-60 depending on model).
4. Judge test-retest (same open judge, temperature>0, resampled): unblocked.
5. Terminal-Bench rubrics + extraction with the same open stack: unblocked
   after corpus census.

Remaining risks:
- q20-panel resolution (~0.01 MAE) until step 3 lands.
- Judge switch mid-project: mitigated by step 1's paired comparison on
  identical rubrics + F21's finding that construction > extractor capability.
- Rank-64 PCA chosen on-panel; fold into CV at scale-up.
- Naming: "qubric" (query-specific rubric) -- decide before writing.

## Evidence ledger (claim -> artifact)

- Modality census: FINDINGS §1; loader handles 10+ formats (leaderboard.py).
- Naive equivalence class: F4; feasibility: F4 + embedding-cost notes.
- Nuisance dominance + leakage: F13, F19 table (figures/…, scripts inline).
- Quotient results: F19; construction ranking: F15/F16; judge gradient: F9+nano.
- Benchmarking utility: F1, F2, F15, F17, F18; figures error_vs_probes_*,
  pipeline_final; data JSONs in figures/.
