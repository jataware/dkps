# Working narrative: Embedding Agentic Traces

Five-act narrative (HH, 2026-08-18), with the evidence map and outstanding
gaps. Numbers reference FINDINGS.md (F1-F19).

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

## Figure/table plan (ICML, revised 2026-08-18)

Central claim for exhibits: across many embedding models, qubric processing is
(1) MORE sensitive to axes that matter (outcome/behavior) and (2) LESS
sensitive to axes that don't (task, provenance, lineage, format).

- Fig 1: methods schematic (trace -> render -> qubric -> section embeddings ->
  centering -> perspective space; nuisance axes annotated where they die).
- Fig 2: RADAR CHARTS, one panel per embedding model (~7: 6 local
  [nomic-1.5, bge-large, gte-large, e5-large, mpnet, MiniLM] + cached
  text-embedding-3-small). Spokes = chance-normalized axis sensitivities;
  two polygons per panel (raw vs qubric). Same shape-change in every panel =
  the embedder-general claim. Data: data/multiembed_*.npz (local GPU).
- Fig 3: sensitivity to the processing model -- axis profiles + prediction
  accuracy across judge models (4o-mini/5.4-nano/5.4-mini) and rubric
  constructions (blob/generic/verdict/qubric). All cached.
- Fig 4: QUENCH-style probe-efficiency validation (exists:
  error_vs_probes_qspec_pkps.png).
- Fig 5 (proposed): measurement error & aggregation -- identity-recovery and
  prediction vs #instances aggregated; cross-judge (mini/nano) reliability.
- Table 1: axis-ladder numbers behind Fig 2 (main or appendix).
- Table 2: headline budget table (sample / geometry / ensemble / +selection).
- Appendix: CV-then-freeze ablation, PKPS-vs-paired equalized twins,
  mediation controls, greedy-picks interpretability, protocols.

## Gaps / risks (state of 2026-08-18)

- **Scale**: all qubric results are on the 20-instance panel; q150
  confirmation requires ~14K judge calls -- BLOCKED (no OpenAI API access).
  Either scope claims to the panel with explicit error bars, or wait.
  Head/tail results are at q150 already.
- **Single benchmark / single embedder**: SWE-bench only;
  text-embedding-3-small only. Cached-data mitigations available:
  re-embed cached judge texts with local nomic (embedder robustness);
  small-cohort replicate axis via local nomic (gold-standard similarity +
  second corpus for the transfer claim).
- **Reliability number**: test-retest (same judge, resampled) blocked;
  cross-judge agreement (mini vs nano caches) computable now as a proxy.
- Rank-64 PCA chosen on-panel; fold into CV at scale-up.
- Naming: "qubric" (query-specific rubric) -- decide before writing.

## Evidence ledger (claim -> artifact)

- Modality census: FINDINGS §1; loader handles 10+ formats (leaderboard.py).
- Naive equivalence class: F4; feasibility: F4 + embedding-cost notes.
- Nuisance dominance + leakage: F13, F19 table (figures/…, scripts inline).
- Quotient results: F19; construction ranking: F15/F16; judge gradient: F9+nano.
- Benchmarking utility: F1, F2, F15, F17, F18; figures error_vs_probes_*,
  pipeline_final; data JSONs in figures/.
