# Embedding Agentic Traces: Project Narrative

*Helivan Research — August 2026. Companion documents: `FINDINGS.md` (technical
log, F1–F20), `PAPER.md` (exhibit plan), `RELATED.md` (literature context and
annotated bibliography). This document is the story: what we are doing, what
we have done, and why it might matter.*

## What we are doing

AI evaluation used to produce short answers. It now produces *trajectories*:
an agent given a software issue works for tens of minutes, running commands,
reading code, editing files, running tests — and leaves behind a log tens of
thousands of tokens long. These traces are piling up in public archives (the
SWE-bench Verified leaderboard alone holds ~119 systems × 500 instances of
them, in dozens of incompatible formats), and they contain far more
information about the systems that produced them than the single pass/fail
bit each run is reduced to. Nobody has a principled way to use them.

The obvious move — feed traces to an off-the-shelf embedding model — turns
out to measure the wrong things. Our thesis, and the sentence the whole
project stands on:

> **A trace representation should be faithful to content and invariant to
> authorship.**

Content is what happened: which task, what the agent did, how it went.
Authorship is who did it: which agent, built on which model, running in which
harness. Raw embeddings of traces are dominated by authorship — they are
fingerprint readers — while being nearly blind to the thing an evaluator
actually cares about. We are building, and quantitatively characterizing, a
processing method that reverses that sensitivity profile, and demonstrating
that the result enables radically cheaper benchmarking.

## What we have done

**Built the corpus and infrastructure.** A format-agnostic loader that renders
any of the leaderboard's trajectory formats to text; official resolved labels
for all 134 Verified submissions harvested from the leaderboard archive;
embedding caches across seven embedding models (one API, six local); and an
evaluation protocol — leave-one-model-out — designed after we measured that
naive evaluation inflates accuracy ~25% by letting systems copy scores from
their model-family siblings.

**Established what does not work, carefully.** A week of representation
engineering — chunking, pooling, rubric sections by embedding similarity,
positional pyramids, distributional distances, trajectory dynamics, path
footprints, supervised reweighting — produced a robust negative: every
unsupervised variant derived from raw embeddings lands at the same
performance, and the extra machinery adds nothing. This negative is itself a
finding (the information ceiling of the raw representation), and it redirected
the project toward changing the *information source* rather than the geometry.

**Found the method: qubric.** For each benchmark instance, an LLM first writes
a *query-specific rubric* — what understanding, localization, reproduction,
editing, verification, and completion specifically mean for this issue
(from the public problem statement; no labels). A judge model then reads each
trace and extracts a short factual description per rubric section; sections
are embedded separately, centered against the cross-system consensus, and
concatenated. The construction is a *similarity-specification device*: the
rubric prompt is a natural-language definition of the equivalence relation
the embedding space should respect. Instance-conditioning is the active
ingredient — the same pipeline with a generic rubric, verdict-style answers,
or a free-form summary is measurably worse, and description quality scales
with judge capability.

**Quantified the sensitivity claim across embedders.** The paper's central
exhibit: across seven embedding models spanning three families and two orders
of magnitude of size, qubric processing produces the same profile shift —
authorship sensitivity (identity, model family, harness) collapses to
approximately chance while task and behavior fidelity are retained or
improved. The pillars themselves were stress-tested into their final form: two
content axes and a three-rung authorship ladder (which agent ⊃ which model ⊃
which harness), each rung separably measurable (including an
identity-beyond-type test on submissions with identical model and harness),
with outcome deliberately *excluded* from the intrinsic battery — outcome is
not trace content; it is the thing you infer. The taxonomy is complete by
construction: a trace is a sample from P(trace | task, agent, seed), and the
pillars are exactly the conditioning variables (content, authorship,
reliability).

**Validated the payoff: query-efficient benchmarking.** Using the perspective
space machinery (DKPS/PKPS) over qubric representations, with true leaderboard
labels and the honest protocol: a new agent's full-benchmark resolve rate can
be predicted within ~0.10 MAE from a *single* probe trace — accuracy the raw
sample score needs eight to ten scored runs to reach. The full pipeline
(representation fusion, product-kernel comparison against references' full
cached coverage, cross-validated hyperparameters, a bias–variance blend with
the sample score, and greedily selected probe instances) reaches ~0.057 MAE at
five probes and ~0.05 at twenty. Every layer is ablated; several "obvious
improvements" (CV inside selection, PCA composed with selection) were shown to
overfit and are documented as negative results.

**Measured the noise honestly.** The reliability analysis separates three
noise sources: agent stochasticity (replicate runs retrieve each other at 6×
chance, but with a within/between distance ratio of 0.81 — a single trace is
noisy before any judge touches it), judge sampling (two judges describing the
same trace agree at trace level only 12–28% — the real cost of semantic
compression), and embedder choice (the profile is stable across all seven).
The redemption result: aggregating ~20 instances drives cross-judge agreement
to 0.82–0.95. Per-trace unreliable, per-agent reliable — and per-agent is what
evaluation uses.

## Potential impact

**Evaluation economics.** Running an agent on a full benchmark costs hundreds
of dollars and hours; the marginal cost of our pipeline per new agent is a few
probe runs, a handful of judge calls, and pennies of embedding. If the results
scale (see caveats), leaderboard-grade estimates from 5 probes instead of 500
changes who can afford systematic evaluation — every fine-tune, every
checkpoint, every config variant becomes benchmarkable.

**Traces as a first-class evaluation object.** The archives already exist.
This work is a template for mining them: the same cached corpus answered
questions about leakage, model fingerprinting, harness artifacts, and
capability prediction without a single new benchmark run. The intrinsic pillar
battery requires *no labels at all* — it can characterize any trace corpus,
including proprietary ones, before anyone decides what to evaluate.

**Programmable similarity.** The deepest idea, if it holds up: embedding
spaces need not inherit their notion of similarity from the embedder. A
natural-language rubric makes the equivalence relation an *input* — change
the prompt, change what "similar" means. Evaluation is our demonstration;
retrieval, deduplication, clustering, and forensics over agent behavior are
the obvious next applications (including the inverse use: raw embeddings as
fingerprint detectors — we showed they identify the exact agent configuration
at twice chance even among same-model, same-harness siblings, which is both a
leakage warning and a provenance-verification tool).

**Methodological hygiene for a new modality.** The protocol findings travel
independently of the method: family leakage inflates naive trace-based
evaluation; intrinsic metrics must be aggregation-aware and lineage-controlled
or they reward memorization; verbosity is a mediator, not a confound; and on
pools of ~50–100 systems, every degree of optimization freedom gets spent on
overfitting — we hit this four separate times.

## Where things stand, honestly

All qubric results rest on a 20-instance panel of SWE-bench Verified with one
judge family; the scale confirmation (150 instances, ~16K judge calls) is
specified and will run on an open-weight judge (local vLLM or OpenRouter) —
we no longer assume access to closed APIs, which also makes the final
pipeline fully reproducible with open models. One benchmark, one task domain
(Terminal-Bench trajectories are public and queued as the second).
Differences below ~0.01 MAE on the panel are near its resolution, and the
identity-beyond-type estimate rests on seven clean systems. The figure set
(sensitivity radars, independence heatmap, QUENCH pair, reliability) is built,
styled, and regenerable from committed data; the schematic, one figure panel,
tables, and the prose remain. Everything — code, caches' recipes, findings
F1–F20, and this narrative — lives on the `agentic-traces` branch.
