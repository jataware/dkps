# Linear recovery of trace attributes

Date: 2026-09-10. This is a new supervised experiment, not a redraw of the
nearest-neighbor radar metrics. The existing radar plots and LaTeX report have
not been changed.

## Main finding

**Both rubric constructions reduce linearly accessible identity information;
qubric reduces it much more. Neither makes identity unrecoverable.** Task
identity remains highly accessible, and correctness prediction is retained.
This supports a claim of selective attenuation of identity-related signals,
not complete authorship invariance or a demonstrated causal explanation for
QUENCH's performance.

For example, exact-system balanced accuracy falls from **89.5% → 56.0% →
14.4%** for OpenAI raw → common rubric → qubric, and from **83.3% → 49.8% →
9.3%** for Nomic. The chance baseline is only **0.93%**. Thus even qubric leaves
substantial recoverable system information. Near-chance nearest-neighbor
retrieval would not justify claiming that identity information is absent.

## Results

Entries are held-out **balanced accuracy (%)**, averaged over target classes.
Higher means more recoverable, not universally more desirable. The two
embedders are separate matched comparisons; their raw-text truncation differs.

### OpenAI text-embedding-3-small

| Predicted attribute | Classes | Chance | Raw head + tail | Common rubric | Qubric |
|---|---:|---:|---:|---:|---:|
| Exact system configuration | 107 | 0.93 | 89.5 | 56.0 | 14.4 |
| Reported model | 36 | 2.78 | 88.0 | 55.8 | 21.9 |
| Model vendor / coarse lineage | 8 | 12.50 | 97.7 | 61.1 | 30.8 |
| Reported harness / agent | 61 | 1.64 | 95.9 | 70.9 | 19.9 |
| Task identity | 20 | 5.00 | 99.8 | 99.5 | 100.0 |
| Correctness | 2 | 50.00 | 81.9 | 81.8 | 82.1 |

### Nomic embed-text-v1.5

| Predicted attribute | Classes | Chance | Raw head + tail | Common rubric | Qubric |
|---|---:|---:|---:|---:|---:|
| Exact system configuration | 107 | 0.93 | 83.3 | 49.8 | 9.3 |
| Reported model | 36 | 2.78 | 81.1 | 50.0 | 16.6 |
| Model vendor / coarse lineage | 8 | 12.50 | 90.7 | 55.2 | 28.1 |
| Reported harness / agent | 61 | 1.64 | 91.8 | 63.9 | 14.6 |
| Task identity | 20 | 5.00 | 96.7 | 99.6 | 100.0 |
| Correctness | 2 | 50.00 | 78.6 | 82.8 | 82.4 |

The complete [results table](results.md) includes 95% group-bootstrap
intervals. [CSV](results.csv) and [JSON](results.json) include additional
metrics and auditing details. For exact-system recovery, the intervals are:

- OpenAI: raw 88.3–90.7%, common 54.2–57.7%, qubric 12.9–15.8%.
- Nomic: raw 81.6–85.0%, common 47.4–52.2%, qubric 8.0–10.6%.

OpenAI correctness is essentially unchanged across constructions. Nomic
correctness is approximately four percentage points higher with either
rubric. Qubric does **not** consistently outperform the common rubric on this
binary outcome probe; its clearest distinction here is lower identity
recoverability. This experiment does not measure QUENCH resolve-rate MAE.

## Protocol and coverage

The panel contains 107 system configurations and 20 tasks. Eleven traces with
missing, empty, or unparseable common/qubric extractions are excluded from every
representation, leaving **2,129 matched traces**. Metadata availability reduces
the model target to 1,191 traces, vendor to 1,591, and harness to 2,069. Unknown
or mixed models are not treated as meaningful model classes. The label CSV
and exclusion list are included with the handoff.

The readout is a class-balanced linear ridge classifier with an unpenalized
intercept. Six penalties are selected in three grouped inner folds, inside
five grouped outer folds (seed 17). The global ridge solution is computed by
eigendecomposition, not an iterative optimizer. All normalization is
sample-local and all fitted quantities use training rows only.

- System/model/vendor/harness: hold out entire **tasks**. The classifier must
  recognize identity-related attributes on tasks absent from its training set.
- Task/correctness: hold out **reported model groups**, grouping configurations
  with the same disclosed model tag. Unknown/mixed configurations are singleton
  groups. This does not guarantee exclusion of shared underlying parameters
  when metadata is ambiguous or aliases differ.
- Confidence intervals resample held-out groups, not individual trace rows.
  They are conditional on the fitted out-of-fold models and do not include
  full retraining uncertainty.

The portable [README](README.md) specifies the objective, preprocessing,
metadata interpretation, and full reproduction commands.

## What this does and does not establish

1. The broad pattern is consistent with rubric extraction suppressing
   identity-related signals while retaining task/outcome information. It does
   not prove that the suppressed signals are all incidental or that their
   suppression causes downstream QUENCH gains.
2. Model and harness are correlated with each other and with exact system
   configuration. Because the identity probes hold out tasks, not systems,
   recognition of a known configuration can support model/harness prediction.
   This is legitimate evidence of recoverability, but not disentangled effects.
3. There is no direct “Behavior” column: correctness is not an annotation of
   evidence gathering, localization, editing, or verification strategy. Such
   categorical annotations can be added through the manifest without changing
   the classifier code.
4. These probes operate on cached embeddings **before per-task consensus
   centering**. They test the raw/common/qubric construction, not the entire
   downstream QUENCH geometry pipeline. Normalization is section-wise L2 and
   equal-weight concatenation for all constructions.
5. Task identity can be apparent from task-specific rubric language. Its
   perfect recovery is not evidence, by itself, that behavioral information
   is preserved. A 100–100% bootstrap interval merely means no errors were
   observed in this finite panel.
6. Linear recovery provides a lower bound on what a richer supervised probe
   might recover; low accuracy does not establish information-theoretic
   invariance. Different embedding dimensions and finite-sample probe
   learnability also affect the comparison.

## Validation

Eight unit tests pass: direct primal/dual agreement, grouped partitioning,
sample-local normalization, recovery of a known linear signal, test-feature
independence, rejection of missing training classes, balanced-accuracy
behavior, and explicit embedding-ID joins.

An additional synthetic outer-label perturbation check leaves the affected
outer fold's predictions, selected penalty, and inner validation scores
unchanged. A random-feature synthetic control obtains 32.5% balanced accuracy
against a 33.3% chance baseline.

All 36 primary representation/target pairs and all 36 second-seed pairs
completed. Their source hashes, row coverage, and disjoint inner/outer group
assignments were checked against the saved artifacts. Restarting the completed
primary command correctly resumes without rerunning completed pairs.

### Sensitivity checks

Changing the partition seed from 17 to 29 preserves **raw > common > qubric**
identity recoverability for every identity target under both embedders. The
largest change among all 36 measurements is 3.36 percentage points. Selected
examples (seed 17 → seed 29):

| Measurement | OpenAI | Nomic |
|---|---:|---:|
| Raw system recovery | 89.5 → 90.1 | 83.3 → 83.1 |
| Common-rubric system recovery | 56.0 → 55.6 | 49.8 → 49.9 |
| Qubric system recovery | 14.4 → 14.4 | 9.3 → 10.4 |
| Raw correctness | 81.9 → 81.9 | 78.6 → 77.9 |
| Common-rubric correctness | 81.8 → 82.4 | 82.8 → 82.5 |
| Qubric correctness | 82.1 → 82.2 | 82.4 → 81.6 |

Some raw identity fits choose the weakest penalty in the main grid. A targeted
check extends the grid from `1e-5` down to `1e-7` for exact-system recovery in
both raw representations: all held-out predictions remain unchanged. This is
a boundary check for that target, not an exhaustive optimizer comparison.
Qubric task identification is perfect across the grid; its selection of the
largest penalty reflects the tie-breaking rule, not a failed optimization.

See [second-seed results](results_seed29.md) and
[regularization sensitivity](results_ridge_sensitivity.md) for the separate
artifacts. Run the checks sequentially: concurrent dense linear algebra was
substantially slower on this machine despite the requested thread limits.
