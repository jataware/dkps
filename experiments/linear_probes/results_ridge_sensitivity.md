# Linear-probe results

Held-out balanced accuracy (%); brackets are 95% group-bootstrap intervals.
Higher means the attribute is more linearly recoverable, not necessarily better.

| Representation | Target | N | Classes | Balanced accuracy [95% CI] | Chance |
|---|---|---:|---:|---:|---:|
| openai_raw | system | 2129 | 107 | 89.5 [88.3, 90.7] | 0.9 |
| nomic_raw | system | 2129 | 107 | 83.3 [81.6, 85.0] | 0.9 |

See README.md for the classifier, splits, preprocessing, and interpretation limits.
