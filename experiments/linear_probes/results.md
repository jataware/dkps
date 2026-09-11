# Linear-probe results

Held-out balanced accuracy (%); brackets are 95% group-bootstrap intervals.
Higher means the attribute is more linearly recoverable, not necessarily better.

| Representation | Target | N | Classes | Balanced accuracy [95% CI] | Chance |
|---|---|---:|---:|---:|---:|
| openai_generic | system | 2129 | 107 | 56.0 [54.2, 57.7] | 0.9 |
| openai_generic | model | 1191 | 36 | 55.8 [52.9, 58.8] | 2.8 |
| openai_generic | vendor | 1591 | 8 | 61.1 [57.4, 65.2] | 12.5 |
| openai_generic | harness | 2069 | 61 | 70.9 [69.5, 72.3] | 1.6 |
| openai_generic | task | 2129 | 20 | 99.5 [99.1, 100.0] | 5.0 |
| openai_generic | outcome | 2129 | 2 | 81.8 [79.8, 84.0] | 50.0 |
| openai_qubric | system | 2129 | 107 | 14.4 [12.9, 15.8] | 0.9 |
| openai_qubric | model | 1191 | 36 | 21.9 [19.5, 24.8] | 2.8 |
| openai_qubric | vendor | 1591 | 8 | 30.8 [27.8, 33.9] | 12.5 |
| openai_qubric | harness | 2069 | 61 | 19.9 [18.2, 22.0] | 1.6 |
| openai_qubric | task | 2129 | 20 | 100.0 [100.0, 100.0] | 5.0 |
| openai_qubric | outcome | 2129 | 2 | 82.1 [80.2, 84.1] | 50.0 |
| openai_raw | system | 2129 | 107 | 89.5 [88.3, 90.7] | 0.9 |
| openai_raw | model | 1191 | 36 | 88.0 [86.7, 89.2] | 2.8 |
| openai_raw | vendor | 1591 | 8 | 97.7 [96.9, 98.3] | 12.5 |
| openai_raw | harness | 2069 | 61 | 95.9 [95.0, 96.9] | 1.6 |
| openai_raw | task | 2129 | 20 | 99.8 [99.5, 100.0] | 5.0 |
| openai_raw | outcome | 2129 | 2 | 81.9 [79.9, 84.0] | 50.0 |
| nomic_generic | system | 2129 | 107 | 49.8 [47.4, 52.2] | 0.9 |
| nomic_generic | model | 1191 | 36 | 50.0 [47.6, 52.4] | 2.8 |
| nomic_generic | vendor | 1591 | 8 | 55.2 [51.6, 58.8] | 12.5 |
| nomic_generic | harness | 2069 | 61 | 63.9 [61.5, 66.0] | 1.6 |
| nomic_generic | task | 2129 | 20 | 99.6 [99.2, 100.0] | 5.0 |
| nomic_generic | outcome | 2129 | 2 | 82.8 [80.7, 85.0] | 50.0 |
| nomic_qubric | system | 2129 | 107 | 9.3 [8.0, 10.6] | 0.9 |
| nomic_qubric | model | 1191 | 36 | 16.6 [14.4, 18.8] | 2.8 |
| nomic_qubric | vendor | 1591 | 8 | 28.1 [24.3, 32.0] | 12.5 |
| nomic_qubric | harness | 2069 | 61 | 14.5 [13.0, 16.2] | 1.6 |
| nomic_qubric | task | 2129 | 20 | 100.0 [100.0, 100.0] | 5.0 |
| nomic_qubric | outcome | 2129 | 2 | 82.4 [80.5, 84.5] | 50.0 |
| nomic_raw | system | 2129 | 107 | 83.3 [81.6, 85.0] | 0.9 |
| nomic_raw | model | 1191 | 36 | 81.1 [79.2, 83.2] | 2.8 |
| nomic_raw | vendor | 1591 | 8 | 90.8 [88.7, 92.7] | 12.5 |
| nomic_raw | harness | 2069 | 61 | 91.8 [90.2, 93.4] | 1.6 |
| nomic_raw | task | 2129 | 20 | 96.7 [94.9, 98.0] | 5.0 |
| nomic_raw | outcome | 2129 | 2 | 78.6 [76.4, 81.0] | 50.0 |

See README.md for the classifier, splits, preprocessing, and interpretation limits.
