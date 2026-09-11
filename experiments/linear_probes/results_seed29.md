# Linear-probe results

Held-out balanced accuracy (%); brackets are 95% group-bootstrap intervals.
Higher means the attribute is more linearly recoverable, not necessarily better.

| Representation | Target | N | Classes | Balanced accuracy [95% CI] | Chance |
|---|---|---:|---:|---:|---:|
| openai_generic | system | 2129 | 107 | 55.6 [53.7, 57.4] | 0.9 |
| openai_generic | model | 1191 | 36 | 57.3 [54.3, 60.5] | 2.8 |
| openai_generic | vendor | 1591 | 8 | 57.7 [54.5, 61.1] | 12.5 |
| openai_generic | harness | 2069 | 61 | 68.4 [66.7, 70.0] | 1.6 |
| openai_generic | task | 2129 | 20 | 99.5 [99.0, 99.9] | 5.0 |
| openai_generic | outcome | 2129 | 2 | 82.4 [80.3, 84.6] | 50.0 |
| openai_qubric | system | 2129 | 107 | 14.4 [12.7, 15.9] | 0.9 |
| openai_qubric | model | 1191 | 36 | 24.8 [21.6, 27.9] | 2.8 |
| openai_qubric | vendor | 1591 | 8 | 30.2 [27.2, 33.4] | 12.5 |
| openai_qubric | harness | 2069 | 61 | 20.8 [18.8, 22.9] | 1.6 |
| openai_qubric | task | 2129 | 20 | 100.0 [100.0, 100.0] | 5.0 |
| openai_qubric | outcome | 2129 | 2 | 82.2 [80.4, 84.2] | 50.0 |
| openai_raw | system | 2129 | 107 | 90.1 [89.0, 91.2] | 0.9 |
| openai_raw | model | 1191 | 36 | 88.1 [86.4, 89.9] | 2.8 |
| openai_raw | vendor | 1591 | 8 | 96.8 [95.7, 97.8] | 12.5 |
| openai_raw | harness | 2069 | 61 | 95.6 [94.7, 96.5] | 1.6 |
| openai_raw | task | 2129 | 20 | 100.0 [99.8, 100.0] | 5.0 |
| openai_raw | outcome | 2129 | 2 | 81.9 [80.0, 83.9] | 50.0 |
| nomic_generic | system | 2129 | 107 | 49.9 [47.4, 52.2] | 0.9 |
| nomic_generic | model | 1191 | 36 | 51.0 [48.6, 53.5] | 2.8 |
| nomic_generic | vendor | 1591 | 8 | 53.6 [50.1, 57.2] | 12.5 |
| nomic_generic | harness | 2069 | 61 | 64.0 [61.7, 66.1] | 1.6 |
| nomic_generic | task | 2129 | 20 | 99.5 [99.1, 99.9] | 5.0 |
| nomic_generic | outcome | 2129 | 2 | 82.5 [80.6, 84.7] | 50.0 |
| nomic_qubric | system | 2129 | 107 | 10.4 [9.1, 11.9] | 0.9 |
| nomic_qubric | model | 1191 | 36 | 16.0 [13.6, 18.4] | 2.8 |
| nomic_qubric | vendor | 1591 | 8 | 30.9 [27.3, 35.0] | 12.5 |
| nomic_qubric | harness | 2069 | 61 | 14.8 [13.2, 16.5] | 1.6 |
| nomic_qubric | task | 2129 | 20 | 100.0 [100.0, 100.0] | 5.0 |
| nomic_qubric | outcome | 2129 | 2 | 81.6 [79.6, 83.9] | 50.0 |
| nomic_raw | system | 2129 | 107 | 83.1 [81.8, 84.5] | 0.9 |
| nomic_raw | model | 1191 | 36 | 81.1 [78.5, 83.6] | 2.8 |
| nomic_raw | vendor | 1591 | 8 | 90.8 [89.0, 92.4] | 12.5 |
| nomic_raw | harness | 2069 | 61 | 91.4 [90.1, 92.6] | 1.6 |
| nomic_raw | task | 2129 | 20 | 96.9 [95.3, 98.2] | 5.0 |
| nomic_raw | outcome | 2129 | 2 | 77.9 [75.5, 80.1] | 50.0 |

See README.md for the classifier, splits, preprocessing, and interpretation limits.
