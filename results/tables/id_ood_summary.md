| Category | Test | Metric | Ours | Authors EnvTok | Authors DIRECT | Random (seeds) | MACE-OMAT-0 |
|---|---|---|---|---|---|---|---|
| ID | OMat24 validation (591,981 frames) | energy MAE (meV/atom, lower) | 24.6 | 25.5 | 28.7 | 26.7 | 17.5 |
| ID | OMat24 validation (591,981 frames) | force MAE (meV/A, lower) | 109 | 112 | 119 | 117 | 84.5 |
| ID | OMat24 validation (591,981 frames) | stress MAE (meV/A^3, lower) | 2.88 | 2.95 | 3.13 | 3.08 | 2.58 |
| shift | MatPES bulk (242,319) | force MAE per structure, median (meV/A, lower) | 110 | 113 | 119 | 115 | 83.6 |
| shift | MatPES bulk (242,319) | force MAE, pooled mean (meV/A, lower) | 143 | 149 | 151 | 151 | 126 |
| OOD | MatPES non-bulk (80,766) | force MAE per structure, median (meV/A, lower) | 240 | 252 | 229 | 249 | 159 |
| OOD | MatPES non-bulk (80,766) | force MAE, pooled mean (meV/A, lower) | 321 | 3.9e+03 | 327 | 3.76e+04 | 275 |
| shift | MatCalc WBM (474) | formation energy MAE (meV/atom, lower) | 62.9 | 62.7 | 74.3 | 65.5 | 56 |
| shift | MatCalc MP binaries (3,127) | bulk modulus, median signed error (%, closer to 0) | 1.41 | 1.47 | 2.01 | 2.73 | 3.17 |
| shift | MatCalc MP binaries (3,127) | shear modulus, median signed error (%, closer to 0) | -4.88 | -5.77 | -8.03 | -5.62 | 1.42 |
| shift | MatCalc Alexandria (1,050) | heat capacity, median abs error (%, lower) | 0.731 | 0.757 | 0.825 | 0.682 | 0.362 |
| OOD | MatCalc WBM high-energy states (480) | force softening, median (%, lower) | 3.63 | 3.77 | 3.9 | 3.32 | 2.4 |
| OOD | MD, NPT 300-3000 K, 0-500 GPa (80 runs) | mean completion (%, higher) | 96.1 | 92.1 | 92.9 | 81.1 | 75.4 |
| OOD | MD, NPT 300-3000 K, 0-500 GPa (80 runs) | physically diverged runs (% of runs, lower) | 0 | 7.5 | 11.2 | 26.6 | 38.8 |
| OOD | nebDFT2k Li-ion migration (1,656 hops) | barrier MAE, hosts without Hubbard U (meV, lower) | 138 | 124 | 116 | 148 | 146 |
| OOD | nebDFT2k Li-ion migration (1,656 hops) | barrier MAE, hosts with Hubbard U (meV, lower) | 259 | 269 | 316 | 264 | 270 |
