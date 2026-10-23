# MatPES benchmark

The paper's out-of-distribution force test: MatPES PBE 2025.2, a separate DFT
dataset sampled from 300 K MD (DIRECT sampling), never seen in training.

## Test set

| Step | Command | Check |
|---|---|---|
| Download the release (2.36 GB JSONL) | `bash scripts/data/download_matpes.sh` | SHA-256 from Hugging Face |
| Take the authors' sids | `envtok-mace matpes --dump-sids <authors>/EnvTok/per_structure.npz --out sids.txt` | |
| Convert | `envtok-mace matpes-build --src MatPES-PBE-2025.2.jsonl --sids sids.txt --expect --dst f.h5` | 323,096 structures, 3,011,678 atoms |
| Neighbour lists | `envtok-mace preprocess --src f.h5 --dst matpes_test.h5` | r_max 6 Å |

The release has 433,189 structures. Dropping every structure that contains an
element outside the model's table ({1..56} ∪ {72..83}) gives the paper's 323,096.
The authors' published sid list is used as well, so the set and its order are
exactly theirs. Atom order is checked against their per-atom file, atom by atom,
by `--compare`.

## Units and conventions

Energies are in eV and forces in eV/Å. The release does not document its
stress, so it was measured. The authors' EnvTok model was scored against all
four readings of the 6-vector, on 121 structures from the release:

| Reading of the 6-vector | Stress MAE (meV/Å³) |
|---|---|
| **kbar, sign flipped, Voigt order xx yy zz yz xz xy** | **6.4** |
| kbar, sign flipped, VASP order XX YY ZZ XY YZ ZX | 20.0 |
| kbar, sign kept, VASP order | 44.3 |
| kbar, sign kept, Voigt order | 51.9 |

The first reading is the one used. On the full bulk domain the authors report a
pooled stress MAE of 7.4 meV/Å³ for the same model, which agrees with it.
`envtok-mace matpes --calibrate-stress` repeats the measurement.

## Quantities and estimators

Per structure, as in the authors' `per_structure.npz`:

- `f_mae_comp`: mean of |ΔF| over the 3N components. This is MACE's MAE_F.
- `f_rmse_comp`: RMSE over the same components.
- `f_max_norm`: the largest per-atom ‖ΔF_i‖.
- `abs_d_energy_per_atom`: |ΔE| / N.
- `s_mae_voigt6`: mean of |Δσ| over the six Voigt components.
- `s_err_norm`: Frobenius norm of the 3×3 stress error.

Force quantities are pooled with N as the weight, so the pooled mean over a
domain is the mean over its atoms.

On a domain, every model is summarised by the estimators below. They are the
authors' definitions, reimplemented:

| Estimator | Definition |
|---|---|
| median, q75, q90 | weighted quantiles |
| winsor_mean | weighted mean after clipping at one cap for every model: the 0.99 quantile of all models' values pooled |
| trimmed_mean* | weighted mean after dropping the worst 5 % of the weight; not comparable across models |
| geometric_mean | weighted, floor 1e-9 |
| pooled_mean_MAE | weighted mean |

Robust estimators are needed because the non-bulk domain contains structures
where a subset model's force error runs to thousands of meV/Å. The authors'
EnvTok model has a non-bulk pooled force MAE of 3,904 meV/Å, against a median
of about 0.1 eV/Å. A plain mean there is a statement about a handful of
structures.

## Domains

The bulk / non_bulk / overcompressed labels come from the `domain` column of the
authors' published per-structure file and are not recomputed. Their labels depend
only on the reference geometry (dimensionality and fragment detection, and a
close-contact screen), never on a model's errors. Reusing them makes the bins
identical to the paper's.

## Validation before scoring our model

1. Score the authors' EnvTok model with this pipeline. Then check it atom by
   atom against their per-atom file:
   `envtok-mace matpes --compare <ours>/authors_envtok_rerun <authors>/EnvTok`.
2. Reproduce their bulk numbers for that model:
   - pooled force MAE 148.60 meV/Å (component convention);
   - median per-structure force MAE 112.6 meV/Å.

The reference numbers for the other models are in the paper tables. Bulk pooled
force MAE: MACE-OMAT-0 126.40 meV/Å, DIRECT 151.20 meV/Å, random seeds
149.72–151.68 meV/Å.

The whole run is `scripts/nscc/matpes.pbs`.
