# Session Log — Fit-Based IPA Methods (2026-05-22)

This directory and its diagnostic plot were built in a single Claude Code session. This file records what was done, why, and what to know if re-running anything.

## What was built

Four side-by-side methods for computing **fit-based IPA** (Information Processing rate / Asymptote) from the existing power-law fit `CE(x) = A + B/(x+1)^n`. All four consume CSVs already produced by `fitting_function_IPA.ipynb` and `v_4_Combined_experiment_avg_and_plot.ipynb` — no fitting is re-run end-to-end.

```
IPA_methods/
├── Approach_1/
│   ├── compute_ipa_approach_1.ipynb       fit on averaged raw CE; learn_BN from avg data
│   ├── intermediate/                      per-(P%, BS) CSVs including avg_CE_learn_at_BN
│   ├── ipa_summary_approach_1.csv
│   ├── ipa_plot_approach_1.png
│   └── test/
│       └── avg_plot_v_fitting_curve/
│           ├── plot_avg_vs_fit.ipynb      averaged CE scatter vs fitted curve; BNL marker
│           ├── BS_64/                     fitting_avg_plot_A_1_p_{p}_bs_64.png (19 files)
│           ├── BS_1024/                   fitting_avg_plot_A_1_p_{p}_bs_1024.png
│           └── BS_60000/                  fitting_avg_plot_A_1_p_{p}_bs_60000.png
├── Approach_2a1/      compute_ipa_approach_2a1.ipynb  refit per-run-mean fitted curve
├── Approach_2a2/      compute_ipa_approach_2a2.ipynb  average per-run (A,B,n) directly
├── Approach_2b/       compute_ipa_approach_2b.ipynb   per-run IPA (analytical BNL) then average
├── b_physical_meaning/
│   └── plot_b_effect.py                  CONFIG script: raw scatter + fit + B override
├── plot_ipa_summaries.py                 IPA vs P% plots
└── readme_claude.md   (this file)
```

Each `Approach_X/` contains:
- `compute_ipa_approach_X.ipynb` — config + computation + summary build
- `intermediate/` — per-(P%, BS) CSVs with all named variables (A, B, n, CE_o, CE_L, learn_BN, IPA)
- `ipa_summary_approach_X.csv` — final wide schema: `P%, IPA_Avg_64, STD_64, IPA_Avg_1024, STD_1024, IPA_Avg_60000, STD_60000`
- `ipa_plot_approach_X.png` — IPA vs P% (log Y), 3 batch sizes

## IPA definition

Kept in the **physically meaningful form**, not the algebraically simplified one:

```
CE_o      = ln(10) ≈ 2.302585
CE_L      = CE_o - 0.9 * (CE_o - A)        # learning threshold
IPA       = abs(CE_o - CE_L) / learn_BN     # NOT 0.9*(CE_o - A)/learn_BN
```

`CE_L` and `CE_o` appear as named variables in every print, every intermediate CSV column, and every notebook. The two forms are algebraically equivalent, but keeping `CE_L` explicit preserves the physics in the code.

## learn_BN: per-approach strategy

Each approach has a different data source and therefore a different strategy for finding `learn_BN`. All share the same analytical fallback formula when the primary source ends before CE_L is crossed.

### Approach 1 — raw averaged CE data (primary), analytical fallback

`compute_ipa_from_fit(x_grid, ce_data, A, B, n)`:

1. Compare the **actual averaged CE data** (`Avg_CE_Test`) directly against `CE_L`. `mask = ce_data <= CE_L`.
2. If any True: `learn_BN = first BN in data where Avg_CE_Test ≤ CE_L`. `avg_CE_learn_at_BN` = that data value.
3. If none True (data ends before crossing): fall back to analytical: `learn_BN = ceil((B / (CE_L - A))^(1/n) - 1)`. `avg_CE_learn_at_BN = NaN` (BN is beyond recorded data).

This means `learn_BN` is grounded in the observed averaged data for most pruning levels, and only extrapolated from the fit when the training data runs out early. The intermediate CSVs include `avg_CE_learn_at_BN` to make this distinction traceable.

Previously, Approach 1 compared the **fitted curve values** (not the raw data) against CE_L on the data grid. The change to raw data makes `learn_BN` physically observable rather than model-derived in the normal case.

**NaN fix:** P%=90, 92, 94, 96 at BS=64 (and P%=92 at BS=1024; P%=90, 92 at BS=60000) previously returned NaN because the averaged data never crossed CE_L and there was no fallback. These now produce finite `learn_BN` and `IPA` via the analytical formula, tagged `[analytic]` in the notebook output.

### Approach 2b — always analytical

`compute_ipa_from_fit(A, B, n)` (no data grid argument):

`learn_BN = ceil((B / (CE_L - A))^(1/n) - 1)` directly from fitted parameters. No grid search at all.

Previously 2b searched the raw per-run BN grid to find the first crossing of the fitted curve, which snapped `learn_BN` to the noisy observed BN values. Using the analytical inverse of the fit makes `learn_BN` consistent with the fitted model.

### Approaches 2a1 and 2a2 — fitted curve on data grid (unchanged)

Evaluate `fitted = A + B/(x_grid+1)^n` on the source BN array (`mean_fit_*.csv`). `mask = fitted ≤ CE_L`. Analytical fallback if no crossing found. These were not changed.

## STD policy

Per-method STD columns in `ipa_summary_*.csv`:

| Method | STD source | Result |
|---|---|---|
| 1, 2a1, 2a2 | Single curve, no variability to estimate | NaN (blank) |
| 2b | Std across 100 per-run IPAs | Populated |

This was an explicit choice — fit uncertainty was not propagated for the single-curve methods.

## Verification (matches the plan's checks)

- **Approach 1 vs v_4 reference** (p=0.5, bs=1024): A=0.25397, B=6.8181, n=0.8783 — matches v_4's reported A=0.2540, B=6.8181, n=0.8783 to 4 decimals.
- **CE_L formula** spot-check: at A=0.306 (p=80% row), CE_L=0.506 — consistent with `CE_o - 0.9*(CE_o - A)`.
- **Approach 2a2 vs 2a1** at low pruning (p=0%, bs=1024): 0.066 vs 0.063 — agree within ~3% as expected.
- **Approach 2b sanity** (p=0.5, bs=1024): IPA_mean=0.058 ± 0.114, same order of magnitude as Approach 1's 0.034 and existing raw-data IPA.

## Input paths (canonical)

- Raw averaged data: `prune_layers_ALL/p-percentage_{p}/batch_size_{bs}/averaged_runs_p_{p}_bs_{bs}.csv`
- Fitted curves: `Fitting_IPA_curves_data_I/BS_{bs}/{mean_fit,per_run_fits}_p_{p}_bs_{bs}.csv`
- Fit-parameter summary: `Fitting_IPA_curves_data_I/Summary/per_run_fit_parameters_bs_{bs}.csv`

The `_I` suffix matters — `fitting_function_IPA.ipynb` still references the old (stale) `Fitting_IPA_curves_data/` path internally. None of the IPA-method notebooks here are affected because they read from `_I` directly.

## Re-running

Each `compute_ipa_approach_X.ipynb` is independently runnable (no shared kernel state). Use the DataS conda env:

```
"C:\Users\Student\miniconda3\envs\DataS\python.exe" -m jupyter nbconvert --to notebook --execute --inplace Approach_X/compute_ipa_approach_X.ipynb
```

Then refresh plots:

```
"C:\Users\Student\miniconda3\envs\DataS\python.exe" plot_ipa_summaries.py
```

The diagnostic plot is one-shot — `python test/plot_why_nan_p_0.9.py` to regenerate.

## Decisions made during the session

| Question | Choice |
|---|---|
| STD for single-curve methods | NaN (blank) |
| learn_BN search — Approach 1 | Raw averaged CE data first; analytical fallback when data ends early |
| learn_BN search — Approach 2b | Always analytical (`ceil((B/(CE_L-A))^(1/n) - 1)`); no grid search |
| learn_BN search — 2a1/2a2 | Discrete on fitted-curve values on data grid; analytical fallback |
| Code form | One `.ipynb` per approach |
| IPA formula form | Keep `abs(CE_o - CE_L) / learn_BN` literal (not simplified) |
| Approach 1 intermediate CSV | Added `avg_CE_learn_at_BN` column (NaN when fallback used) |
| High-pruning NaN resolution | Analytical fallback in Approach 1 now fills P%=90–96 rows |

## Out of scope (still pending)

- Reconciling the stale `Fitting_IPA_curves_data/` (no `_I`) path inside `fitting_function_IPA.ipynb`.
- Any change to the upstream fitting itself.

## Resolved issues

- **Approach 1 NaN at high pruning**: P%=90–96 at BS=64 (and a few at BS=1024/60000) returned NaN because the averaged CE data ended before crossing CE_L (early stopping). Fixed by adding analytical fallback to `compute_ipa_from_fit`. Previously there was no fallback at all in Approach 1.
- **Approach 2b grid-snapping**: `learn_BN` was being read off the raw per-run BN grid, making it noisy. Changed to always use the analytical inverse of the fitted curve.
