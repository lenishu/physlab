# approach_1_avoid_step_weighted — notes

Experimental variant of `approach_1_avoid_step` (which is the working version). Adds:

1. **Per-run tail fit quality** — `R2_last50` and `RMSE_last50` (unweighted, on the last
   `TAIL_N=50` points of the fit region) saved into each `intermediate/approach_1_fit_params_bs_{bs}.csv`.
2. **Weight-power sweep** — residual weight is now `x**power`. Fit at `power=1.0` first; if its
   last-50 **RMSE** > `RMSE_TAIL_THRESH` (0.01), re-fit with the other `WEIGHT_POWERS`
   (`[1.0, 1.5, 2.0]`) and keep the lowest-RMSE fit. Chosen power saved as `weight_power`.

## Key finding: use RMSE, not R², on these tails

The averaged-CE tail is nearly flat, so its variance is tiny and **R² is dominated by noise**
(negative even for visually-perfect fits — e.g. P=0% has RMSE≈0.0037 but R²≈0.30; P=88% before
sweep had RMSE 0.036 and R² ≈ −448). **RMSE cleanly separates good (~0.004) from bad (0.03–0.05)
fits.** So the sweep is RMSE-driven; R² is kept in the CSV for reference only.

## Behaviour vs original
Runs that already fit well stay at `power=1.0` and reproduce the original IPA exactly
(verified P=0, P=10 at BS=64). Only the genuinely poor-RMSE high-pruning runs (P≈88/94/96/98)
re-fit at a higher power and change.

Config knobs are at the top of Cell 1: `WEIGHT_POWERS`, `TAIL_N`, `RMSE_TAIL_THRESH`.
Outputs (CSVs, summary, plots) are co-located under this folder, unlike the original which
writes to a sibling `test/` dir.
