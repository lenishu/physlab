# approach_1_avoid_step_floor — notes

Floor-based-asymptote variant of `approach_1_avoid_step`. Fixes the IPA-vs-P% **bumps**
(e.g. BS=64 at P=82%/88%, BS=60000 at P=92%) that came from inconsistent asymptote estimation.

## What it does differently

- **A = empirical tail floor** = mean of the last `TAIL_N=50` points of the fit window
  (the curve's actual plateau), instead of the power-law BN→∞ limit.
- `A` is **pinned** (`vary=False`); only `B, n` are fitted (BN-weighted, `weight = x`) so the
  overlaid curve hugs the tail and matches the floor line.
- `CE_L = CE_o - 0.85*(CE_o - A)` and IPA are computed from this floor.

## Why (root cause of the bumps)

The model `A + B/(BN+1)^n` defines A as the BN→∞ limit, but the power-law tail decays slower than
the real saturating curve, so the leftover decay term at the last batch is exactly the gap:
`floor ≈ A + B/(BNmax+1)^n` (verified on all 57 runs). That gap grows with pruning and its *local*
variation (driven by how steep an `n` the fit happens to pick, which depends on the step-detection
window) created the bumps. Pinning A to the floor removes the gap entirely.

## Results

- IPA is **monotonically decreasing in all three batch sizes** — every bump gone.
  (BS=64 P=88% = 0.01296, cleanly between P=86%=0.01442 and P=90%=0.01106.)
- High-pruning runs (P≥92%) now cross `CE_L` in **real data** — the analytic-fallback cases are gone.

## Comparing against avoid-step

`compare_ipa_avoid_step_vs_floor.ipynb` overlays this method vs the working `approach_1_avoid_step`.
It **reads each method's saved `ipa_summary_*.csv` from disk** (does not recompute), so after you
change/re-run the avoid-step notebook, just re-run this comparison and it reflects the new numbers.
Paths are config vars at the top of its Cell 1. Output: `ipa_compare_avoid_step_vs_floor.png`.

(`ipa_compare_floor_vs_powerlaw.png` was a one-off scratch figure that recomputed the old fit
inline — superseded by the notebook above; ignore/delete it.)

## Fit-quality reporting

`RMSE_last50` is the meaningful metric (the floor approach targets the tail; the curve hugs it,
RMSE_last50 ~0.003–0.05). `RMSE_full` is large and only-for-reference: a single power law can't fit
both the steep early drop and the flat tail when A is pinned low, and the BN-weighting de-emphasizes
the early region by design. IPA does not depend on B,n (only on A=floor + the data crossing), so the
B,n curve is illustrative only.

Config knobs at top of Cell 1: `TAIL_N`, `STEP_THRESH`, `BN_STEP_MIN`.
