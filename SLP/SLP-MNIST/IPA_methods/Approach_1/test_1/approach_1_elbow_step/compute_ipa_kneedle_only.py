"""
Minimal Kneedle-only IPA computation.

IPA = |CE_o - CE_learned| / BN_learned, where (BN_learned, CE_learned) is the
elbow of the averaged CE-vs-BN curve found by the Kneedle algorithm
(Satopaa et al. 2011, `kneed` package).

Kneedle operates directly on the raw averaged data, so this script deliberately
OMITS everything the elbow does not need:
  - no curve fitting (lmfit A + B/(BN+1)^n)      — diagnostics only
  - no empirical floor A                          — Kneedle normalizes by data min/max
  - no plotting

What IS kept (required for correctness):
  - step detection: truncate at the first BN >= BN_STEP_MIN where
    |CE[i]-CE[i-1]| > STEP_THRESH (averaging artifact from early-stopped runs)
  - degenerate-curve guard (flat curve, e.g. P=100% -> IPA = NaN)

Usage:  pip install kneed pandas numpy
        python compute_ipa_kneedle_only.py

Outputs (in OUT_DIR):
  intermediate/kneedle_only_bs_{bs}.csv   per-BS elbow + IPA per pruning level
  ipa_summary_kneedle_only.csv            same column layout as the other summaries
"""

import os, glob, re
import numpy as np
import pandas as pd
from kneed import KneeLocator

# ── CONFIG (shared conventions with the other methods) ───────────────────────
BN_STEP_MIN = 100    # don't check for steps before this BN
STEP_THRESH = 0.01   # |CE[i]-CE[i-1]| threshold for step artifact
KNEEDLE_S   = 1.0    # Kneedle sensitivity (larger = more conservative knee)

BASE_DIR = r"C:\Users\Student\Desktop\Projects\research\physlab\SLP\SLP-MNIST\prune_layers_ALL"
OUT_DIR  = r"C:\Users\Student\Desktop\Projects\research\physlab\SLP\SLP-MNIST\IPA_methods\Approach_1\test_1\approach_1_elbow_step"
INTERMEDIATE_DIR = os.path.join(OUT_DIR, "intermediate")

BATCH_SIZES = [64, 1024, 60000]
CE_o = np.log(10)   # max CE for 10-class problem, ~2.302585
# ─────────────────────────────────────────────────────────────────────────────


def detect_step_cutoff(bns, ces):
    """First BN >= BN_STEP_MIN where the averaged CE jumps by > STEP_THRESH."""
    for i in range(1, len(bns)):
        if bns[i] >= BN_STEP_MIN and abs(ces[i] - ces[i - 1]) > STEP_THRESH:
            return float(bns[i]), True
    return float(bns[-1]), False


def compute_ipa_kneedle(BN_data, CE_data, S=KNEEDLE_S):
    """
    Kneedle elbow -> IPA. CE vs BN is convex and decreasing.
    Returns dict with BN_learned, CE_learned, IPA (NaN if degenerate/no knee).
    """
    BN_data = np.asarray(BN_data, float)
    CE_data = np.asarray(CE_data, float)
    nan = {"BN_learned": np.nan, "CE_learned": np.nan, "IPA": np.nan}

    if len(BN_data) < 3 or np.ptp(CE_data) <= 1e-10:   # flat curve, e.g. P=100%
        return nan
    try:
        kl = KneeLocator(BN_data, CE_data, curve="convex",
                         direction="decreasing", S=S)
    except Exception:
        return nan
    if kl.knee is None:
        return nan

    idx = int(np.argmin(np.abs(BN_data - kl.knee)))    # snap to actual data point
    BN_learned, CE_learned = float(BN_data[idx]), float(CE_data[idx])
    if BN_learned <= 0:
        return nan
    return {"BN_learned": BN_learned, "CE_learned": CE_learned,
            "IPA": abs(CE_o - CE_learned) / BN_learned}


def main():
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    p_dirs = glob.glob(os.path.join(BASE_DIR, "p-percentage_*"))
    pruning_levels = sorted(
        float(re.search(r"p-percentage_([\d.]+)", d).group(1)) for d in p_dirs
    )
    print(f"Found {len(pruning_levels)} pruning percentages: {pruning_levels}")

    inter_by_bs = {}
    for bs in BATCH_SIZES:
        print(f"\n=== Kneedle-only IPA — Batch size {bs} ===")
        rows = []
        for p in pruning_levels:
            avg_csv = os.path.join(BASE_DIR, f"p-percentage_{p}",
                                   f"batch_size_{bs}",
                                   f"averaged_runs_p_{p}_bs_{bs}.csv")
            if not os.path.exists(avg_csv):
                print(f"  [SKIP] P%={p*100:5.1f}%  — missing {avg_csv}")
                continue
            df = pd.read_csv(avg_csv)
            df.columns = df.columns.str.strip()
            ce_col = next((c for c in df.columns
                           if c in ("Avg_CE_Test", "Avg_CE_test")), None)
            bn_col = next((c for c in df.columns if "Batch" in c), None)
            if ce_col is None or bn_col is None:
                print(f"  [SKIP] P%={p*100:5.1f}%  — unexpected columns {list(df.columns)}")
                continue
            df = df.dropna(subset=[ce_col, bn_col]).reset_index(drop=True)

            bns = df[bn_col].values.astype(float)
            ces = df[ce_col].values.astype(float)
            cutoff_BN, step_detected = detect_step_cutoff(bns, ces)

            mask = bns < cutoff_BN
            res = compute_ipa_kneedle(bns[mask], ces[mask])

            step_tag = "[step detected]" if step_detected else "[no step, full data]"
            print(f"  P%={p*100:5.1f}%  cutoff_BN={cutoff_BN:>6.0f} {step_tag:<22}  "
                  f"BN_learned={res['BN_learned']!r:>8}  IPA={res['IPA']}")

            rows.append({"P%": p * 100, "CE_o": CE_o, "cutoff_BN": cutoff_BN,
                         **res})

        if rows:
            bs_df = pd.DataFrame(rows)
            path = os.path.join(INTERMEDIATE_DIR, f"kneedle_only_bs_{bs}.csv")
            bs_df.to_csv(path, index=False)
            print(f"  Saved: {path}")
            inter_by_bs[bs] = bs_df

    # Summary CSV, same column layout as the other methods
    summary_rows = []
    for p in pruning_levels:
        row = {"P%": p * 100}
        for bs in BATCH_SIZES:
            df = inter_by_bs.get(bs)
            sub = df[df["P%"] == p * 100] if df is not None else None
            row[f"IPA_Avg_{bs}"] = (float(sub["IPA"].iloc[0])
                                    if sub is not None and not sub.empty else np.nan)
            row[f"STD_{bs}"] = np.nan   # single averaged curve, no per-run spread
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows, columns=[
        "P%", "IPA_Avg_64", "STD_64", "IPA_Avg_1024", "STD_1024",
        "IPA_Avg_60000", "STD_60000"])
    out_csv = os.path.join(OUT_DIR, "ipa_summary_kneedle_only.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary written: {out_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
