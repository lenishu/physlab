"""
Kneedle-only IPA computation with a noise-robust bottom anchor.

Same as compute_ipa_kneedle_only.py, except the y-normalization anchor of the
Kneedle diagonal. Standard `kneed` uses single extreme points:
    y_norm = (y - min(y)) / (max(y) - min(y))
so the bottom anchor is whichever single noisy tail point dips lowest.

Here, if TAIL_N is not None, the bottom anchor is instead the empirical floor
    floor = mean(CE of the last TAIL_N points of the fit window)
implemented by clipping the curve at the floor before handing it to KneeLocator:
    y_clipped = max(y, floor)   =>   min(y_clipped) == floor
Only noise dips below the floor are affected; the knee region is untouched.
CE_learned / IPA are always taken from the ORIGINAL (unclipped) data.

TAIL_N = None reproduces standard Kneedle (min(y) anchor) exactly.

Usage:  pip install kneed pandas numpy
        python compute_ipa_kneedle_tailavg.py

Outputs (in OUT_DIR):
  intermediate/kneedle_tailavg_bs_{bs}.csv   per-BS elbow + IPA per pruning level
  ipa_summary_kneedle_tailavg.csv            same column layout as the other summaries
"""

import os, glob, re
import numpy as np
import pandas as pd
from kneed import KneeLocator

# ── CONFIG (shared conventions with the other methods) ───────────────────────
BN_STEP_MIN = 100    # don't check for steps before this BN
STEP_THRESH = 0.01   # |CE[i]-CE[i-1]| threshold for step artifact
KNEEDLE_S   = 1.0    # Kneedle sensitivity (larger = more conservative knee)
TAIL_N      = 20     # bottom anchor = mean of last TAIL_N points;
                     # None = standard Kneedle (single-point min(y) anchor)

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


def compute_ipa_kneedle(BN_data, CE_data, S=KNEEDLE_S, tail_n=TAIL_N):
    """
    Kneedle elbow -> IPA. CE vs BN is convex and decreasing.

    tail_n:
      int  -> bottom normalization anchor = mean of the last `tail_n` CE points
              (curve clipped at that floor before knee detection; robust to
              single noise dips in the tail)
      None -> standard Kneedle behavior, anchor = min(CE) (single point)

    Returns dict with floor, BN_learned, CE_learned, IPA (NaN if degenerate).
    CE_learned is read from the ORIGINAL data at the detected knee.
    """
    BN_data = np.asarray(BN_data, float)
    CE_data = np.asarray(CE_data, float)
    nan = {"floor": np.nan, "BN_learned": np.nan,
           "CE_learned": np.nan, "IPA": np.nan}

    if len(BN_data) < 3 or np.ptp(CE_data) <= 1e-10:   # flat curve, e.g. P=100%
        return nan

    if tail_n is not None:
        k = int(min(tail_n, len(CE_data)))
        floor = float(np.mean(CE_data[-k:]))
        if CE_data[0] - floor <= 1e-10:                # degenerate after anchoring
            return nan
        y_for_knee = np.clip(CE_data, floor, None)     # min(y_for_knee) == floor
    else:
        floor = float(np.min(CE_data))
        y_for_knee = CE_data

    try:
        kl = KneeLocator(BN_data, y_for_knee, curve="convex",
                         direction="decreasing", S=S)
    except Exception:
        return nan
    if kl.knee is None:
        return nan

    idx = int(np.argmin(np.abs(BN_data - kl.knee)))    # snap to actual data point
    BN_learned = float(BN_data[idx])
    CE_learned = float(CE_data[idx])                   # original, unclipped CE
    if BN_learned <= 0:
        return nan
    return {"floor": floor, "BN_learned": BN_learned, "CE_learned": CE_learned,
            "IPA": abs(CE_o - CE_learned) / BN_learned}


def main():
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    p_dirs = glob.glob(os.path.join(BASE_DIR, "p-percentage_*"))
    pruning_levels = sorted(
        float(re.search(r"p-percentage_([\d.]+)", d).group(1)) for d in p_dirs
    )
    anchor_tag = f"mean of last {TAIL_N}" if TAIL_N is not None else "min(y) [standard]"
    print(f"Found {len(pruning_levels)} pruning percentages: {pruning_levels}")
    print(f"Kneedle bottom anchor: {anchor_tag}")

    inter_by_bs = {}
    for bs in BATCH_SIZES:
        print(f"\n=== Kneedle (tail-avg anchor) IPA — Batch size {bs} ===")
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
                  f"floor={res['floor']!r:>8}  BN_learned={res['BN_learned']!r:>8}  "
                  f"IPA={res['IPA']}")

            rows.append({"P%": p * 100, "CE_o": CE_o, "cutoff_BN": cutoff_BN,
                         "TAIL_N": np.nan if TAIL_N is None else TAIL_N, **res})

        if rows:
            bs_df = pd.DataFrame(rows)
            path = os.path.join(INTERMEDIATE_DIR, f"kneedle_tailavg_bs_{bs}.csv")
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
    out_csv = os.path.join(OUT_DIR, "ipa_summary_kneedle_tailavg.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary written: {out_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
