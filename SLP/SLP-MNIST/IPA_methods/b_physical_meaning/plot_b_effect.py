"""
B physical meaning experiment.

Shows for one run:
  - raw CE_TEST scatter from the training log
  - fitted curve using the actual (A, B, n) from the fitting CSV
  - an overlay curve where only B is replaced by B_OVERRIDE

Lets you vary B to see what it controls in:
    CE(BN) = A + B / (BN + 1)^n

CONFIG: change the four variables below.
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────
RUN_INDEX  = 0      # integer: which run (0-based)
BATCH_SIZE = 64     # 64 | 1024 | 60000
PRUNING    = 0.5    # pruning fraction  0.0 – 1.0  (e.g. 0.5 = 50%)
B_OVERRIDE = None   # float to replace B, or None to use the fitted value
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = (r"C:\Users\Student\Desktop\Projects\research\physlab"
            r"\SLP\SLP-MNIST\prune_layers_ALL")
FIT_DIR  = (r"C:\Users\Student\Desktop\Projects\research\physlab"
            r"\SLP\SLP-MNIST\Fitting_IPA_curves_data_I")
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
CE_o     = np.log(10)

# ── Raw data ──────────────────────────────────────────────────────────────────
raw_name = f"slp_{PRUNING}_{BATCH_SIZE}_run_{RUN_INDEX}.txt"
raw_path = os.path.join(BASE_DIR, f"p-percentage_{PRUNING}",
                        f"batch_size_{BATCH_SIZE}", raw_name)
raw_df   = pd.read_csv(raw_path, sep=r"\s+", engine="python")
raw_df.columns = raw_df.columns.str.strip()
bn_raw   = raw_df["Batch_Number"].values.astype(float)
ce_raw   = raw_df["CE_TEST"].values.astype(float)

# ── Fitted parameters for this run ───────────────────────────────────────────
fit_csv  = os.path.join(FIT_DIR, f"BS_{BATCH_SIZE}",
                        f"per_run_fits_p_{PRUNING}_bs_{BATCH_SIZE}.csv")
fit_df   = pd.read_csv(fit_csv)
fit_df.columns = fit_df.columns.str.strip()

run_label = f"slp_{PRUNING}_{BATCH_SIZE}_run_{RUN_INDEX}.txt"
sub = fit_df[fit_df["Run"] == run_label]
if sub.empty:
    available = fit_df["Run"].unique()[:5]
    raise ValueError(f"Run '{run_label}' not found. Available (first 5): {available}")

A   = float(sub["A"].iloc[0])
B   = float(sub["B"].iloc[0])
n   = float(sub["n"].iloc[0])
B_override = float(B_OVERRIDE) if B_OVERRIDE is not None else B

# ── Derived quantities ────────────────────────────────────────────────────────
CE_L = CE_o - 0.9 * (CE_o - A)   # learning threshold — depends only on A

def ce_curve(bn, B_val):
    return A + B_val / ((bn + 1) ** n)

def get_bnl(B_val):
    denom = CE_L - A
    if denom <= 0 or n <= 0 or B_val <= 0:
        return np.nan
    val = (B_val / denom) ** (1.0 / n) - 1.0
    if not np.isfinite(val) or val <= 0:
        return np.nan
    return float(np.ceil(val))

def get_ipa(B_val):
    bnl = get_bnl(B_val)
    if not np.isfinite(bnl) or bnl == 0:
        return np.nan
    return abs(CE_o - CE_L) / bnl

BNL_fit      = get_bnl(B)
BNL_override = get_bnl(B_override)
IPA_fit      = get_ipa(B)
IPA_override = get_ipa(B_override)

# ── Smooth BN range ───────────────────────────────────────────────────────────
bn_end    = max(bn_raw.max(),
                BNL_fit      if np.isfinite(BNL_fit)      else 0,
                BNL_override if np.isfinite(BNL_override) else 0) * 1.25
bn_smooth = np.linspace(0, bn_end, 800)

y_fit      = ce_curve(bn_smooth, B)
y_override = ce_curve(bn_smooth, B_override)

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(11, 6.5))

# Raw scatter
ax.scatter(bn_raw, ce_raw, s=6, color="#aaaaaa", alpha=0.5, zorder=1,
           label="Raw CE_TEST")

# Fitted curve with actual B
lbl_fit = (f"Fit  A={A:.4f}, B={B:.4f}, n={n:.4f}"
           + (f"   BNL={BNL_fit:.0f},  IPA={IPA_fit:.5f}"
              if np.isfinite(BNL_fit) else "   BNL=NaN"))
ax.plot(bn_smooth, y_fit, color="#1f77b4", linewidth=2.2, zorder=3,
        label=lbl_fit)

# Override curve (only when B actually differs)
show_override = B_OVERRIDE is not None and not np.isclose(B_override, B)
if show_override:
    lbl_over = (f"B_override={B_override:.4f}  (A, n fixed)"
                + (f"   BNL={BNL_override:.0f},  IPA={IPA_override:.5f}"
                   if np.isfinite(BNL_override) else "   BNL=NaN"))
    ax.plot(bn_smooth, y_override, color="#d62728", linewidth=2.2,
            linestyle="--", zorder=3, label=lbl_over)

# Horizontal reference lines
x_right = bn_end
ax.axhline(CE_o, color="#888888", linewidth=1.0, linestyle=":")
ax.text(x_right, CE_o + 0.03, f"CE_o = {CE_o:.4f}", ha="right",
        fontsize=10, color="#666666")

ax.axhline(CE_L, color="#9467bd", linewidth=1.4, linestyle="--")
ax.text(x_right, CE_L + 0.03, f"CE_L = {CE_L:.4f}", ha="right",
        fontsize=10, color="#9467bd")

ax.axhline(A, color="#2ca02c", linewidth=1.4, linestyle="--")
ax.text(x_right, A - 0.07, f"A = {A:.4f}  (asymptote)", ha="right",
        fontsize=10, color="#2ca02c")

# BNL verticals
if np.isfinite(BNL_fit):
    ax.axvline(BNL_fit, color="#1f77b4", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.annotate(f"BNL={BNL_fit:.0f}", xy=(BNL_fit, CE_L),
                xytext=(BNL_fit + bn_end * 0.02, CE_L + 0.15),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))

if show_override and np.isfinite(BNL_override):
    ax.axvline(BNL_override, color="#d62728", linewidth=1.2,
               linestyle="--", alpha=0.6)
    ax.annotate(f"BNL={BNL_override:.0f}", xy=(BNL_override, CE_L),
                xytext=(BNL_override + bn_end * 0.02, CE_L - 0.18),
                fontsize=10, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

ax.set_xlabel("Batch Number (BN)")
ax.set_ylabel("CE_TEST")
ax.set_xlim(0, bn_end)
ax.set_ylim(max(0, A - 0.15), CE_o + 0.25)
ax.set_title(
    f"B physical meaning — run {RUN_INDEX}, BS={BATCH_SIZE}, P%={PRUNING*100:.1f}%\n"
    f"CE(BN) = A + B / (BN+1)^n    "
    f"B shifts starting height; CE_L and A are fixed by A alone"
)
ax.legend(fontsize=10, frameon=False, loc="upper right")
ax.grid(True, alpha=0.25)

b_tag = f"_B{B_override:.3f}" if show_override else ""
out_png = os.path.join(OUT_DIR,
    f"b_effect_run{RUN_INDEX}_bs{BATCH_SIZE}_p{int(PRUNING*100)}{b_tag}.png")
plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_png}")
print(f"  A={A:.4f}  B={B:.4f}  n={n:.4f}")
print(f"  CE_o={CE_o:.4f}  CE_L={CE_L:.4f}")
print(f"  BNL={BNL_fit}  IPA={IPA_fit}")
if show_override:
    print(f"  BNL_override={BNL_override}  IPA_override={IPA_override}")
