"""
Plot IPA vs Pruning% for each of the four approaches, using only the
ipa_summary_approach_*.csv files. One PNG per approach, saved next to its CSV.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

APPROACHES = [
    ("Approach_1",   "1",   "Approach 1 — Fit on averaged raw CE"),
    ("Approach_2a1", "2a1", "Approach 2a1 — Refit averaged per-run fitted curves"),
    ("Approach_2a2", "2a2", "Approach 2a2 — Average per-run (A, B, n) directly"),
    ("Approach_2b",  "2b",  "Approach 2b — Per-run IPA, averaged across runs"),
]

BS_LIST  = [64, 1024, 60000]
BS_COLOR = {64: "#1f77b4", 1024: "#d62728", 60000: "#2ca02c"}

plt.rcParams.update({"font.size": 14})

for dir_name, tag, title in APPROACHES:
    csv_path = os.path.join(HERE, dir_name, f"ipa_summary_approach_{tag}.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for bs in BS_LIST:
        mean_col = f"IPA_Avg_{bs}"
        std_col  = f"STD_{bs}"
        sub = df.dropna(subset=[mean_col])
        if sub.empty:
            continue
        x = sub["P%"].values
        y = sub[mean_col].values
        kwargs = dict(label=f"BS={bs}", color=BS_COLOR[bs],
                      marker="o", markersize=6, linewidth=2)
        if tag == "2b":
            yerr = sub[std_col].values
            ax.errorbar(x, y, yerr=yerr, capsize=3, **kwargs)
        else:
            ax.plot(x, y, **kwargs)

    ax.set_xlabel("Pruning Percentage (%)")
    ax.set_ylabel("IPA")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    out_png = os.path.join(HERE, dir_name, f"ipa_plot_approach_{tag}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
