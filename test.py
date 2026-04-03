import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
# Base directory containing all pruning experiments
BASE_DIR = r"C:\Users\Student\Desktop\Neural_research\physlab\SLP\SLP-MNIST\prune_layers_ALL"

# Directory template: "p-percentage_0.1\batch_size_64" etc.
BATCH_DIR_TEMPLATE = "p-percentage_{:.1f}\\batch_size_{}"

# File pattern for individual run files
# Pattern: "slp_0.1_64_run_0.txt", "slp_0.1_64_run_1.txt", etc.
FILE_PATTERN = "slp_{:.1f}_{}_run_*"

# Batch sizes to analyze
BATCH_SIZES = [64, 1024, 60000]

# Mathematical constant: ln(10) ≈ 2.303 (max CE for 10-class classification)
LN10 = np.log(10)

# =========================
# COLOR SCHEME FOR PRUNING PERCENTAGES
# =========================
# 11 colors for P% = 0%, 10%, ..., 100%
COLOR_LIST = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#800080",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#B9D9EB", "#17becf"
]

# Reverse so P%=0% appears last in legend (top of legend box)
COLOR_LIST = COLOR_LIST[::-1]

# =========================
# MATPLOTLIB STYLING (Nature-like publication format)
# =========================
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15
})

# =========================
# MAIN LOOP: Process each batch size
# =========================
for bs in BATCH_SIZES:
    print(f"\n{'='*50}")
    print(f"Processing batch size: {bs}")
    print(f"{'='*50}")
    
    all_avg_dfs = {}  # Store averaged DataFrames for each pruning %

    # ===============================================
    # STEP 1: LOAD & AVERAGE ALL RUNS FOR THIS BS
    # ===============================================
    for idx, p in enumerate([round(x * 0.1, 1) for x in range(0, 11)]):
        # Construct folder path for this pruning % and batch size
        folder = os.path.join(BASE_DIR, BATCH_DIR_TEMPLATE.format(p, bs))
        
        # Find all run files matching the pattern
        pattern = FILE_PATTERN.format(p, bs)
        files = glob.glob(os.path.join(folder, pattern))

        if not files:
            print(f"  [WARNING] No files found for P%={p*100:.0f}, BS={bs}")
            continue

        print(f"  Loading P%={p*100:.0f} ({len(files)} runs)...")

        # Load all individual runs
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=r"\s+")  # Whitespace-delimited
            df.columns = df.columns.str.strip()  # Remove trailing spaces from column names

            # Convert string columns to numeric, replacing errors with NaN
            df["CE_Train"] = pd.to_numeric(df["CE_Train"], errors="coerce")
            df["CE_TEST"] = pd.to_numeric(df["CE_TEST"], errors="coerce")
            df["Accuracy(%)"] = pd.to_numeric(df["Accuracy(%)"], errors="coerce")

            dfs.append(df)

        # Concatenate all runs into one DataFrame
        all_runs = pd.concat(dfs, ignore_index=True)

        # ===============================================
        # STEP 2: COMPUTE AVERAGE BY BATCH NUMBER
        # ===============================================
        # Group by Batch_Number and compute means across all runs
        avg_df = all_runs.groupby("Batch_Number", as_index=False).agg(
            Avg_CE_Train=("CE_Train", "mean"),
            Avg_CE_Test=("CE_TEST", "mean"),
            Avg_Accuracy=("Accuracy(%)", "mean"),
            Num_Runs=("CE_TEST", "count")  # Count valid runs per batch
        )

        # ===============================================
        # STEP 3: SAVE AVERAGED CSV
        # ===============================================
        out_csv = os.path.join(folder, f"averaged_runs_p_{p}_bs_{bs}.csv")
        avg_df.to_csv(out_csv, index=False)
        print(f"    → Saved: {os.path.basename(out_csv)}")

        # Store for plotting
        all_avg_dfs[p] = avg_df

    # ===============================================
    # STEP 4: CREATE COMBINED PLOTS (3 metrics)
    # ===============================================

    # =========== PLOT 1: CE_Train vs Batch Number ===========
    print(f"\n  Creating CE_Train plot...")
    plt.figure(figsize=(12, 6))
    
    for idx, (p, avg_df) in enumerate(all_avg_dfs.items()):
        color = COLOR_LIST[idx % len(COLOR_LIST)]
        plt.plot(avg_df["Batch_Number"], avg_df["Avg_CE_Train"],
                 label=f"P%={round(p * 100)}", color=color, linewidth=2)

    # Labels and limits
    plt.xlabel("Batch Number")
    plt.ylabel("Average CE")
    plt.ylim(0, 2.5)
    plt.yticks([x * 0.25 for x in range(0, 11)])

    # Add ln(10) reference line
    plt.text(0.06, LN10 + 0.05, r"$\ln(10)$",
             transform=plt.gca().get_yaxis_transform(),
             fontsize=14, va="center", ha="left")

    plt.gca().yaxis.set_minor_locator(plt.FixedLocator([LN10]))
    plt.tick_params(axis='y', which='minor', length=5, color='black')

    # Title
    plt.text(0.50, 0.95, "Cross-Entropy",
             transform=plt.gca().transAxes, va="top", ha="center")
    plt.text(0.50, 0.90,
             f"(SLP Average of MNIST Training-Vectors, BS={bs})",
             transform=plt.gca().transAxes, va="top", ha="center", fontsize=16)

    # Legend (reversed: P%=100 appears at top)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1],
               loc="upper right", bbox_to_anchor=(1.0, 1.02),
               frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"CE_Train_Avg_SLP_MNIST_BS_{bs}.png",
                bbox_inches="tight", dpi=300)
    print(f"    → Saved: CE_Train_Avg_SLP_MNIST_BS_{bs}.png")
    plt.show()

    # =========== PLOT 2: CE_Test vs Batch Number ===========
    print(f"  Creating CE_Test plot...")
    plt.figure(figsize=(12, 6))
    
    for idx, (p, avg_df) in enumerate(all_avg_dfs.items()):
        color = COLOR_LIST[idx % len(COLOR_LIST)]
        plt.plot(avg_df["Batch_Number"], avg_df["Avg_CE_Test"],
                 label=f"P%={round(p * 100)}", color=color, linewidth=2)

    plt.xlabel("Batch Number")
    plt.ylabel("Average CE")
    plt.ylim(0, 2.5)
    plt.yticks([x * 0.25 for x in range(0, 11)])

    # Add ln(10) reference line
    plt.text(0.06, LN10 + 0.05, r"$\ln(10)$",
             transform=plt.gca().get_yaxis_transform(),
             fontsize=14, va="center", ha="left")

    plt.gca().yaxis.set_minor_locator(plt.FixedLocator([LN10]))
    plt.tick_params(axis='y', which='minor', length=5, color='black')

    plt.text(0.50, 0.95, "Cross-Entropy",
             transform=plt.gca().transAxes, va="top", ha="center")
    plt.text(0.50, 0.90,
             f"(SLP Average of MNIST Test-Vectors, BS={bs})",
             transform=plt.gca().transAxes, va="top", ha="center", fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1],
               loc="upper right", frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"CE_Test_Avg_SLP_MNIST_BS_{bs}.png",
                bbox_inches="tight", dpi=300)
    print(f"    → Saved: CE_Test_Avg_SLP_MNIST_BS_{bs}.png")
    plt.show()

    # =========== PLOT 3: Accuracy vs Batch Number ===========
    print(f"  Creating Accuracy plot...")
    plt.figure(figsize=(12, 6))
    
    for idx, (p, avg_df) in enumerate(all_avg_dfs.items()):
        color = COLOR_LIST[idx % len(COLOR_LIST)]
        plt.plot(avg_df["Batch_Number"], avg_df["Avg_Accuracy"],
                 label=f"P%={round(p * 100)}", color=color, linewidth=2)

    plt.xlabel("Batch Number")
    plt.ylabel("Average Accuracy (%)")
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))

    plt.text(0.50, 0.55, "Accuracy",
             transform=plt.gca().transAxes, va="top", ha="center")
    plt.text(0.50, 0.50,
             f"(SLP Average of MNIST Test-Vectors, BS={bs})",
             transform=plt.gca().transAxes, va="top", ha="center", fontsize=16)

    # NOTE: Accuracy legend NOT reversed (lower right, P%=100 at bottom)
    plt.legend(loc="lower right",
               bbox_to_anchor=(1.0, 0.05),
               frameon=False)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Accuracy_Avg_SLP_MNIST_BS_{bs}.png",
                bbox_inches="tight", dpi=300)
    print(f"    → Saved: Accuracy_Avg_SLP_MNIST_BS_{bs}.png")
    plt.show()

print(f"\n{'='*50}")
print("✓ All plots completed successfully!")
print(f"{'='*50}")