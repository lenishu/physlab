# Fitting Data Storage Location & Structure

## SLP-MNIST Fitting Data

All fitting analysis data is organized in the following directories within `SLP/SLP-MNIST/`:

### 1. Fitting Parameters & Summary Statistics
**Directory:** `Fitting_IPA_data/`

| File | Contents | Format |
|------|----------|--------|
| `fitting_IPA_summary_bs_64.csv` | Averaged IPA metrics per pruning % (BS=64) | CSV |
| `fitting_IPA_summary_bs_1024.csv` | Averaged IPA metrics per pruning % (BS=1024) | CSV |
| `fitting_IPA_summary_bs_60000.csv` | Averaged IPA metrics per pruning % (BS=60000) | CSV |
| `fitting_IPA_summary_ALL.csv` | Combined summary across all batch sizes | CSV |
| `fitting_IPA_output.txt` | Plain-text IPA summary table (matches IPA_Analysis format) | TXT |
| `Fitting_Processing_Log.txt` | Detailed processing log (file counts, errors, stats) | TXT |

**Columns in summary CSVs:**
```
Pruning_Percentage, Batch_Size, N_valid_runs,
Avg_CEasy, Avg_B, Avg_n,              [averaged fitted parameters]
Avg_CE_l, Std_CE_l,                   [learning threshold]
Avg_BN_learn, Std_BN_learn,           [learning batch number]
IPA                                   [Information Processing Accuracy]
```

#### Per-Run Parameters
**Directory:** `Fitting_IPA_data/per_run/`

| File | Contents |
|------|----------|
| `fitting_per_run_p_{p}_bs_{bs}.csv` | Individual run fitting parameters (one row per run) |

**Columns:**
```
file, CEasy, B, n, CE_l, BN_learn
```
- `file`: Original run filename (e.g., `slp_0.1_64_run_1.txt`)
- `CEasy, B, n`: Fitted power-law parameters for that run
- `CE_l`: Learning threshold for that run (90% of asymptote gap)
- `BN_learn`: First batch where CE_TEST ≤ CE_l

---

### 2. Fitted Curve Coordinates (For Graphing)
**Directory:** `Fitting_IPA_curves_data/`

Organized by batch size: `BS_64/`, `BS_1024/`, `BS_60000/`

Each batch size directory contains:

#### A. Averaged Raw Data
| File | Contents |
|------|----------|
| `averaged_raw_p_{p}_bs_{bs}.csv` | Averaged CE_TEST across all 100 runs |

**Columns:** `Batch_Number`, `Avg_CE_Test`

**Example:** For p=0%, bs=64:
```
Batch_Number,Avg_CE_Test
0,2.28
1,2.27
2,2.25
...
100,0.15
```

#### B. Mean Fitted Curves
| File | Contents |
|------|----------|
| `mean_fit_p_{p}_bs_{bs}.csv` | Reconstructed mean fit (smooth power-law curve) |

**Columns:** `Batch_Number`, `Fit_CE`

- Computed as: `<CEasy> + <B> / ((Batch_Number + 1)^<n>)`
- Uses averaged parameters from `fitting_IPA_summary_bs_{bs}.csv`
- 600 points evenly spaced over the batch range

#### C. All 100 Individual Fitted Curves
| File | Contents |
|------|----------|
| `per_run_fits_p_{p}_bs_{bs}.csv` | All 100 individual fitted curves |

**Columns:** `Run_ID`, `Batch_Number`, `Fitted_CE`, `CEasy`, `B`, `n`

- **Run_ID**: Original filename (e.g., `slp_0.1_64_run_1.txt`)
- **Batch_Number**: X-coordinate
- **Fitted_CE**: Y-coordinate for that run's fitted curve
- **CEasy, B, n**: That run's fitting parameters

**Example structure (100 runs × ~200 batch points = 20,000 rows per file):**
```
Run_ID,Batch_Number,Fitted_CE,CEasy,B,n
slp_0.0_64_run_1.txt,0,2.30,0.15,2.15,0.8
slp_0.0_64_run_1.txt,1,2.29,0.15,2.15,0.8
slp_0.0_64_run_1.txt,2,2.27,0.15,2.15,0.8
...
slp_0.0_64_run_2.txt,0,2.31,0.16,2.15,0.75
slp_0.0_64_run_2.txt,1,2.30,0.16,2.15,0.75
...
```

---

### 3. Visualization Plots
**Directory:** `Fitting_IPA_graphs/`

| File | Description |
|------|-------------|
| `CE_Test_Fit_Avg_BS_{bs}.png` | Faint raw + bold fitted curves for all P% (300 DPI) |
| `IPA_vs_Pruning_BS_{bs}.png` | **Line plot**: IPA vs pruning percentage |
| `BNL_vs_Pruning_BS_{bs}.png` | Bar chart: Learning batch number vs pruning % |
| `CEL_vs_Pruning_BS_{bs}.png` | Bar chart: Learning threshold vs pruning % |

---

## Complete Directory Tree

```
SLP/SLP-MNIST/
├── Fitting_IPA_data/
│   ├── fitting_IPA_summary_bs_64.csv
│   ├── fitting_IPA_summary_bs_1024.csv
│   ├── fitting_IPA_summary_bs_60000.csv
│   ├── fitting_IPA_summary_ALL.csv
│   ├── fitting_IPA_output.txt
│   ├── Fitting_Processing_Log.txt
│   └── per_run/
│       ├── fitting_per_run_p_0.0_bs_64.csv
│       ├── fitting_per_run_p_0.0_bs_1024.csv
│       ├── fitting_per_run_p_0.0_bs_60000.csv
│       ├── fitting_per_run_p_0.1_bs_64.csv
│       ⋮
├── Fitting_IPA_curves_data/
│   ├── BS_64/
│   │   ├── averaged_raw_p_0.0_bs_64.csv
│   │   ├── mean_fit_p_0.0_bs_64.csv
│   │   ├── per_run_fits_p_0.0_bs_64.csv
│   │   ├── averaged_raw_p_0.1_bs_64.csv
│   │   ⋮
│   ├── BS_1024/
│   │   ├── averaged_raw_p_0.0_bs_1024.csv
│   │   ├── mean_fit_p_0.0_bs_1024.csv
│   │   ├── per_run_fits_p_0.0_bs_1024.csv
│   │   ⋮
│   └── BS_60000/
│       ├── averaged_raw_p_0.0_bs_60000.csv
│       ├── mean_fit_p_0.0_bs_60000.csv
│       ├── per_run_fits_p_0.0_bs_60000.csv
│       ⋮
└── Fitting_IPA_graphs/
    ├── CE_Test_Fit_Avg_BS_64.png
    ├── CE_Test_Fit_Avg_BS_1024.png
    ├── CE_Test_Fit_Avg_BS_60000.png
    ├── IPA_vs_Pruning_BS_64.png
    ├── IPA_vs_Pruning_BS_1024.png
    ├── IPA_vs_Pruning_BS_60000.png
    ├── BNL_vs_Pruning_BS_64.png
    ├── BNL_vs_Pruning_BS_1024.png
    ├── BNL_vs_Pruning_BS_60000.png
    ├── CEL_vs_Pruning_BS_64.png
    ├── CEL_vs_Pruning_BS_1024.png
    └── CEL_vs_Pruning_BS_60000.png
```

---

## How to Use This Data

### Option 1: Recreate Plots with Your Own Code
1. Load `Fitting_IPA_curves_data/BS_{bs}/averaged_raw_p_{p}_bs_{bs}.csv`
2. Load `Fitting_IPA_curves_data/BS_{bs}/mean_fit_p_{p}_bs_{bs}.csv`
3. Load `Fitting_IPA_curves_data/BS_{bs}/per_run_fits_p_{p}_bs_{bs}.csv` (optional, for individual run visualization)
4. Plot using matplotlib, R, or any graphing tool

### Option 2: Access IPA Results
1. Load `Fitting_IPA_data/fitting_IPA_summary_bs_{bs}.csv`
2. Read IPA values, BN_learn, CE_l, etc. directly
3. Or read the text summary in `Fitting_IPA_output.txt`

### Option 3: Access Per-Run Details
1. Load `Fitting_IPA_data/per_run/fitting_per_run_p_{p}_bs_{bs}.csv`
2. Filter or analyze individual run fitting parameters
3. Reconstruct fitted curves: `y = CEasy + B / ((x+1)^n)`

---

## Key Advantages

✓ **No Re-fitting Required**: All fitted curve coordinates are pre-computed and saved
✓ **Multiple Levels**: Access individual run details, averages, or final statistics
✓ **Ready for Analysis**: Use curve CSVs directly with any plotting/analysis tool
✓ **Reproducible**: Fitted parameters and curves can be regenerated deterministically
✓ **Comprehensive Logging**: Know exactly what processing was done and any issues encountered

---

## Data Generated By

**Notebook:** `SLP/SLP-MNIST/fitting_function_IPA.ipynb`

- **Cell 2**: Computes per-run fitting → generates `per_run/` CSVs and summary CSVs
- **Cell 3**: Loads summaries → reconstructs and saves `Fitting_IPA_curves_data/` CSVs
- **Cell 3.5**: Reconstructs all 100 individual fits → saves `per_run_fits_*.csv`
- **Cell 5**: Creates plots from curve data → saves PNG files

**Date Last Updated:** 2026-03-13
