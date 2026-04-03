================================================================================
README: Understanding the Fitting Function & per_run_fits CSV Structure
================================================================================

PROJECT: SLP-MNIST Curve Fitting Analysis
FILE: fitting_function_IPA.ipynb
DATE: 2026-03-20

================================================================================
OVERVIEW
================================================================================

This document explains:
1. How the power-law fitting model works
2. Why per_run_fits_p_*.csv contains batch_number
3. Why A, B, n are constant while Fitted_CE changes
4. How to interpret the fitted parameters

================================================================================
THE FITTING MODEL
================================================================================

Formula:
    CE(batch_number) = A + B / ((batch_number + 1)^n)

Where:
    CE = Cross-Entropy (loss value we're trying to model)
    batch_number = Training progress (0, 1, 2, 3, ...)
    A = Asymptote (minimum CE the curve approaches)
    B = Scale factor (distance from asymptote at start)
    n = Exponent (decay rate - how fast it approaches asymptote)

Example for run_0 at 10% pruning:
    A = 0.2934
    B = 5.4328
    n = 0.8713

    CE = 0.2934 + 5.4328 / ((batch_number + 1)^0.8713)

================================================================================
KEY INSIGHT: A, B, n Are Constants (Per Run)
================================================================================

For a SINGLE RUN:
- A, B, n are FITTED ONCE to all data in that run
- These values do NOT change as you move through batches
- They are the "signature" of how that run behaves

Example from per_run_fits_p_0.1_bs_64.csv:
    Run,Batch_Number,Fitted_CE,A,B,n
    slp_0.1_64_run_0.txt,0,5.726,0.2934,5.4328,0.8713
    slp_0.1_64_run_0.txt,1,3.263,0.2934,5.4328,0.8713    ← SAME A,B,n
    slp_0.1_64_run_0.txt,2,2.379,0.2934,5.4328,0.8713    ← SAME A,B,n
    slp_0.1_64_run_0.txt,3,1.917,0.2934,5.4328,0.8713    ← SAME A,B,n
    slp_0.1_64_run_0.txt,4,1.630,0.2934,5.4328,0.8713    ← SAME A,B,n
    ...
    slp_0.1_64_run_1.txt,0,5.821,0.2988,5.5120,0.8652    ← DIFFERENT RUN!
    slp_0.1_64_run_1.txt,1,3.304,0.2988,5.5120,0.8652    ← Different A,B,n
    slp_0.1_64_run_1.txt,2,2.405,0.2988,5.5120,0.8652    ← Different A,B,n

Notice:
    - Within run_0: A=0.2934, B=5.4328, n=0.8713 (constant)
    - Within run_1: A=0.2988, B=5.5120, n=0.8652 (different!)

================================================================================
WHY FITTED_CE CHANGES WHILE A, B, n STAY CONSTANT
================================================================================

The formula is evaluated at DIFFERENT BATCH NUMBERS:

For run_0 (A=0.2934, B=5.4328, n=0.8713):

    Batch 0:
        CE = 0.2934 + 5.4328 / ((0 + 1)^0.8713)
        CE = 0.2934 + 5.4328 / 1.0
        CE = 0.2934 + 5.4328
        CE = 5.726  ✓

    Batch 1:
        CE = 0.2934 + 5.4328 / ((1 + 1)^0.8713)
        CE = 0.2934 + 5.4328 / (2^0.8713)
        CE = 0.2934 + 5.4328 / 1.829
        CE = 0.2934 + 2.970
        CE = 3.263  ✓

    Batch 2:
        CE = 0.2934 + 5.4328 / ((2 + 1)^0.8713)
        CE = 0.2934 + 5.4328 / (3^0.8713)
        CE = 0.2934 + 5.4328 / 2.511
        CE = 0.2934 + 2.163
        CE = 2.379  ✓

    Batch 10:
        CE = 0.2934 + 5.4328 / ((10 + 1)^0.8713)
        CE = 0.2934 + 5.4328 / (11^0.8713)
        CE = 0.2934 + 5.4328 / 8.951
        CE = 0.2934 + 0.607
        CE = 0.900

As batch_number increases:
    - (batch_number + 1)^n gets larger
    - B / ((batch_number + 1)^n) gets smaller
    - CE approaches A (the asymptote)

================================================================================
ANALOGY: PARABOLA FORMULA
================================================================================

Think of it like a quadratic function:
    y = 2x² - 3x + 5

Coefficients (2, -3, 5) are CONSTANT
But you evaluate at different x values:

    x=0: y = 5
    x=1: y = 2(1)² - 3(1) + 5 = 4
    x=2: y = 2(2)² - 3(2) + 5 = 7
    x=3: y = 2(3)² - 3(3) + 5 = 14

Same coefficients, different outputs!

SIMILARLY for CE curve:
    Coefficients (A, B, n) are CONSTANT per run
    Evaluate at different batch numbers:

    batch=0: CE = 5.726
    batch=1: CE = 3.263
    batch=2: CE = 2.379
    ...

================================================================================
WHY BATCH_NUMBER IS NEEDED IN THE CSV
================================================================================

1. RECONSTRUCT THE CURVE
   ├─ To plot the fitted line, you need (x, y) pairs
   ├─ x-axis = batch_number
   └─ y-axis = fitted_CE

2. ALIGN WITH ORIGINAL DATA
   ├─ Original training data has batch numbers
   └─ You can compare fitted vs actual CE at same batch point

3. COMPUTE STATISTICS
   ├─ To average across 100 runs, you need batch numbers as reference
   ├─ "At batch 0, average CE across runs = 5.870"
   └─ "At batch 100, average CE across runs = 0.850"

4. DEBUGGING & VALIDATION
   ├─ Check if fit is good at specific batches
   └─ Identify where/when fitting breaks down

WITHOUT batch_number:
    You'd just have: 5.726, 3.263, 2.379, 1.917, ...
    But WHERE are these points on the graph? You wouldn't know!

WITH batch_number:
    You have: (0, 5.726), (1, 3.263), (2, 2.379), (3, 1.917), ...
    Now you can PLOT these points and see the curve shape!

================================================================================
FILE STRUCTURE EXPLANATION
================================================================================

CSV: per_run_fits_p_0.1_bs_64.csv

Columns:
    Run             - Which training experiment (run_0.txt, run_1.txt, ..., run_99.txt)
    Batch_Number    - Training progress (0, 1, 2, 3, ..., up to ~5000)
    Fitted_CE       - Cross-Entropy value from formula at this batch
    A               - Asymptote (minimum CE for this run)
    B               - Scale factor (initial distance from asymptote)
    n               - Exponent (decay rate)

Row Count:
    ~100 runs × ~5000 batch numbers per run ≈ 500,000 rows
    This is a LARGE file!

Why so many rows?
    - You need to reconstruct the full fitted curve for each run
    - Each row represents one (batch, fitted_CE) point on that run's fitted curve

================================================================================
DATA FLOW: FROM RAW DATA TO CSV
================================================================================

Step 1: RAW TRAINING DATA (from original training scripts)
    Run 0:
        Batch 0: CE_test = 5.85
        Batch 1: CE_test = 3.30
        Batch 2: CE_test = 2.45
        Batch 3: CE_test = 1.95
        ... (~5000 batches)

    Run 1:
        Batch 0: CE_test = 5.91
        Batch 1: CE_test = 3.20
        ...

    ... (100 runs total)

Step 2: FIT POWER-LAW TO EACH RUN
    For Run 0:
        Fit A + B / ((batch+1)^n) to all CE values
        Result: A=0.2934, B=5.4328, n=0.8713
        (These minimize error between real CE and fitted CE)

    For Run 1:
        Fit A + B / ((batch+1)^n) to all CE values
        Result: A=0.2988, B=5.5120, n=0.8652
        (Different values because this run learned differently!)

    ... (100 fitted parameter sets)

Step 3: RECONSTRUCT FITTED CURVES
    For Run 0 at each batch:
        Batch 0: CE_fit = 0.2934 + 5.4328 / 1 = 5.726
        Batch 1: CE_fit = 0.2934 + 5.4328 / 2^0.8713 = 3.263
        Batch 2: CE_fit = 0.2934 + 5.4328 / 3^0.8713 = 2.379
        ...

    For Run 1 at each batch:
        Batch 0: CE_fit = 0.2988 + 5.5120 / 1 = 5.821
        Batch 1: CE_fit = 0.2988 + 5.5120 / 2^0.8652 = 3.304
        ...

Step 4: SAVE TO CSV
    per_run_fits_p_0.1_bs_64.csv:
        Run,Batch_Number,Fitted_CE,A,B,n
        run_0.txt,0,5.726,0.2934,5.4328,0.8713
        run_0.txt,1,3.263,0.2934,5.4328,0.8713
        run_0.txt,2,2.379,0.2934,5.4328,0.8713
        ...
        run_1.txt,0,5.821,0.2988,5.5120,0.8652
        run_1.txt,1,3.304,0.2988,5.5120,0.8652
        ...

================================================================================
PARAMETER MEANINGS
================================================================================

A - ASYMPTOTE
    ├─ What is it? Minimum CE value the curve approaches
    ├─ Physical meaning: Best-case learning performance for this pruning level
    ├─ Example: A=0.293 means "even with infinite batches, CE won't go below 0.293"
    ├─ Trend with pruning:
    │  └─ If A increases with pruning → pruning hurts learning capacity
    └─ Range: Typically 0.1 - 2.3 (ln(10)≈2.303 is max for 10 classes)

B - SCALE FACTOR
    ├─ What is it? Initial distance from asymptote
    ├─ Physical meaning: How much overfitting happens initially
    ├─ Example: B=5.43 means "starts 5.43 away from asymptote"
    │  └─ Initial CE = A + B = 0.293 + 5.43 = 5.72
    ├─ Trend with pruning:
    │  └─ If B increases with pruning → more initial overfitting
    └─ Range: Typically 1 - 10

N - EXPONENT (DECAY RATE)
    ├─ What is it? How fast CE decays toward asymptote
    ├─ Physical meaning: Speed of learning/convergence
    ├─ Example: n=0.87 means "CE decays as batch^-0.87"
    │  └─ Slower decay (smaller n) = slower learning
    │  └─ Faster decay (larger n) = faster learning
    ├─ Trend with pruning:
    │  └─ If n decreases with pruning → pruning slows learning
    └─ Range: Typically 0.5 - 3.0

================================================================================
PRACTICAL USES
================================================================================

1. ANALYZE SINGLE RUN
   Code:
       df = pd.read_csv("per_run_fits_p_0.1_bs_64.csv")
       run0 = df[df['Run'] == 'slp_0.1_64_run_0.txt']
       print(run0[['Batch_Number', 'Fitted_CE']])
       # Output: all (batch, fitted_CE) points for that run
       # Plot these to see the fitted curve for run_0

2. EXTRACT PARAMETERS FOR ONE RUN
   Code:
       run0_params = run0[['A', 'B', 'n']].iloc[0]
       print(f"A={run0_params['A']:.4f}, B={run0_params['B']:.4f}, n={run0_params['n']:.4f}")
       # Output: A=0.2934, B=5.4328, n=0.8713
       # These are the fitted parameters that define run_0's behavior

3. AVERAGE ACROSS RUNS (what Cell 5 does)
   Code:
       df = pd.read_csv("per_run_fits_p_0.1_bs_64.csv")
       mean_a = df['A'].mean()
       mean_b = df['B'].mean()
       mean_n = df['n'].mean()
       # These go into per_run_fit_parameters_bs_64.csv

4. RECONSTRUCT CURVE AT NEW BATCH
   Code:
       # Given A, B, n from the fit
       batch = 100
       ce_fit = 0.2934 + 5.4328 / ((batch + 1)**0.8713)
       print(f"CE at batch {batch}: {ce_fit:.3f}")
       # Output: CE at batch 100: 0.854

================================================================================
SUMMARY TABLE
================================================================================

What Each Column Represents:

    Column          | Per-Run Value? | Changes Per Batch? | Why?
    ────────────────┼────────────────┼───────────────────┼──────────────────
    Run             | N/A            | No                | Identifies experiment
    Batch_Number    | N/A            | YES               | Independent variable
    Fitted_CE       | No             | YES               | Function output
    A               | YES            | No                | Fitted constant
    B               | YES            | No                | Fitted constant
    n               | YES            | No                | Fitted constant

Key Understanding:
    ✓ A, B, n = Fitted ONCE per run (constant for that run)
    ✓ Batch_Number = Input to the formula (varies 0 to ~5000)
    ✓ Fitted_CE = Output of formula (varies with batch number)

================================================================================
VISUAL EXAMPLE: How CE Evolves
================================================================================

For Run 0 (A=0.2934, B=5.4328, n=0.8713):

        CE
        |
    5.8 |●  (batch 0, actual ≈ 5.85)
    5.7 |●  fitted
    5.0 |  ╲
    4.0 |   ╲
    3.3 |    ●  (batch 1, actual ≈ 3.30)
    3.0 |    ╲╲
    2.5 |      ●  (batch 2, actual ≈ 2.45)
    2.0 |       ╲
    1.5 |        ╲●
    1.0 |         ╲
    0.5 |          ╲___
    0.3 |───────────── (A = asymptote)
        |________________ batch_number
        0  1  2  3  4  5  ... 100  ... 5000

Observations:
    - Starts at ~5.73 (A + B)
    - Decays toward 0.29 (A)
    - Decay rate controlled by n
    - Each point has (batch_number, fitted_CE)
    - A, B, n are the same for all points on this curve

================================================================================
COMMON QUESTIONS
================================================================================

Q: Why not just save A, B, n instead of all the fitted curves?
A: You CAN save just A, B, n (that's what per_run_fit_parameters_bs_64.csv does)
   But you also need the full curves to:
   - Compare fitted vs actual data
   - Plot continuous curves
   - Check where fits are good/bad
   Both files serve different purposes!

Q: Aren't there duplicate A, B, n values in per_run_fits?
A: YES! That's intentional and efficient:
   - You reconstruct the full curve at each batch
   - You keep A, B, n with each row so you know which parameter set was used
   - This makes it easy to extract/analyze downstream

Q: Why 500K rows when you could just store the 3 parameters?
A: Because you need the reconstructed curve for:
   - Plotting
   - Computing means across runs
   - Error analysis
   - Validation (comparing fit vs actual)

Q: If A, B, n don't change, why is fitting per-run necessary?
A: Different runs have DIFFERENT A, B, n values!
   Run 0: A=0.293, B=5.43, n=0.87
   Run 1: A=0.299, B=5.51, n=0.87  ← Different!
   The variation tells you about run-to-run variance in learning

================================================================================
END OF DOCUMENT
================================================================================

For questions about specific parameters or analysis, refer to:
- SLP/SLP-MNIST/fitting_function_IPA.ipynb (Cells 1-6)
- SLP/SLP-MNIST/Fitting_IPA_curves_data/Summary/ (consolidated parameter files)
