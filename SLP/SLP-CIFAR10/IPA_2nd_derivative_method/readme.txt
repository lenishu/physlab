SLP-CIFAR10 version of SLP-MNIST/IPA_2nd_derivative_method (same notebooks, same settings).
IPA from the max-second-derivative elbow, two ways of averaging the 100 runs.
Both use the same elbow rule (second_derivative_elbow, SMOOTH_W = 30) and
IPA = |CE_o - CE_learned| / BN_learned with CE_o = ln(10). No curve fitting.
Data: ../prune_layers_ALL (paths are relative, so the notebooks run on any machine).
CIFAR-10 has 11 pruning levels (P = 0, 10, ..., 100%), BS = 64 and 50000, 100 runs each
(P=60%, BS=64 has 99: run_0 is missing). The 1-run set in ../SLP-CIFAR10_91-99 is not used.
Changes from MNIST: BATCH_SIZES = [64, 50000], run files are slp__{p}_{bs}_run_*.txt
(double underscore), and the comparison plot has two panels.

Run order:
  1. ../02_average_and_plots.ipynb   writes averaged_runs_p_{p}_bs_{bs}.csv into every
                                      prune_layers_ALL/p-percentage_{p}/batch_size_{bs}/
                                      plus the CE_Train / CE_Test / Accuracy plots in SLP-CIFAR10/
  2. CE_Avg_method, IPA_Avg_method (either order)
  3. Comparison


CE_Avg_method/ipa_2nd_derivative_standalone.ipynb
At each BN, average the CE from each of the 100 raw data curves to get <CE> at each BN
(averaged_runs_p_{p}_bs_{bs}.csv), cut at the step artifact, find the elbow of that
single <CE> vs BN curve, and get one IPA per (P%, BS).
Output: ipa_summary_2nd_deriv_standalone.csv, ipa_plot_2nd_deriv_standalone.png


IPA_Avg_method/ipa_2nd_derivative_ipa_avg.ipynb
Find the elbow of each raw run (slp__{p}_{bs}_run_*.txt) separately, get one IPA per
run, then average the IPA over the runs: <IPA> and STD (sample std over runs).
No step-artifact cut by default (STEP_CUT = False).
Empty or flat runs are skipped (P=100% is flat, so all 200 of its runs give NaN).
Output: ipa_per_run_2nd_deriv.csv (one row per run, 2199 rows),
        ipa_summary_2nd_deriv_ipa_avg.csv (P%, IPA_Avg_{bs}, STD_{bs}, n_valid_{bs}, ...),
        ipa_plot_2nd_deriv_ipa_avg.png


Comparison/compare_ce_avg_vs_ipa_avg.ipynb
Reads the two summary CSVs above (run both method notebooks first) and compares them.
Output: ipa_compare_ce_avg_vs_ipa_avg.csv (per P% and BS: both IPAs, both BN_learned,
        Diff = IPA_Avg - CE_Avg, RelDiff_pct, Diff_over_STD),
        ipa_compare_methods.png (IPA vs P%, one panel per BS, both methods overlaid),
        ipa_compare_differences.png (relative difference and BN_learned vs P%)


How CIFAR-10 differs from MNIST / FMNIST (read before using these numbers)

1. Test CE goes ABOVE ln(10). The averaged test CE rises from ~2.4 to as much as 4.4
   during training while accuracy reaches ~25-30%. Late-training CE stays above ln(10)
   for P <= 40% (BS=64) and P <= 70% (BS=50000); at BS=64, P = 50-70% overshoot above
   ln(10) early and then settle below it. At the elbow, CE_learned > ln(10) for
   P <= 50% (BS=64) and P <= 60% (BS=50000), and the |.| in the IPA formula turns
   that increase into a positive IPA. Check the CE_learned columns: CE_learned > 2.303
   means the test CE got worse than chance, not better.

2. No early stopping. Every run has the same length (BS=64: BN 0-782, one epoch;
   BS=50000: BN 0-100, one full-batch step per epoch), and Num_Runs is constant,
   so the averaged curves have no run-dropout step. The CE_Avg step cut
   (|dCE| > 0.01 at BN >= 100) still fires: on the noisy BS=64 curves it cuts at
   BN 100-107 for P <= 80% (no cut at P = 90%). That cut also removes the spike at
   BN = 782 (the last, partial batch of 16 samples). Without the cut, the BS=64 elbow
   for P <= 70% sits at BN = 766, the last searchable point next to that spike.
   For BS=50000 the cut only drops BN = 100 and does not change any elbow.

3. The elbow is not pinned at BN = 17 here (unlike MNIST / FMNIST), except at P = 80-90%
   (and P = 70%, BS=50000), where the curve is a clean decay. Elsewhere the curve rises
   or is noisy, and the max of the 2nd derivative moves around: CE_Avg BN_learned = 17-77,
   per-run BN_learned = 17-766 for BS=64 (median ~120-300 for P <= 80%) and 17-84
   for BS=50000.
   This is why the two methods differ by 30-75% at most P (RelDiff_pct in the
   comparison CSV); at P=60%, BS=64 the <CE> curve crosses ln(10) near the elbow,
   so CE_Avg IPA is ~0 and the relative difference blows up (~1300%).
