SLP-FMNIST version of SLP-MNIST/IPA_2nd_derivative_method (same notebooks, same settings).
IPA from the max-second-derivative elbow, two ways of averaging the 100 runs.
Both use the same elbow rule (second_derivative_elbow, SMOOTH_W = 30) and
IPA = |CE_o - CE_learned| / BN_learned with CE_o = ln(10). No curve fitting.
Data: ../prune_layers_ALL (paths are relative, so the notebooks run on any machine).
FMNIST has 11 pruning levels (P = 0, 10, ..., 100%), 100 runs each, BS = 64, 1024, 60000.


CE_Avg_method/ipa_2nd_derivative_standalone.ipynb
At each BN, average the CE from each of the 100 raw data curves to get <CE> at each BN
(averaged_runs_p_{p}_bs_{bs}.csv), cut at the step artifact, find the elbow of that
single <CE> vs BN curve, and get one IPA per (P%, BS).
Output: ipa_summary_2nd_deriv_standalone.csv, ipa_plot_2nd_deriv_standalone.png


IPA_Avg_method/ipa_2nd_derivative_ipa_avg.ipynb
Find the elbow of each raw run (slp_{p}_{bs}_run_*.txt) separately, get one IPA per
run, then average the IPA over the runs: <IPA> and STD (sample std over runs).
No step-artifact cut by default (STEP_CUT = False): the step comes from runs dropping
out of the average, and a single run has no such step.
Empty or flat runs are skipped (P=100% is flat, so all 300 of its runs give NaN).
Output: ipa_per_run_2nd_deriv.csv (one row per run, 3300 rows),
        ipa_summary_2nd_deriv_ipa_avg.csv (P%, IPA_Avg_{bs}, STD_{bs}, n_valid_{bs}, ...),
        ipa_plot_2nd_deriv_ipa_avg.png


Comparison/compare_ce_avg_vs_ipa_avg.ipynb
Reads the two summary CSVs above (run both method notebooks first) and compares them.
Output: ipa_compare_ce_avg_vs_ipa_avg.csv (per P% and BS: both IPAs, both BN_learned,
        Diff = IPA_Avg - CE_Avg, RelDiff_pct, Diff_over_STD),
        ipa_compare_methods.png (IPA vs P%, one panel per BS, both methods overlaid),
        ipa_compare_differences.png (relative difference and BN_learned vs P%)


Note on BN_learned: with SMOOTH_W = 30 the elbow is at BN = 17 for almost every curve,
in both datasets. The argmax search starts at BN = 16 (endpoint + w//2 edge points
excluded), and the max of the 2nd derivative is always the next point, BN = w//2 + 2
(w = 10, 20, 40 give 7, 12, 22). So BN_learned is set by SMOOTH_W, not by the data,
and IPA here is in effect (ln 10 - CE(BN=17)) / 17.
