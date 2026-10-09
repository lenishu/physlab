IPA from the max-second-derivative elbow, two ways of averaging the 100 runs.
Both use the same elbow rule (second_derivative_elbow, SMOOTH_W = 30) and
IPA = |CE_o - CE_learned| / BN_learned with CE_o = ln(10). No curve fitting.
Data: ../prune_layers_ALL (paths are relative, so the notebooks run on any machine).


CE_Avg_method/ipa_2nd_derivative_standalone.ipynb
At each BN, average the CE from each of the 100 raw data curves to get <CE> at each BN
(averaged_runs_p_{p}_bs_{bs}.csv), cut at the step artifact, find the elbow of that
single <CE> vs BN curve, and get one IPA per (P%, BS).
Output: ipa_summary_2nd_deriv_standalone.csv, ipa_plot_2nd_deriv_standalone.png
(same numbers as ../result/ipa_summary_2nd_deriv_standalone.csv)


IPA_Avg_method/ipa_2nd_derivative_ipa_avg.ipynb
Find the elbow of each raw run (slp_{p}_{bs}_run_*.txt) separately, get one IPA per
run, then average the IPA over the runs: <IPA> and STD (sample std over runs).
No step-artifact cut by default (STEP_CUT = False): the step comes from runs dropping
out of the average, and a single run has no such step.
Empty or flat runs are skipped (P=100% is flat; 0.92/60000/run_62 is an empty file).
Output: ipa_per_run_2nd_deriv.csv (one row per run),
        ipa_summary_2nd_deriv_ipa_avg.csv (P%, IPA_Avg_{bs}, STD_{bs}, n_valid_{bs}, ...),
        ipa_plot_2nd_deriv_ipa_avg.png


Comparison/compare_ce_avg_vs_ipa_avg.ipynb
Reads the two summary CSVs above (run both method notebooks first) and compares them.
Output: ipa_compare_ce_avg_vs_ipa_avg.csv (per P% and BS: both IPAs, both BN_learned,
        Diff = IPA_Avg - CE_Avg, RelDiff_pct, Diff_over_STD),
        ipa_compare_methods.png (IPA vs P%, one panel per BS, both methods overlaid),
        ipa_compare_differences.png (relative difference and BN_learned vs P%)
