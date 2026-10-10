SLP-CIFAR100 version of SLP-MNIST/IPA_2nd_derivative_method (same notebooks, same settings,
except CE_o). IPA from the max-second-derivative elbow, two ways of averaging the runs.
Both use the same elbow rule (second_derivative_elbow, SMOOTH_W = 30) and
IPA = |CE_o - CE_learned| / BN_learned with CE_o = ln(100) = 4.605 (100 classes). No curve fitting.
Data: ../prune_layers_ALL (paths are relative, so the notebooks run on any machine).
CIFAR-100 has 11 pruning levels (P = 0, 10, ..., 100%), BS = 64 and 50000,
and ONLY 1 RUN per (P%, BS) (run_0).
Changes from MNIST: CE_o = ln(100), BATCH_SIZES = [64, 50000], run files are
slp__{p}_{bs}_run_*.txt (double underscore), and the comparison plot has two panels.

Run order:
  1. ../02_average_and_plots.ipynb   writes averaged_runs_p_{p}_bs_{bs}.csv into every
                                      prune_layers_ALL/p-percentage_{p}/batch_size_{bs}/
                                      plus the CE_Train / CE_Test / Accuracy plots in SLP-CIFAR100/
  2. CE_Avg_method, IPA_Avg_method (either order)
  3. Comparison


CE_Avg_method/ipa_2nd_derivative_standalone.ipynb
Elbow of the averaged curve (averaged_runs_p_{p}_bs_{bs}.csv), cut at the step artifact,
one IPA per (P%, BS).
Output: ipa_summary_2nd_deriv_standalone.csv, ipa_plot_2nd_deriv_standalone.png


IPA_Avg_method/ipa_2nd_derivative_ipa_avg.ipynb
Elbow of each raw run (slp__{p}_{bs}_run_*.txt), one IPA per run, then <IPA> and STD.
No step-artifact cut by default (STEP_CUT = False). P=100% is flat -> NaN.
Output: ipa_per_run_2nd_deriv.csv (22 rows), ipa_summary_2nd_deriv_ipa_avg.csv,
        ipa_plot_2nd_deriv_ipa_avg.png


Comparison/compare_ce_avg_vs_ipa_avg.ipynb
Reads the two summary CSVs above (run both method notebooks first) and compares them.
Output: ipa_compare_ce_avg_vs_ipa_avg.csv, ipa_compare_methods.png, ipa_compare_differences.png


How CIFAR-100 differs (read before using these numbers)

1. One run per point. The "average" curve is the single run (Num_Runs = 1), so
   STD = NaN everywhere (n_valid = 1) and the two methods see the same curve. The only
   difference between them is the CE_Avg step cut:
   - BS=50000: the cut only drops BN = 100, so both methods give identical IPA.
   - BS=64: the single-run curve is noisy, so the cut fires at BN 100-102 for every P.
     CE_Avg searches BN 16-84; IPA_Avg searches the whole run and its elbow lands at
     BN 214-766 for P = 10, 20, 50-80% (68-99.9% lower IPA). They agree at P = 0, 30, 40, 90%.
   Rerun all notebooks when more runs are added; nothing needs to change in the code.

2. Test CE goes ABOVE ln(100). It rises from ~4.7 to as much as 9 (BS=64, P=0%) while
   accuracy stays at 7-16%. At the CE_Avg elbow, CE_learned > ln(100) for P <= 60%
   (BS=64) and P <= 40% (BS=50000). The |.| in the IPA formula turns that increase into
   a positive IPA, so check the CE_learned columns: CE_learned > 4.605 means the test
   CE got worse than chance, not better.

3. Elbows are not pinned at BN = 17 (unlike MNIST / FMNIST): CE_Avg BN_learned = 17-82.

4. The accuracy at BN = 0 is 9.02% in every run file. SLP-CIFAR100.py writes that
   value as a fixed string for the first row (line 35); it is not measured. Chance level
   for CIFAR-100 is 1%, so ignore the BN = 0 point of the accuracy plots.
