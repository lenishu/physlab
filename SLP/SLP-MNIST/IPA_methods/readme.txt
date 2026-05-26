Calculating IPA Using different methods of averaging and fitting 


Approach 1
At each BN, average the CE from each of the 100 raw data curves to get <CE> at each BN.
Then fit the single file of <CE> versus BN data points to get a single fitting function:

(avg file - ( eg averaged_runs_p_0.0_bs_64.csv)) (done)
 
Use the single fitting function to get a single value of CEasy (i.e. A in the fitting function).
Use this fitted CEasy to bet CEL (This is the only reason to use the fitting function).
Once CEL is determined by using the fitting function, obtain BNL at which CEL ocurs by using the data points
in the file containing <CE> (of raw data of 100 runs) versus BN; i.e. the first BN at which <CE> < CEL.


Approach 2. 

Start by fitting each run separately using the fitting function. 

Use the average of fitting curve (Per-run fitting → save per_run_fits_p_{p}_bs_{bs}.csv and mean_fit_p_{p}_bs_{bs}.csv ) 
to calculate IPA in two different ways. 

a. - Find the average curve of all the 100 fitting curves. Using this average curve, get the average CE_asymptote, then use that to calcualte IPA.
a1: To get the average fit curve, use the 100 different fit curves to calculate 100 CE at each BN (per_run_fits_p_{p}_bs_{bs}.csv). Average the 100 CE values to get <CE> at each BN (mean_fit_p_{p}_bs_{bs}.csv). Fit the <CE> versus BN to get a sigle fit curve. Note: If each fit curve matches the raw data well, these <CE> will match the <CE> closely taken from the raw data for each curve as in Approach 1.

a2: calculate the avergae of the fitting curves by separately calculating <A>, <B>, <n>. Note: this may not be valid because the fitting function is non-linear.


b - Find IPA for each fitting curve and then average the 100 IPA to get <IPA> for each P%. 


Output file:(.csv)
P%,IPA_Avg_64,STD_64,IPA_Avg_1024,STD_1024,IPA_Avg_60000,STD_60000    


