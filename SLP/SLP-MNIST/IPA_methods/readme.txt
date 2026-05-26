Calculating IPA Using different methods of averaging and fitting 


Approach 1
At each BN, average the CE from each of the 100 raw data curves to get <CE> at each BN.
Then fit the single file of <CE> versus BN data points to get a single fitting function:

(avg file - ( eg averaged_runs_p_0.0_bs_64.csv)) (done)
 
Use the single fitting function to get a single value of CEasy (e.g. A in the fitting function).
Use this fitted CEasy to get CEL (This is the only reason to use the fitting function).
Once CEL is determined by using CEasy from the fitting function, obtain BNL at which CEL ocurs by using the data points
in the file containing <CE> (of raw data of 100 runs) versus BN; i.e. the first BN at which <CE> < CEL.


Approach 2. 

Start by fitting each of the 100 raw data runs separately using the fitting function. 

Use the average of the 100 fitted curves (Per-run fitting → save per_run_fits_p_{p}_bs_{bs}.csv and mean_fit_p_{p}_bs_{bs}.csv ) 
to calculate IPA. Averaging of the 100 fitted curves can be performed in two different ways. 

a. - Find the average curve of all the 100 fitting curves. Using this average curve, get the average CE_asymptote, then use that to calcualte CEL. 
To calculate the average <CE> versus BN from the 100 fitted curves, at each BN use each fitted curve to calculate 100 CE (per_run_fits_p_{p}_bs_{bs}.csv). Average the 100 CE values to get <CE> at each BN (mean_fit_p_{p}_bs_{bs}.csv). Fit the <CE> versus BN to get a single fit curve. Determine CEasy from this single fit curve. Use this CEasy to get CEL. Use the CEL to get BNL by going 
back to the <CE> versus BN data points obtained by averaging the 100 raw data curves.
Note: If each fit curve matches the raw data well, these <CE> will closely match the <CE> curve gotten from the raw data for each 
curve as in Approach 1.

a2: calculate a single 'average' fitting curve from the 100 fitting curves by separately calculating <A>, <B>, <n>. Note: this may not
be valid because the fitting function is non-linear.


b - Find IPA for each of the 100 fitting curve separately and then average the 100 IPA to get <IPA> for each P%. First fit separately 
each 100 CE versus BN raw data curves. For each of the 100 fitted curves, get its specific CEasy. Use CEasy for each run to get CEL 
for each run. Use CEL for each run to get BNL for each run by finding the first BN at which CE < CEL. Though we prefer to get BNLfrom 
the raw data, in this situation this will create a problem because each individual run is noisy with ups and owns. Therefore, in this
situation, BNL for each run should be determined from the smooth, monotonic fitted function of CE versus BN not from the raw data 
points. This should be valid because the fitting function closely matches the raw data for BN after the elbow.




Output file:(.csv)
P%,IPA_Avg_64,STD_64,IPA_Avg_1024,STD_1024,IPA_Avg_60000,STD_60000    


