*ssc install bcoeff

global inputfolder "/Users/xuanqi/Research/Study1_network project/IJOC_Final_Data_Code/Final_github/src/Final_simulated_data_positive_peer_effect"
cd "$inputfolder"

************ 1.Centrality Model ************

*Create an empty matrix to store coefficients
mat coef_mat = J(100, 1, .)
mat se_mat = J(100, 1, .)
mat pvalue_mat = J(100, 1, .)
mat sig_mat = J(100, 1, .)

*CODE to loop over 100 Datasets
forvalues i=1/100{
import delimited "Final_regression_feature/`i'_positve_peer_effect_beta0.2_final_data.csv", clear
*import delimited "Final_regression_feature/`i'_pure_homophily_beta0_final_data.csv", clear

* stepwise
qui stepwise, pe(.05)lockterm1: reg y1 (influ y0) c0-c9
return list
mat result5 = r(table)

mat coef_mat[`i', 1] = result5[1,1]
mat se_mat[`i', 1] = result5[2,1]
mat pvalue_mat[`i', 1] = result5[4,1]
mat sig_mat[`i', 1] = (result5[4,1]<0.05)
clear
}
set obs 101
gen dataset = _n
svmat coef_mat, names(beta_coef)
svmat se_mat, names(sevalue)
svmat pvalue_mat, names(pvalue)
svmat sig_mat, names(sig)

egen coef_mean = mean(beta_coef)
egen se_mean = mean(sevalue)
egen p_mean = mean(pvalue)
egen s_mean = mean(sig)
replace beta_coef = coef_mean in 101
replace sevalue = se_mean in 101
replace pvalue = p_mean in 101
replace sig = s_mean in 101
keep dataset beta_coef sevalue pvalue sig

export excel "regression_stepwise_results_centrality.xlsx", replace firstrow(variables) sheet("Centrality")
exit
clear

************ 2. Chen et al. (2022) ************

*Create an empty matrix to store coefficients
mat coef_mat = J(100, 1, .)
mat se_mat = J(100, 1, .)
mat pvalue_mat = J(100, 1, .)
mat sig_mat = J(100, 1, .)

*CODE to loop over 100 Datasets
forvalues i=1/100{
import delimited "Final_regression_feature/`i'_positve_peer_effect_beta0.2_final_data.csv", clear
*import delimited "Final_regression_feature/`i'_pure_homophily_beta0_final_data.csv", clear
* stepwise
qui stepwise, pe(.05)lockterm1: reg y1 (influ y0) e0-e63
return list
mat result5 = r(table)

mat coef_mat[`i', 1] = result5[1,1]
mat se_mat[`i', 1] = result5[2,1]
mat pvalue_mat[`i', 1] = result5[4,1]
mat sig_mat[`i', 1] = (result5[4,1]<0.05)
clear
}
set obs 101
gen dataset = _n
svmat coef_mat, names(beta_coef)
svmat se_mat, names(sevalue)
svmat pvalue_mat, names(pvalue)
svmat sig_mat, names(sig)

egen coef_mean = mean(beta_coef)
egen se_mean = mean(sevalue)
egen p_mean = mean(pvalue)
egen s_mean = mean(sig)
replace beta_coef = coef_mean in 101
replace sevalue = se_mean in 101
replace pvalue = p_mean in 101
replace sig = s_mean in 101
keep dataset beta_coef sevalue pvalue sig

export excel using "regression_stepwise_results_chen.xlsx", replace firstrow(variables) sheet("Chen 2022")
exit
clear

************ 3. Chen et al. (2022) + Centrality************

*Create an empty matrix to store coefficients
mat coef_mat = J(100, 1, .)
mat se_mat = J(100, 1, .)
mat pvalue_mat = J(100, 1, .)
mat sig_mat = J(100, 1, .)

*CODE to loop over 100 Datasets
forvalues i=1/100{
import delimited "Final_regression_feature/`i'_positve_peer_effect_beta0.2_final_data.csv", clear
*import delimited "Final_regression_feature/`i'_pure_homophily_beta0_final_data.csv", clear
* stepwise
qui stepwise, pe(.05)lockterm1: reg y1 (influ y0) e0-e63 c0-c9
return list
mat result5 = r(table)

mat coef_mat[`i', 1] = result5[1,1]
mat se_mat[`i', 1] = result5[2,1]
mat pvalue_mat[`i', 1] = result5[4,1]
mat sig_mat[`i', 1] = (result5[4,1]<0.05)
clear
}
set obs 101
gen dataset = _n
svmat coef_mat, names(beta_coef)
svmat se_mat, names(sevalue)
svmat pvalue_mat, names(pvalue)
svmat sig_mat, names(sig)

egen coef_mean = mean(beta_coef)
egen se_mean = mean(sevalue)
egen p_mean = mean(pvalue)
egen s_mean = mean(sig)
replace beta_coef = coef_mean in 101
replace sevalue = se_mean in 101
replace pvalue = p_mean in 101
replace sig = s_mean in 101
keep dataset beta_coef sevalue pvalue sig

export excel using "regression_stepwise_results_chen_c.xlsx", replace firstrow(variables) sheet("Chen + C")
clear


************ 4. Chen et al. (2022) + Neighbor Embedding************

*Create an empty matrix to store coefficients
mat coef_mat = J(100, 1, .)
mat se_mat = J(100, 1, .)
mat pvalue_mat = J(100, 1, .)
mat sig_mat = J(100, 1, .)

*CODE to loop over 100 Datasets
forvalues i=1/100{
import delimited "Final_regression_feature/`i'_positve_peer_effect_beta0.2_final_data.csv", clear
*import delimited "Final_regression_feature/`i'_pure_homophily_beta0_final_data.csv", clear
* stepwise
qui stepwise, pe(.05)lockterm1: reg y1 (influ y0) e0-e63 n0-n63
return list
mat result5 = r(table)

mat coef_mat[`i', 1] = result5[1,1]
mat se_mat[`i', 1] = result5[2,1]
mat pvalue_mat[`i', 1] = result5[4,1]
mat sig_mat[`i', 1] = (result5[4,1]<0.05)
clear
}
set obs 101
gen dataset = _n
svmat coef_mat, names(beta_coef)
svmat se_mat, names(sevalue)
svmat pvalue_mat, names(pvalue)
svmat sig_mat, names(sig)

egen coef_mean = mean(beta_coef)
egen se_mean = mean(sevalue)
egen p_mean = mean(pvalue)
egen s_mean = mean(sig)
replace beta_coef = coef_mean in 101
replace sevalue = se_mean in 101
replace pvalue = p_mean in 101
replace sig = s_mean in 101
keep dataset beta_coef sevalue pvalue sig

export excel using "regression_stepwise_results_chen_n.xlsx", replace firstrow(variables) sheet("Chen + N")
clear


************ 5. Chen et al. (2022) + Neighbor Embedding + Centrality************

*Create an empty matrix to store coefficients
mat coef_mat = J(100, 1, .)
mat se_mat = J(100, 1, .)
mat pvalue_mat = J(100, 1, .)
mat sig_mat = J(100, 1, .)

*CODE to loop over 100 Datasets
forvalues i=1/100{
import delimited "Final_regression_feature/`i'_positve_peer_effect_beta0.2_final_data.csv", clear
*import delimited "Final_regression_feature/`i'_pure_homophily_beta0_final_data.csv", clear
* stepwise
qui stepwise, pe(.05)lockterm1: reg y1 (influ y0) e0-e63 n0-n63 c0-c9
return list
mat result5 = r(table)

mat coef_mat[`i', 1] = result5[1,1]
mat se_mat[`i', 1] = result5[2,1]
mat pvalue_mat[`i', 1] = result5[4,1]
mat sig_mat[`i', 1] = (result5[4,1]<0.05)
clear
}
set obs 101
gen dataset = _n
svmat coef_mat, names(beta_coef)
svmat se_mat, names(sevalue)
svmat pvalue_mat, names(pvalue)
svmat sig_mat, names(sig)

egen coef_mean = mean(beta_coef)
egen se_mean = mean(sevalue)
egen p_mean = mean(pvalue)
egen s_mean = mean(sig)
replace beta_coef = coef_mean in 101
replace sevalue = se_mean in 101
replace pvalue = p_mean in 101
replace sig = s_mean in 101
keep dataset beta_coef sevalue pvalue sig

export excel using "regression_stepwise_results_chen_n_c.xlsx", replace firstrow(variables) sheet("Chen + N + C")
clear
