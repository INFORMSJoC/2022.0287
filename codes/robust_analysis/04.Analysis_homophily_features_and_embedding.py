import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from pandas import ExcelWriter
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi
from scipy import stats

def get_variable(datafile,dimensions):
    data = pd.read_csv(datafile)
    emb_list = ['E'+str(i) for i in range(dimensions)]
    emb = data[emb_list].values
    x = data[['x1','x2']].values
    return x,emb

def get_corr(x,emb):
    corr_result_emd = []
    for i in range(len(x[0].T)):
        new_corr = []
        new_corr2 = []
        for j in range(len(emb[0].T)):
            corr_value = np.corrcoef(x[:,i],emb[:,j])
            new_corr.append(float(corr_value[0,1]))
            new_corr2.append(abs(float(corr_value[0,1])))
        corr_result_emd.append(new_corr)
        corr_result_emd.append(new_corr2)
    corr_result = pd.DataFrame(corr_result_emd).T
    corr_result.columns = ['cor_x1','abs_cor_x1','cor_x2','abs_cor_x2']
    corr_result_max = corr_result.max(axis=0)
    corr_result_min = corr_result.min(axis=0)
    corr_result_mean = corr_result.mean(axis=0)
    corr_result_std = corr_result.std(axis=0)
    corr_result['max'] = corr_result_max
    corr_result['min'] = corr_result_min
    corr_result['avg'] = corr_result_mean
    corr_result['std'] = corr_result_std


    x1_max_abs = corr_result_max[1]
    x1_min_abs = corr_result_min[1]
    x1_avg_abs = corr_result_mean[1]
    x1_std_abs = corr_result_std[1]

    x2_max_abs = corr_result_max[3]
    x2_min_abs = corr_result_min[3]
    x2_avg_abs = corr_result_mean[3]
    x2_std_abs = corr_result_std[3]

    return [x1_max_abs,x1_min_abs,x1_avg_abs,x1_std_abs],[x2_max_abs,x2_min_abs,x2_avg_abs,x2_std_abs]    

def get_mi(x,emb):
    mic_result_emd = []
    for i in range(len(x[0,:].T)):
        new_mi = []
        data1 = x[:,i]
        for j in range(len(emb[0].T)):
            data2 = emb[:,j].reshape(-1,1)
            
            result1 = mutual_info_regression(data2,data1)
            new_mi.append(result1[0])
            
        mic_result_emd.append(new_mi)
    mic_result = pd.DataFrame(mic_result_emd).T
    mic_result.columns = ['mi_x1','mi_x2']
    mic_result_max = mic_result.max(axis=0)
    mic_result_min = mic_result.min(axis=0)
    mic_result_mean = mic_result.mean(axis=0)
    mic_result_std = mic_result.std(axis=0)

    x1_max_abs = mic_result_max[0]
    x1_min_abs = mic_result_min[0]
    x1_avg_abs = mic_result_mean[0]
    x1_std_abs = mic_result_std[0]

    x2_max_abs = mic_result_max[1]
    x2_min_abs = mic_result_min[1]
    x2_avg_abs = mic_result_mean[1]
    x2_std_abs = mic_result_std[1]

    return [x1_max_abs,x1_min_abs,x1_avg_abs,x1_std_abs],[x2_max_abs,x2_min_abs,x2_avg_abs,x2_std_abs]    

def get_poly_one_var_result(poly,dv,reg_base):
    best_poly = []
    best_fstat = 0
    for j in range(len(emb[0].T)):
        v1 = emb[:,j].reshape(-1,1)
        X_train_poly = poly.fit_transform(v1)
        remain = emb[:, np.setdiff1d(np.arange(emb.shape[1]), j)]
        emb_all2 = np.hstack((X_train_poly, remain))

        reg_poly = sm.OLS(dv, emb_all2).fit()
        R2_poly = reg_poly.rsquared

        # F-test
        f_test_poly = reg_poly.compare_f_test(reg_base)
        if f_test_poly[0]> best_fstat:
            best_fstat = f_test_poly[0]
            sig = 1 if f_test_poly[1]<0.01 else 0

            best_poly = [R2_poly,f_test_poly[0],f_test_poly[1],sig] # R2, f_statistic, p_value, #p_value<0.01
    return best_poly

def get_poly_two_var_result(poly,dv,reg_base):
    best_poly = []
    best_fstat = 0
    for j in range(0,len(emb[0].T)-1):
        for k in range(j+1,len(emb[0].T)):
            v1 = emb[:,[j,k]]
            X_train_poly = poly.fit_transform(v1)
            remain = emb[:, np.setdiff1d(np.arange(emb.shape[1]), [j,k])]
            emb_all2 = np.hstack((X_train_poly, remain))

            reg_poly_v2 = sm.OLS(dv, emb_all2).fit()
            R2_poly_v2 = reg_poly_v2.rsquared     
            # Perform the F-test
            f_test_poly_v2 = reg_poly_v2.compare_f_test(reg_base)
            if f_test_poly_v2[0]> best_fstat:
                best_fstat = f_test_poly_v2[0]
                sig = 1 if f_test_poly_v2[1]<0.01 else 0

                best_poly = [R2_poly_v2,f_test_poly_v2[0],f_test_poly_v2[1],sig] # R2, f_statistic, p_value, #p_value<0.01

    return best_poly

num_data = 100
dimensions = 64

feature_folder = 'data/Final_simulated_data_positive_peer_effect/Final_regression_feature/'
inputfile = '_positve_peer_effect_beta0.2_final_data.csv'
output_folder = 'data/Final_simulated_data_positive_peer_effect/'

#------------------------------------------------------------------------------------------------------#
#   Section 1. Get pairwise relationship between latent homophily feature and invidual emb dimension   #
#            Calculate Pearson correlation and Mutual Information                                      #
#------------------------------------------------------------------------------------------------------#
x1_result = []
x2_result = []
for i in range(1,101): 
    print('Dataset:', str(i))
    datafile = feature_folder + str(i) + inputfile
    x,emb = get_variable(datafile,dimensions)
    #--------- Get the overall relationship for a single dataset ---------#
    x1_corr, x2_corr = get_corr(x,emb)
    x1_mi, x2_mi = get_mi(x,emb)
    x1_result.append(x1_corr + x1_mi)
    x2_result.append(x2_corr + x2_mi)


x1_result = pd.DataFrame(x1_result)
x1_result_max = x1_result.max(axis=0)
x1_result_min = x1_result.min(axis=0)
x1_result_mean = x1_result.mean(axis=0)

# Get the maximum among all max scores, the minimum among all min scores, and the average among all average scores
x1_result.loc['Final'] = [x1_result_max[0],x1_result_min[1],x1_result_mean[2],x1_result_mean[3],x1_result_max[4],x1_result_min[5],x1_result_mean[6],x1_result_mean[7]]
x1_result.columns = ['Corr_Max(abs)','Corr_Min(abs)','Corr_Avg(abs)','Corr_SD','MI_Max(abs)','MI_Min(abs)','MI_Avg(abs)','MI_SD']

x2_result = pd.DataFrame(x2_result)
x2_result_max = x2_result.max(axis=0)
x2_result_min = x2_result.min(axis=0)
x2_result_mean = x2_result.mean(axis=0)
# Get the maximun among all max scores, the minimun among all min scores, and the average among all average scores
x2_result.loc['Final'] = [x2_result_max[0],x2_result_min[1],x2_result_mean[2],x2_result_mean[3],x2_result_max[4],x2_result_min[5],x2_result_mean[6],x2_result_mean[7]]
x2_result.columns = ['Corr_Max(abs)','Corr_Min(abs)','Corr_Avg(abs)','Corr_SD','MI_Max(abs)','MI_Min(abs)','MI_Avg(abs)','MI_SD']

outputpath =output_folder+'TableA2_Pairwise relationship between latent homophily features and embedding.xlsx'

writer = ExcelWriter(outputpath)
x1_result.to_excel(writer,'x1_emb_pairwise_result')
x2_result.to_excel(writer,'x2_emb_pairwise_result')
writer.save()
print("#--------- Complete Table A2 results! ---------#")

 
#--------------------------------------------------------------------------------------------#
#   Section 2. Pairwise relationship: illustrate 4 Patterns in Figure A2
#   Use dataset "2_positve_peer_effect_beta0.2_final_data.csv" to illustrate our findings    #
#--------------------------------------------------------------------------------------------#
case_datafile = feature_folder + str(2) + inputfile
data = pd.read_csv(case_datafile)

#-----------------(a) Linear correction-----------------#
x_feature = data['x2'].values  # x2
Emb26 = data['E26'].values

fig1 = plt.figure(figsize=(6,6))
ax1 = fig1.add_subplot(1,1,1)
mi_score1 = mutual_info_regression(Emb26.reshape(-1,1),x_feature)[0]

ax1.set_title(f'$r$ = {np.corrcoef(x_feature, Emb26)[0][1]:.2f}, $I(X,Emb)$ = {mi_score1:.2f}',fontsize=24)
plt.xlabel('X',fontsize=22)
plt.ylabel('Emb (26)',fontsize=22)
sValue = x_feature*10
ax1.scatter(x_feature,Emb26,c='r',marker='o')
plt.savefig('FigureA2(a).png')

#-----------------(b) No correction-----------------#
x_feature = data['x2'].values  # x2
Emb3 = data['E3'].values

fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(1,1,1)
mi_score2 = mutual_info_regression(Emb3.reshape(-1,1),x_feature)[0]

ax2.set_title(f'$r$ = {np.corrcoef(x_feature, Emb3)[0][1]:.2f}, $I(X,Emb)$ = {mi_score2:.2f}',fontsize=24)
plt.xlabel('X',fontsize=22)
plt.ylabel('Emb (3)',fontsize=22)
sValue = x_feature*10
ax2.scatter(x_feature,Emb3,c='r',marker='o')
plt.savefig('FigureA2(b).png')

#-----------------(c) Spurious correlation-----------------#
x_feature = data['x2'].values  # x2
Emb60 = data['E60'].values

fig3 = plt.figure(figsize=(6,6))
ax3 = fig3.add_subplot(1,1,1)
mi_score3 = mutual_info_regression(Emb60.reshape(-1,1),x_feature)[0]

ax3.set_title(f'$r$ = {np.corrcoef(x_feature, Emb60)[0][1]:.2f}, $I(X,Emb)$ = {mi_score3:.2f}',fontsize=24)
plt.xlabel('X',fontsize=22)
plt.ylabel('Emb (60)',fontsize=22)
sValue = x_feature*10
ax3.scatter(x_feature,Emb60,c='r',marker='o')
plt.savefig('FigureA2(c).png')

#-----------------(d) Nonlinear correlation-----------------#
x_feature = data['x2'].values  # x2
Emb53 = data['E53'].values

fig4 = plt.figure(figsize=(6,6))
ax4 = fig4.add_subplot(1,1,1)
mi_score4 = mutual_info_regression(Emb53.reshape(-1,1),x_feature)[0]

ax4.set_title(f'$r$ = {np.corrcoef(x_feature, Emb53)[0][1]:.2f}, $I(X,Emb)$ = {mi_score4:.2f}',fontsize=24)
plt.xlabel('X',fontsize=22)
plt.ylabel('Emb (60)',fontsize=22)
sValue = x_feature*10
ax4.scatter(x_feature,Emb53,c='r',marker='o')
plt.savefig('FigureA2(d).png')


#--------------------------------------------------------------------------------------------#
#   Section 3. Regression analysis - Explanatory power of entire embedding vector 
#                                                                                            #
#--------------------------------------------------------------------------------------------#

# polynomial degree 2
poly2 = PolynomialFeatures(2)
# polynomial degree 3
poly3 = PolynomialFeatures(3)


x1_result = []
x2_result = []

for i in range(1,101): 
    print('Dataset:', str(i))
    datafile = feature_folder + str(i) + inputfile
    x,emb = get_variable(datafile,dimensions)

    data1 = x[:,0]
    emb_all = sm.add_constant(emb)
    reg_base= sm.OLS(data1, emb_all).fit()
    R2_base = reg_base.rsquared
    #-------------include Degree-2 polynomial in one variable-------------#
    best_poly2_v1 = get_poly_one_var_result(poly2,data1,reg_base)

    #-------------include Degree-3 polynomial in one variable-------------#
    best_poly3_v1 = get_poly_one_var_result(poly3,data1,reg_base)

    #-------------include Degree-2 polynomial in two variable-------------#
    best_poly2_v2 = get_poly_two_var_result(poly2,data1,reg_base)
    
    #-------------include Degree-3 polynomial in two variable-------------#
    best_poly3_v2 = get_poly_two_var_result(poly3,data1,reg_base)

    x1_result.append([R2_base] + best_poly2_v1 + best_poly3_v1 + best_poly2_v2 + best_poly3_v2)


    data2 = x[:,1]
    emb_all = sm.add_constant(emb)
    reg_base= sm.OLS(data2, emb_all).fit()
    R2_base = reg_base.rsquared
    #-------------include Degree-2 polynomial in one variable-------------#
    best_poly2_v1 = get_poly_one_var_result(poly2,data2,reg_base)

    #-------------include Degree-3 polynomial in one variable-------------#
    best_poly3_v1 = get_poly_one_var_result(poly3,data2,reg_base)

    #-------------include Degree-2 polynomial in two variable-------------#
    best_poly2_v2 = get_poly_two_var_result(poly2,data2,reg_base)
    
    #-------------include Degree-3 polynomial in two variable-------------#
    best_poly3_v2 = get_poly_two_var_result(poly3,data2,reg_base)

    x2_result.append([R2_base] + best_poly2_v1 + best_poly3_v1 + best_poly2_v2 + best_poly3_v2)


column_name = ['r2_linear','r2_poly2v1','poly2v1_f_stat','poly2v1_pvalue','poly2v1_sig0.01',
                        'r2_poly3v1','poly3v1_f_stat','poly3v1_pvalue','poly3v1_sig0.01',
                        'r2_poly2v2','poly2v2_f_stat','poly2v2_pvalue','poly2v2_sig0.01',
                        'r2_poly3v2','poly3v2_f_stat','poly3v2_pvalue','poly3v2_sig0.01']
reg_x1_result = pd.DataFrame(x1_result)
reg_x1_result_mean = reg_x1_result.mean(axis=0)
reg_x1_result.loc['Final'] = reg_x1_result_mean
reg_x1_result.columns = column_name

reg_x2_result = pd.DataFrame(x2_result)
reg_x2_result_mean = reg_x2_result.mean(axis=0)
reg_x2_result.loc['Final'] = reg_x2_result_mean
reg_x2_result.columns = column_name

outputpath = output_folder+'TableA3_Regression analysis of latent homophily features and embedding.xlsx'
writer = ExcelWriter(outputpath)
reg_x1_result.to_excel(writer,'x1_reg_result')
reg_x2_result.to_excel(writer,'x2_reg_result')
writer.save()
