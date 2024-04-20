import numpy as np
import pandas as pd
import math
import os
import random
from random import shuffle 
from pandas import ExcelWriter
import statsmodels.formula.api as smf
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from psmpy import PsmPy

from baseline_utils import get_true_coef, get_basic_coef,get_iv_coef,get_centrality_coef,get_chen_coef,get_chen_c_coef,get_chen_n_coef,get_chen_n_c_coef
from baseline_utils import get_match_psm, get_match_psw

import warnings

# Ignore all FutureWarnings
warnings.filterwarnings('ignore')

dimensions = 64
num_data = 100

#--------------------------------------------------------------#
#    Get regression baseline results for pure homophily case   #
#--------------------------------------------------------------#

feature_folder = 'data/Final_simulated_data_pure_homophily/Final_regression_feature/'
inputfile = '_pure_homophily_beta0_final_data.csv'
output_folder = 'data/Final_simulated_data_pure_homophily/'

print('---------Generate results for homophily pure case!---------')

#-------Results in Panel (a) and Panel (b)---------#
model1_result = []
model2_result = []
model3_result = []
model4_result = []
model5_result = []
model6_result = []
model7_result = []
model8_result = []
for i in range(1,num_data+1):
   datafile = feature_folder + str(i) + inputfile
   result1  = get_true_coef(datafile,dimensions)
   result2  = get_basic_coef(datafile,dimensions)
   result3  = get_iv_coef(datafile,dimensions)
   result4  = get_centrality_coef(datafile,dimensions)
   result5  = get_chen_coef(datafile,dimensions)
   result6  = get_chen_c_coef(datafile,dimensions)
   result7  = get_chen_n_coef(datafile,dimensions)
   result8  = get_chen_n_c_coef(datafile,dimensions)

   model1_result.append(result1)
   model2_result.append(result2)
   model3_result.append(result3)
   model4_result.append(result4)
   model5_result.append(result5)
   model6_result.append(result6)
   model7_result.append(result7)
   model8_result.append(result8)


true_model = pd.DataFrame(model1_result, columns = ['beta_coef','se','pvalue'])
true_model['#pvalue<0.05'] = np.where(true_model['pvalue']<0.05,1,0)
true_model_result_mean = true_model.mean(axis=0)
true_model.loc['Avg_value'] = true_model_result_mean

basic_model = pd.DataFrame(model2_result, columns = ['beta_coef','se','pvalue'])
basic_model['#pvalue<0.05'] = np.where(basic_model['pvalue']<0.05,1,0)
basic_model_result_mean = basic_model.mean(axis=0)
basic_model.loc['Avg_value'] = basic_model_result_mean

iv_model = pd.DataFrame(model3_result, columns = ['beta_coef','se','pvalue'])
iv_model['#pvalue<0.05'] = np.where(iv_model['pvalue']<0.05,1,0)
iv_model_result_mean = iv_model.mean(axis=0)
iv_model.loc['Avg_value'] = iv_model_result_mean

centrality_model = pd.DataFrame(model4_result, columns = ['beta_coef','se','pvalue'])
centrality_model['#pvalue<0.05'] = np.where(centrality_model['pvalue']<0.05,1,0)
centrality_model_result_mean = centrality_model.mean(axis=0)
centrality_model.loc['Avg_value'] = centrality_model_result_mean

chen_model = pd.DataFrame(model5_result, columns = ['beta_coef','se','pvalue'])
chen_model['#pvalue<0.05'] = np.where(chen_model['pvalue']<0.05,1,0)
chen_model_result_mean = chen_model.mean(axis=0)
chen_model.loc['Avg_value'] = chen_model_result_mean

chen_centrality_model = pd.DataFrame(model6_result, columns = ['beta_coef','se','pvalue'])
chen_centrality_model['#pvalue<0.05'] = np.where(chen_centrality_model['pvalue']<0.05,1,0)
chen_centrality_model_result_mean = chen_centrality_model.mean(axis=0)
chen_centrality_model.loc['Avg_value'] = chen_centrality_model_result_mean

chen_neighbor_model = pd.DataFrame(model7_result, columns = ['beta_coef','se','pvalue'])
chen_neighbor_model['#pvalue<0.05'] = np.where(chen_neighbor_model['pvalue']<0.05,1,0)
chen_neighbor_model_result_mean = chen_neighbor_model.mean(axis=0)
chen_neighbor_model.loc['Avg_value'] = chen_neighbor_model_result_mean

chen_n_c_model = pd.DataFrame(model8_result, columns = ['beta_coef','se','pvalue'])
chen_n_c_model['#pvalue<0.05'] = np.where(chen_n_c_model['pvalue']<0.05,1,0)
chen_n_c_model_result_mean = chen_n_c_model.mean(axis=0)
chen_n_c_model.loc['Avg_value'] = chen_n_c_model_result_mean

writer = ExcelWriter(output_folder+'Table2(ab)_Baseline_results_pure_homophily.xlsx')
true_model.to_excel(writer,'True_model')
basic_model.to_excel(writer,'Basic_model')
iv_model.to_excel(writer,'IV_model')
centrality_model.to_excel(writer,'Centrality_model')
chen_model.to_excel(writer,'Chen')
chen_centrality_model.to_excel(writer,'Chen&C')
chen_neighbor_model.to_excel(writer,'Chen&N')
chen_n_c_model.to_excel(writer,'Chen&N&C')
writer.save()
print('---------Complete Panel (a) and (b)---------')

#-------Results in Panel (c): PSM---------#
psm1_result = []
psm2_result = []
psm3_result = []
psm4_result = []

for i in range(1,num_data+1):
    datafile = feature_folder + str(i) + inputfile
    # No unobservables
    control_value1 = ['y0','x1','x2']
    psm1 = get_match_psm(datafile,control_value1)
    psm1_result.append(psm1)
    # observable
    control_value2 = ['y0']
    psm2 = get_match_psm(datafile,control_value2)
    psm2_result.append(psm2)
    # observable + centrality
    control_value3 = ['y0'] + ['C'+str(i) for i in range(10)]
    psm3 = get_match_psm(datafile,control_value3)
    psm3_result.append(psm3)
    # observable + embedding
    control_value4 = ['y0'] + ['E'+str(i) for i in range(dimensions)]
    psm4 = get_match_psm(datafile,control_value4)
    psm4_result.append(psm4)

psm1_result = pd.DataFrame(psm1_result, columns = ['beta_coef','se','pvalue'])
psm1_result['#pvalue<0.05'] = np.where(psm1_result['pvalue']<0.05,1,0)
psm1_result_mean = psm1_result.mean(axis=0)
psm1_result.loc['Avg_value'] = psm1_result_mean

psm2_result = pd.DataFrame(psm2_result, columns = ['beta_coef','se','pvalue'])
psm2_result['#pvalue<0.05'] = np.where(psm2_result['pvalue']<0.05,1,0)
psm2_result_mean = psm2_result.mean(axis=0)
psm2_result.loc['Avg_value'] = psm2_result_mean

psm3_result = pd.DataFrame(psm3_result, columns = ['beta_coef','se','pvalue'])
psm3_result['#pvalue<0.05'] = np.where(psm3_result['pvalue']<0.05,1,0)
psm3_result_mean = psm3_result.mean(axis=0)
psm3_result.loc['Avg_value'] = psm3_result_mean

psm4_result = pd.DataFrame(psm4_result, columns = ['beta_coef','se','pvalue'])
psm4_result['#pvalue<0.05'] = np.where(psm4_result['pvalue']<0.05,1,0)
psm4_result_mean = psm4_result.mean(axis=0)
psm4_result.loc['Avg_value'] = psm4_result_mean

writer = ExcelWriter(output_folder+'Table2(c)_PSM_results_pure_homophily.xlsx')
psm1_result.to_excel(writer,'1 no unobservables')
psm2_result.to_excel(writer,'2 observables')
psm3_result.to_excel(writer,'3 observable+centrality')
psm4_result.to_excel(writer,'4 observable+embedding')
writer.save()
print('---------Complete Panel (c) PSMatching---------')

#-------Results in Panel (c): PSWeighting---------#
psw1_result = []
psw2_result = []
psw3_result = []
psw4_result = []

for i in range(1,num_data+1):
    datafile = feature_folder + str(i) + inputfile
    # No unobservables
    control_value1 = ['y0','x1','x2']
    psw1 = get_match_psw(datafile,control_value1)
    psw1_result.append(psw1)
    # observable
    control_value2 = ['y0']
    psw2 = get_match_psw(datafile,control_value2)
    psw2_result.append(psw2)
    # observable + centrality
    control_value3 = ['y0'] + ['C'+str(i) for i in range(10)]
    psw3 = get_match_psw(datafile,control_value3)
    psw3_result.append(psw3)
    # observable + embedding
    control_value4 = ['y0'] + ['E'+str(i) for i in range(dimensions)]
    psw4 = get_match_psw(datafile,control_value4)
    psw4_result.append(psw4)

psw1_result = pd.DataFrame(psw1_result, columns = ['beta_coef','se','pvalue'])
psw1_result['#pvalue<0.05'] = np.where(psw1_result['pvalue']<0.05,1,0)
psw1_result_mean = psw1_result.mean(axis=0)
psw1_result.loc['Avg_value'] = psw1_result_mean

psw2_result = pd.DataFrame(psw2_result, columns = ['beta_coef','se','pvalue'])
psw2_result['#pvalue<0.05'] = np.where(psw2_result['pvalue']<0.05,1,0)
psw2_result_mean = psw2_result.mean(axis=0)
psw2_result.loc['Avg_value'] = psw2_result_mean

psw3_result = pd.DataFrame(psw3_result, columns = ['beta_coef','se','pvalue'])
psw3_result['#pvalue<0.05'] = np.where(psw3_result['pvalue']<0.05,1,0)
psw3_result_mean = psw3_result.mean(axis=0)
psw3_result.loc['Avg_value'] = psw3_result_mean

psw4_result = pd.DataFrame(psw4_result, columns = ['beta_coef','se','pvalue'])
psw4_result['#pvalue<0.05'] = np.where(psw4_result['pvalue']<0.05,1,0)
psw4_result_mean = psw4_result.mean(axis=0)
psw4_result.loc['Avg_value'] = psw4_result_mean

writer = ExcelWriter(output_folder+'Table2(c)_PSWeighting_results_pure_homophily.xlsx')
psw1_result.to_excel(writer,'1 no unobservables')
psw2_result.to_excel(writer,'2 observables')
psw3_result.to_excel(writer,'3 observable+centrality')
psw4_result.to_excel(writer,'4 observable+embedding')
writer.save()
print('---------Complete Panel (c) PSWeighting---------')

#-------------------------------------------------------------#
#    Get regression baseline results for postive peer case    #
#-------------------------------------------------------------#

feature_folder = 'data/Final_simulated_data_positive_peer_effect/Final_regression_feature/'
inputfile = '_positve_peer_effect_beta0.2_final_data.csv'
output_folder = 'data/Final_simulated_data_positive_peer_effect/'

print('---------Generate results for positive peer effect case!---------')

#-------Results in Panel (a) and Panel (b)---------#
model1_result = []
model2_result = []
model3_result = []
model4_result = []
model5_result = []
model6_result = []
model7_result = []
model8_result = []
for i in range(1,num_data+1):
   datafile = feature_folder + str(i) + inputfile
   result1  = get_true_coef(datafile,dimensions)
   result2  = get_basic_coef(datafile,dimensions)
   result3  = get_iv_coef(datafile,dimensions)
   result4  = get_centrality_coef(datafile,dimensions)
   result5  = get_chen_coef(datafile,dimensions)
   result6  = get_chen_c_coef(datafile,dimensions)
   result7  = get_chen_n_coef(datafile,dimensions)
   result8  = get_chen_n_c_coef(datafile,dimensions)

   model1_result.append(result1)
   model2_result.append(result2)
   model3_result.append(result3)
   model4_result.append(result4)
   model5_result.append(result5)
   model6_result.append(result6)
   model7_result.append(result7)
   model8_result.append(result8)


true_model = pd.DataFrame(model1_result, columns = ['beta_coef','se','pvalue'])
true_model['#pvalue<0.05'] = np.where(true_model['pvalue']<0.05,1,0)
true_model_result_mean = true_model.mean(axis=0)
true_model.loc['Avg_value'] = true_model_result_mean

basic_model = pd.DataFrame(model2_result, columns = ['beta_coef','se','pvalue'])
basic_model['#pvalue<0.05'] = np.where(basic_model['pvalue']<0.05,1,0)
basic_model_result_mean = basic_model.mean(axis=0)
basic_model.loc['Avg_value'] = basic_model_result_mean

iv_model = pd.DataFrame(model3_result, columns = ['beta_coef','se','pvalue'])
iv_model['#pvalue<0.05'] = np.where(iv_model['pvalue']<0.05,1,0)
iv_model_result_mean = iv_model.mean(axis=0)
iv_model.loc['Avg_value'] = iv_model_result_mean

centrality_model = pd.DataFrame(model4_result, columns = ['beta_coef','se','pvalue'])
centrality_model['#pvalue<0.05'] = np.where(centrality_model['pvalue']<0.05,1,0)
centrality_model_result_mean = centrality_model.mean(axis=0)
centrality_model.loc['Avg_value'] = centrality_model_result_mean

chen_model = pd.DataFrame(model5_result, columns = ['beta_coef','se','pvalue'])
chen_model['#pvalue<0.05'] = np.where(chen_model['pvalue']<0.05,1,0)
chen_model_result_mean = chen_model.mean(axis=0)
chen_model.loc['Avg_value'] = chen_model_result_mean

chen_centrality_model = pd.DataFrame(model6_result, columns = ['beta_coef','se','pvalue'])
chen_centrality_model['#pvalue<0.05'] = np.where(chen_centrality_model['pvalue']<0.05,1,0)
chen_centrality_model_result_mean = chen_centrality_model.mean(axis=0)
chen_centrality_model.loc['Avg_value'] = chen_centrality_model_result_mean

chen_neighbor_model = pd.DataFrame(model7_result, columns = ['beta_coef','se','pvalue'])
chen_neighbor_model['#pvalue<0.05'] = np.where(chen_neighbor_model['pvalue']<0.05,1,0)
chen_neighbor_model_result_mean = chen_neighbor_model.mean(axis=0)
chen_neighbor_model.loc['Avg_value'] = chen_neighbor_model_result_mean

chen_n_c_model = pd.DataFrame(model8_result, columns = ['beta_coef','se','pvalue'])
chen_n_c_model['#pvalue<0.05'] = np.where(chen_n_c_model['pvalue']<0.05,1,0)
chen_n_c_model_result_mean = chen_n_c_model.mean(axis=0)
chen_n_c_model.loc['Avg_value'] = chen_n_c_model_result_mean

writer = ExcelWriter(output_folder+'Table2(ab)_Baseline_results_positive_peer_effect.xlsx')
true_model.to_excel(writer,'True_model')
basic_model.to_excel(writer,'Basic_model')
iv_model.to_excel(writer,'IV_model')
centrality_model.to_excel(writer,'Centrality_model')
chen_model.to_excel(writer,'Chen')
chen_centrality_model.to_excel(writer,'Chen&C')
chen_neighbor_model.to_excel(writer,'Chen&N')
chen_n_c_model.to_excel(writer,'Chen&N&C')
writer.save()
print('---------Complete Panel (a) and (b)---------')

#-------Results in Panel (c): PSM---------#
psm1_result = []
psm2_result = []
psm3_result = []
psm4_result = []

for i in range(1,num_data+1):
    datafile = feature_folder + str(i) + inputfile
    # No unobservables
    control_value1 = ['y0','x1','x2']
    psm1 = get_match_psm(datafile,control_value1)
    psm1_result.append(psm1)
    # observable
    control_value2 = ['y0']
    psm2 = get_match_psm(datafile,control_value2)
    psm2_result.append(psm2)
    # observable + centrality
    control_value3 = ['y0'] + ['C'+str(i) for i in range(10)]
    psm3 = get_match_psm(datafile,control_value3)
    psm3_result.append(psm3)
    # observable + embedding
    control_value4 = ['y0'] + ['E'+str(i) for i in range(dimensions)]
    psm4 = get_match_psm(datafile,control_value4)
    psm4_result.append(psm4)

psm1_result = pd.DataFrame(psm1_result, columns = ['beta_coef','se','pvalue'])
psm1_result['#pvalue<0.05'] = np.where(psm1_result['pvalue']<0.05,1,0)
psm1_result_mean = psm1_result.mean(axis=0)
psm1_result.loc['Avg_value'] = psm1_result_mean

psm2_result = pd.DataFrame(psm2_result, columns = ['beta_coef','se','pvalue'])
psm2_result['#pvalue<0.05'] = np.where(psm2_result['pvalue']<0.05,1,0)
psm2_result_mean = psm2_result.mean(axis=0)
psm2_result.loc['Avg_value'] = psm2_result_mean

psm3_result = pd.DataFrame(psm3_result, columns = ['beta_coef','se','pvalue'])
psm3_result['#pvalue<0.05'] = np.where(psm3_result['pvalue']<0.05,1,0)
psm3_result_mean = psm3_result.mean(axis=0)
psm3_result.loc['Avg_value'] = psm3_result_mean

psm4_result = pd.DataFrame(psm4_result, columns = ['beta_coef','se','pvalue'])
psm4_result['#pvalue<0.05'] = np.where(psm4_result['pvalue']<0.05,1,0)
psm4_result_mean = psm4_result.mean(axis=0)
psm4_result.loc['Avg_value'] = psm4_result_mean

writer = ExcelWriter(output_folder+'Table2(c)_PSM_results_positive_peer_effect.xlsx')
psm1_result.to_excel(writer,'1 no unobservables')
psm2_result.to_excel(writer,'2 observables')
psm3_result.to_excel(writer,'3 observable+centrality')
psm4_result.to_excel(writer,'4 observable+embedding')
writer.save()
print('---------Complete Panel (c) PSMatching---------')

#-------Results in Panel (c): PSWeighting---------#
psw1_result = []
psw2_result = []
psw3_result = []
psw4_result = []

for i in range(1,num_data+1):
    datafile = feature_folder + str(i) + inputfile
    # No unobservables
    control_value1 = ['y0','x1','x2']
    psw1 = get_match_psw(datafile,control_value1)
    psw1_result.append(psw1)
    # observable
    control_value2 = ['y0']
    psw2 = get_match_psw(datafile,control_value2)
    psw2_result.append(psw2)
    # observable + centrality
    control_value3 = ['y0'] + ['C'+str(i) for i in range(10)]
    psw3 = get_match_psw(datafile,control_value3)
    psw3_result.append(psw3)
    # observable + embedding
    control_value4 = ['y0'] + ['E'+str(i) for i in range(dimensions)]
    psw4 = get_match_psw(datafile,control_value4)
    psw4_result.append(psw4)

psw1_result = pd.DataFrame(psw1_result, columns = ['beta_coef','se','pvalue'])
psw1_result['#pvalue<0.05'] = np.where(psw1_result['pvalue']<0.05,1,0)
psw1_result_mean = psw1_result.mean(axis=0)
psw1_result.loc['Avg_value'] = psw1_result_mean

psw2_result = pd.DataFrame(psw2_result, columns = ['beta_coef','se','pvalue'])
psw2_result['#pvalue<0.05'] = np.where(psw2_result['pvalue']<0.05,1,0)
psw2_result_mean = psw2_result.mean(axis=0)
psw2_result.loc['Avg_value'] = psw2_result_mean

psw3_result = pd.DataFrame(psw3_result, columns = ['beta_coef','se','pvalue'])
psw3_result['#pvalue<0.05'] = np.where(psw3_result['pvalue']<0.05,1,0)
psw3_result_mean = psw3_result.mean(axis=0)
psw3_result.loc['Avg_value'] = psw3_result_mean

psw4_result = pd.DataFrame(psw4_result, columns = ['beta_coef','se','pvalue'])
psw4_result['#pvalue<0.05'] = np.where(psw4_result['pvalue']<0.05,1,0)
psw4_result_mean = psw4_result.mean(axis=0)
psw4_result.loc['Avg_value'] = psw4_result_mean

writer = ExcelWriter(output_folder+'Table3(c)_PSWeighting_results_positive_peer_effect.xlsx')
psw1_result.to_excel(writer,'1 no unobservables')
psw2_result.to_excel(writer,'2 observables')
psw3_result.to_excel(writer,'3 observable+centrality')
psw4_result.to_excel(writer,'4 observable+embedding')
writer.save()
print('---------Complete Panel (c) PSWeighting---------')

