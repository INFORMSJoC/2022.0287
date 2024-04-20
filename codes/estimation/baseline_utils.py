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

def get_true_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values
    x1 =data[['x1']].values
    cosx1 = np.cos(x1)
    x2 =data[['x2']].values
    sinx2 = np.sin(x2)

    # True model: y1 = influ + y0 + cos(x1) + sin(x2)
    X_full_1 = np.hstack([influ,y0,cosx1,sinx2])
    X_full_1 = sm.add_constant(X_full_1)
    model1 = sm.OLS(y1, X_full_1).fit()
    result1 = [model1.params[1],model1.bse[1],model1.pvalues[1]] # coef, se, pvalue

    return result1

def get_basic_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values

    # Basic model: y1 = influ + y0 
    X_full_2 = np.hstack([influ,y0])
    X_full_2 = sm.add_constant(X_full_2)
    model2 = sm.OLS(y1, X_full_2).fit()
    result2 = [model2.params[1],model2.bse[1],model2.pvalues[1]] # coef, se, pvalue

    return result2   

def get_iv_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values
    iv_avg_nn = data['iv_avg_nn'].values

    # IV model 2SLS 
    dependent = y1
    exog = sm.add_constant(y0)
    endog = influ
    instruments = iv_avg_nn
    mod = IV2SLS(dependent, exog, endog, instruments)
    model3 = mod.fit(cov_type='unadjusted')
    result3 = [model3.params[2],model3.std_errors[2],model3.pvalues[2]] # coef, se, pvalue

    return result3

def get_centrality_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values
    centrality_list = ['C'+str(i) for i in range(10)]
    Z_centrality = data[centrality_list].values


    # Centrality model: y1 = influ + y0 + centrality
    X_full_4 = np.hstack([influ,y0,Z_centrality])
    X_full_4 = sm.add_constant(X_full_4)
    model4 = sm.OLS(y1, X_full_4).fit()
    result4 = [model4.params[1],model4.bse[1],model4.pvalues[1]] # coef, se, pvalue

    return result4

def get_chen_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values
    emb_list = ['E'+str(i) for i in range(dimensions)]
    Z_embedding = data[emb_list].values


    # Chen et al. (2022): y1 = influ + y0 + embedding    
    X_full_5 = np.hstack([influ,y0,Z_embedding])
    X_full_5 = sm.add_constant(X_full_5)
    model5 = sm.OLS(y1, X_full_5).fit()
    result5 = [model5.params[1],model5.bse[1],model5.pvalues[1]] # coef, se, pvalue

    return result5

def get_chen_c_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values
    centrality_list = ['C'+str(i) for i in range(10)]
    Z_centrality = data[centrality_list].values
    emb_list = ['E'+str(i) for i in range(dimensions)]
    Z_embedding = data[emb_list].values


    # Chen et al.(2022) + Centrality:  y1 = influ + y0 + embedding + centrality    
    X_full_6 = np.hstack([influ,y0,Z_centrality,Z_embedding])
    X_full_6 = sm.add_constant(X_full_6)
    model6= sm.OLS(y1, X_full_6).fit()
    result6 = [model6.params[1],model6.bse[1],model6.pvalues[1]] # coef, se, pvalue

    return result6 

def get_chen_n_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values

    emb_list = ['E'+str(i) for i in range(dimensions)]
    Z_embedding = data[emb_list].values
    neighbor_list = ['N'+str(i) for i in range(dimensions)]
    avg_neighbor=data[neighbor_list].values

    # Chen et al.(2022) + Neighbor mbedding:  y1 = influ + y0 + embedding + neighbor embeddding
    X_full_7 = np.hstack([influ,y0,Z_embedding,avg_neighbor])
    X_full_7 = sm.add_constant(X_full_7)
    model7 = sm.OLS(y1, X_full_7).fit()
    result7 = [model7.params[1],model7.bse[1],model7.pvalues[1]] # coef, se, pvalue

    return result7

def get_chen_n_c_coef(datafile,dimensions):
    data = pd.read_csv(datafile,header=0,index_col=0)
    print('all data shape:',data.shape)
    index = list(data.index)
    index = np.asarray(index).reshape(-1,1)

    ##### all variables #####
    y1 = data[['y1']].values
    y0 = data[['y0']].values
    influ = data[['influence']].values
    
    centrality_list = ['C'+str(i) for i in range(10)]
    Z_centrality = data[centrality_list].values
    
    emb_list = ['E'+str(i) for i in range(dimensions)]
    Z_embedding = data[emb_list].values
    
    neighbor_list = ['N'+str(i) for i in range(dimensions)]
    avg_neighbor=data[neighbor_list].values

    # Chen et al.(2022) + Neighbor mbedding + Centrality:  y1 = influ + y0 + embedding + neighbor emb + centrality
    X_full_8 = np.hstack([influ,y0,Z_centrality,Z_embedding,avg_neighbor])
    X_full_8 = sm.add_constant(X_full_8)
    model8= sm.OLS(y1, X_full_8).fit()
    result8 = [model8.params[1],model8.bse[1],model8.pvalues[1]] # coef, se, pvalue

    return  result8
    
def get_match_psm(datafile,control_value):
    data = pd.read_csv(datafile,header=0,index_col=0)
    med = data['influence'].median()
    data['binary_influ'] = np.where(data['influence']>med,1,0)
    psm_list = ['node_id','binary_influ']+control_value
    data_psm = data[psm_list]

    psm = PsmPy(data_psm, treatment='binary_influ', indx='node_id', exclude = [])

    try:
        psm.logistic_ps(balance = True)
    except Exception:
        psm.logistic_ps(balance = False)

    # psm.predicted_data
    data_pred = psm.predicted_data
    data_pred['weight'] = np.where(data_pred['binary_influ']==1,1/data_pred['propensity_score'],1/(1-data_pred['propensity_score']))
    contains_nan = data_pred.isin([float('inf'), float('-inf')]).any().any()

    if contains_nan:
        print("The DataFrame contains inf values.")
        infinity_rows = data_pred[data_pred.isin([np.inf]).any(1)]['node_id'].tolist()
        data = data[~data['node_id'].isin(infinity_rows)]
        data.drop(infinity_rows, inplace = True)
        data_psm = data[psm_list]
        psm = PsmPy(data_psm, treatment='binary_influ', indx='node_id', exclude = [])
        try:
            psm.logistic_ps(balance = True)
        except Exception:
            psm.logistic_ps(balance = False)
        # psm.predicted_data
        data_pred = psm.predicted_data
        data_pred['weight'] = np.where(data_pred['binary_influ']==1,1/data_pred['propensity_score'],1/(1-data_pred['propensity_score'])) 
    else:
        pass

    psm.knn_matched(matcher='propensity_logit', replacement=True, drop_unmatched=True)
    
    # Get matched pairs
    match_pairs= psm.matched_ids
    print('PSM predict data:',len(match_pairs))

    ############## For Matching ################
    t_id = match_pairs['node_id'].tolist()
    c_id = match_pairs['matched_ID'].tolist()

    treatment_data = []
    for item in t_id:
        row = data[data['node_id']==item].iloc[0].tolist() + data_pred[data_pred['node_id']==item]['weight'].tolist()
        treatment_data.append(row)

    control_data = []
    for item in c_id:
        row = data[data['node_id']==item].iloc[0].tolist() + data_pred[data_pred['node_id']==item]['weight'].tolist()
        control_data.append(row)

    new_df = treatment_data + control_data
    new_data = pd.DataFrame(new_df)
    new_data.columns = list(data.columns)+['weight']
    print('New data:',len(new_data))

    all_X = ['influence']+control_value
    X_full_1 = sm.add_constant(new_data[all_X])
    psm_model = sm.OLS(new_data['y1'], X_full_1)
    psm_result = psm_model.fit()

    return [psm_result.params[1],psm_result.bse[1],psm_result.pvalues[1]]


def get_match_psw(datafile,control_value):
    data = pd.read_csv(datafile,header=0,index_col=0)
    med = data['influence'].median()
    data['binary_influ'] = np.where(data['influence']>med,1,0)


    psm_list = ['node_id','binary_influ']+control_value
    data_psm = data[psm_list]
    psm = PsmPy(data_psm, treatment='binary_influ', indx='node_id', exclude = [])
    
    try:
        psm.logistic_ps(balance = True)
    except Exception:
        psm.logistic_ps(balance = False)

    # psm.predicted_data
    data_pred = psm.predicted_data
    data_pred['weight'] = np.where(data_pred['binary_influ']==1,1/data_pred['propensity_score'],1/(1-data_pred['propensity_score']))


    ############## For weighting using all data ################
    data_pred2 = [] 
    for i in range(len(data_pred)):
        item = data_pred['node_id'].iloc[i]
        row = data[data['node_id']==item].iloc[0].tolist() + data_pred[data_pred['node_id']==item][['weight','propensity_logit']].iloc[0].tolist()
        data_pred2.append(row)
    data_pred2 = pd.DataFrame(data_pred2)
    data_pred2.columns = list(data.columns)+['weight','propensity_logit']
        

    formula = "y1 ~ influence + "
    for item in control_value[0:len(control_value)-1]:
        formula = formula + str(item) + " + "
    formula = formula + control_value[-1]


    psw_model = smf.glm(formula,data = data_pred2,freq_weights = data_pred2['weight'])
    psw_result = psw_model.fit()
    coef = psw_result.params[1]
    pvalue = psw_result.pvalues[1]
    bse = psw_result.bse[1] 

    ############## For weighting using trim 1% ################

    # lower_bound = data_pred2['propensity_logit'].quantile(0.01)
    # upper_bound = data_pred2['propensity_logit'].quantile(0.99)

    # # Create a new DataFrame with trimmed data
    # trimmed_data = data_pred2[(data_pred2['propensity_logit'] >= lower_bound) & (data_pred2['propensity_logit'] <= upper_bound)]

    # model3 = smf.glm(formula,data = trimmed_data,freq_weights = trimmed_data['weight'])
    # result3 = model3.fit()
    # coef3 = result3.params[1]
    # pvalue3 = result3.pvalues[1]
    # bse3 = result3.bse[1] 


    return [coef,bse,pvalue]