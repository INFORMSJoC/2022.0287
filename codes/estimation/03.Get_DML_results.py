import math
import numpy as np
#import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import r2_score 
from random import shuffle 
from numpy.linalg import inv
import scipy.stats as stats
from scipy.stats import ttest_ind
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 
import statsmodels.api as sm
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, Multiply, LSTM, SimpleRNN
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

from sklearn.model_selection import KFold, StratifiedKFold
import keras.backend as K
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from keras.regularizers import l2
from dml_utils import DML_load_data, DML_generate_fold, DML_CV

num_data = 100

layers= [16,8,2]
neighbor = False
dimensions = 64
batch_size = [20,25,50]
learn_rate = [0.01,0.005]
epochs = [40,60,80,100]


param_grid = dict(batch_size=batch_size,epochs=epochs,learn_rate=learn_rate)

#----------------------------------------------#
#    Get OML results for pure homophily case   #
#----------------------------------------------#

feature_folder = 'data/Final_simulated_data_pure_homophily/Final_regression_feature/'
inputfile = '_pure_homophily_beta0_final_data.csv'
output_folder = 'data/Final_simulated_data_pure_homophily/'

print('---------Generate results for homophily pure case!---------')
new_result = []
for i in range(1,num_data+1):
    datafile = feature_folder + str(i) + inputfile
    x_attr, influ, y = DML_load_data(datafile,dimensions,neighbor)
    print('-------Dataset {}---------'.format(str(i)))
    print('x_attr:\t{}'.format(x_attr.shape))

    data_fold1, data_fold2 = DML_generate_fold(x_attr,y,influ,n_folds=5)
    ## predict y ##
    pred_y,batchsize1,epochs1,lr1,loss1 = DML_CV(x_attr.shape[1],data_fold1,layers,param_grid,model_type='predict_y',n_folds=5)
    e_y1 = y - pred_y
    
    ## predict influ ##
    pred_influ, batchsize2,epochs2,lr2,loss2 = DML_CV(x_attr.shape[1],data_fold2,layers,param_grid,model_type='predict_influ',n_folds=5)
    e_influ = influ - pred_influ

    #### regression results ####
    model = sm.OLS(e_y1, e_influ).fit()
    coef = model.params[0]
    bse = model.bse[0]
    t = model.tvalues[0]
    pvalue = model.pvalues[0]
    print('beta coef :', coef) 
    r2_reg = model.rsquared

    print([coef,bse,t,pvalue,r2_reg,loss1,batchsize1,epochs1,lr1,loss2,batchsize2,epochs2,lr2])
    new_result.append([coef,bse,t,pvalue,r2_reg,loss1,batchsize1,epochs1,lr1,loss2,batchsize2,epochs2,lr2])

dml_df = pd.DataFrame(new_result)
dml_df.columns = ['beta_coef','se','t','pvalue','r2_reg','loss_y','batchsize_y','epochs_y','lr_y','loss_influ','batchsize_influ','epochs_influ','lr_influ']
dml_df['#pvalue<0.05'] = np.where(dml_df['pvalue']<0.05,1,0)
dml_df_mean = dml_df.mean(axis=0)
dml_df.loc['Avg_value'] = dml_df_mean
dml_df_final = dml_df[['beta_coef','se','pvalue','#pvalue<0.05','t','r2_reg','loss_y','batchsize_y','epochs_y','lr_y','loss_influ','batchsize_influ','epochs_influ','lr_influ']]
dml_df_final.to_excel(output_folder+'Table(d)_DML_results_pure_homophily.xlsx')

print('---------Complet DML estimation!---------')

#-------------------------------------------------------------#
#    Get regression baseline results for postive peer case    #
#-------------------------------------------------------------#

feature_folder = 'data/Final_simulated_data_positive_peer_effect/Final_regression_feature/'
inputfile = '_positve_peer_effect_beta0.2_final_data.csv'
output_folder = 'data/Final_simulated_data_positive_peer_effect/'

print('---------Generate results for positive peer effect case!---------')

new_result = []
for i in range(1,num_data+1):
    datafile = feature_folder + str(i) + inputfile
    x_attr, influ, y = DML_load_data(datafile,dimensions,neighbor)
    print('-------Dataset {}---------'.format(str(i)))
    print('x_attr:\t{}'.format(x_attr.shape))

    data_fold1, data_fold2 = DML_generate_fold(x_attr,y,influ,n_folds=5)
    ## predict y ##
    pred_y,batchsize1,epochs1,lr1,loss1 = DML_CV(x_attr.shape[1],data_fold1,layers,param_grid,model_type='predict_y',n_folds=5)
    e_y1 = y - pred_y
    
    ## predict influ ##
    pred_influ, batchsize2,epochs2,lr2,loss2 = DML_CV(x_attr.shape[1],data_fold2,layers,param_grid,model_type='predict_influ',n_folds=5)
    e_influ = influ - pred_influ

    #### regression results ####
    model = sm.OLS(e_y1, e_influ).fit()
    coef = model.params[0]
    bse = model.bse[0]
    t = model.tvalues[0]
    pvalue = model.pvalues[0]
    print('beta coef :', coef) 
    r2_reg = model.rsquared

    print([coef,bse,t,pvalue,r2_reg,loss1,batchsize1,epochs1,lr1,loss2,batchsize2,epochs2,lr2])
    new_result.append([coef,bse,t,pvalue,r2_reg,loss1,batchsize1,epochs1,lr1,loss2,batchsize2,epochs2,lr2])

dml_df = pd.DataFrame(new_result)
dml_df.columns = ['beta_coef','se','t','pvalue','r2_reg','loss_y','batchsize_y','epochs_y','lr_y','loss_influ','batchsize_influ','epochs_influ','lr_influ']
dml_df['#pvalue<0.05'] = np.where(dml_df['pvalue']<0.05,1,0)
dml_df_mean = dml_df.mean(axis=0)
dml_df.loc['Avg_value'] = dml_df_mean
dml_df_final = dml_df[['beta_coef','se','pvalue','#pvalue<0.05','t','r2_reg','loss_y','batchsize_y','epochs_y','lr_y','loss_influ','batchsize_influ','epochs_influ','lr_influ']]
dml_df_final.to_excel(output_folder+'Table(d)_DML_results_positive_peer_effect.xlsx')

print('---------Complet DML estimation!---------')