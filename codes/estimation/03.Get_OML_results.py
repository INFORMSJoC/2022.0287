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

from sklearn.model_selection import KFold
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.models import model_from_json

from oml_utils import OML_load_data,OML_CV,OML_get_estimation_result

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
    x_emb,x_attr, y = OML_load_data(datafile,dimensions,neighbor)

    emb_size = x_emb.shape[1]
    attr_size = x_attr.shape[1]

    # Get best model with CV
    best_model, best_batch_size,best_epochs,best_lr = OML_CV(x_emb,x_attr,y,layers,param_grid,n_folds=5)

    # Get final predition results: beta_coef,se, t,pvalue,total_loss
    estimation_result = OML_get_estimation_result(best_model,x_emb,x_attr, y,layers)

    new_result.append(estimation_result + [best_batch_size,best_epochs,best_lr])

oml_df = pd.DataFrame(new_result)
oml_df.columns = ['beta_coef','se','t','pvalue','loss','batch_size','epochs','lr']
oml_df['#pvalue<0.05'] = np.where(oml_df['pvalue']<0.05,1,0)
oml_df_mean = oml_df.mean(axis=0)
oml_df.loc['Avg_value'] = oml_df_mean
oml_df_final = oml_df[['beta_coef','se','pvalue','#pvalue<0.05','t','loss','batch_size','epochs','lr']]
oml_df_final.to_excel(output_folder+'Table2(d)_OML_results_pure_homophily.xlsx')

print('---------Complet OML estimation!---------')

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
    x_emb,x_attr, y = OML_load_data(datafile,dimensions,neighbor)

    emb_size = x_emb.shape[1]
    attr_size = x_attr.shape[1]

    # Get best model with CV
    best_model, best_batch_size,best_epochs,best_lr = OML_CV(x_emb,x_attr,y,layers,param_grid,n_folds=5)

    # Get final predition results: beta_coef,se, t,pvalue,total_loss
    estimation_result = OML_get_estimation_result(best_model,x_emb,x_attr, y,layers)

    new_result.append(estimation_result + [best_batch_size,best_epochs,best_lr])

oml_df = pd.DataFrame(new_result)
oml_df.columns = ['beta_coef','se','t','pvalue','loss','batch_size','epochs','lr']
oml_df['#pvalue<0.05'] = np.where(oml_df['pvalue']<0.05,1,0)
oml_df_mean = oml_df.mean(axis=0)
oml_df.loc['Avg_value'] = oml_df_mean
oml_df_final = oml_df[['beta_coef','se','pvalue','#pvalue<0.05','t','loss','batch_size','epochs','lr']]
oml_df_final.to_excel(output_folder+'Table2(d)_OML_results_positive_peer_effect.xlsx')

print('---------Complet OML estimation!---------')