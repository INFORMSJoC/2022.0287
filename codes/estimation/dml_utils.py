import math
import numpy as np
#import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import r2_score 
import random
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
from keras.models import model_from_json


def DML_load_data(file_path,dimensions,neighbor=False):
    
    ##### get embedding as input data ######
    data = pd.read_csv(file_path,header=0,index_col=0)
    print('all data shape:',data.shape)
    y0 = data[['y0']].values
    influ = data[['influence']].values
    
    ##### all input #####
    emb_list = ['E'+str(i) for i in range(dimensions)]
    emb = data[emb_list].values
    
    neighbor_list = ['N'+str(i) for i in range(dimensions)]
    if neighbor:
        emb_avg=data[neighbor_list].values
        x_attr = np.hstack((emb,emb_avg)) 
    else:
        x_attr =np.hstack((emb, y0))
    
    ##### output label #####
    y1 = data[['y1']].values

    return x_attr, influ, y1


def predict_y(attr_size, layers):
    inputA = Input(shape=(attr_size,))

    for i in range(len(layers)):
        if i == 0:
            x = Dense(layers[i], activation="linear")(inputA)
        else:
            x = Dense(layers[i], activation="linear")(x)
    
    z = Dense(1, activation="linear", name='final')(x)
    model = Model(inputs=inputA, outputs=z)

    return model

def predict_influ(attr_size, layers):
    inputA = Input(shape=(attr_size,))

    for i in range(len(layers)):
        if i == 0:
            x = Dense(layers[i], activation="linear")(inputA)
        else:
            x = Dense(layers[i], activation="linear")(x)
    
    z = Dense(1, activation="linear", name='final')(x)
    model = Model(inputs=inputA, outputs=z)

    return model

def DML_generate_fold(x_attr,y,influ,n_folds):
    kf = KFold(n_splits = n_folds)
    data_fold1 = dict()
    data_fold2 = dict()
    i=0
    for train_index, val_index in kf.split(x_attr,y):
        i = i+1
        data_fold1['fold_train'+str(i)] = [x_attr[train_index],y[train_index]]
        data_fold1['fold_test'+str(i)] =  [x_attr[val_index],y[val_index]]

        data_fold2['fold_train'+str(i)] = [x_attr[train_index],influ[train_index]]
        data_fold2['fold_test'+str(i)] =  [x_attr[val_index],influ[val_index]]

    return data_fold1, data_fold2

def DML_CV(input_shape, data_fold,layers,param_grid,model_type,n_folds=5):
    best_loss = float('inf')
    best_batch_size = 0
    best_epochs = 0
    best_lr = 0
    best_loss = float("inf")
    batch_size=param_grid['batch_size']
    epochs=param_grid['epochs']
    learn_rate=param_grid['learn_rate']

    for n_batch in batch_size:
        for n_epoch in epochs:
            for lr in learn_rate:
                avg_loss = 0
                if model_type == 'predict_y':
                    model = predict_y(input_shape, layers)
                    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
                else:
                    model = predict_influ(input_shape, layers)
                    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))

                for i in range(n_folds):
                    # fit model
                    train_data = data_fold['fold_train'+str(i+1)]
                    test_data = data_fold['fold_test'+str(i+1)]
                    es = EarlyStopping(monitor='val_loss',patience=5, mode='min', verbose=1)
                    history = model.fit(x=train_data[0],y = train_data[1],validation_data=(test_data[0], test_data[1]), epochs=n_epoch,batch_size=n_batch, verbose=0,callbacks=[es]) 

                    # evaluate the model
                    train_loss = model.evaluate(train_data[0], train_data[1], verbose=0)
                    test_loss = model.evaluate(test_data[0], test_data[1], verbose=0)

                    avg_loss = avg_loss + test_loss/n_folds
                print("Loss={0:.2f}".format(avg_loss))
                if avg_loss < best_loss:
                    best_model = model
                    best_loss = avg_loss
                    best_batch_size = n_batch
                    best_epochs = n_epoch
                    best_lr = lr
    
    pre_result = []
    for i in range(n_folds):
        pre_probs = best_model.predict(data_fold['fold_test'+str(i+1)][0])
        pre_result = pre_result + pre_probs.tolist()

                    
    return pre_result,best_batch_size,best_epochs,best_lr,best_loss