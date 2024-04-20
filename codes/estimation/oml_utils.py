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


def OML_load_data(file_path,dimensions,neighbor=False):
    
    ##### get embedding as input data ######
    data = pd.read_csv(file_path,header=0,index_col=0)
    print('all data shape:',data.shape)
    y0 = data[['y0']].values
    influ = data[['influence']].values
    x_attr = np.hstack((y0,influ))
    
    ##### all input #####
    emb_list = ['E'+str(i) for i in range(dimensions)]
    emb = data[emb_list].values
    
    neighbor_list = ['N'+str(i) for i in range(dimensions)]
    if neighbor:
        emb_avg=data[neighbor_list].values
        x_emb = np.hstack((emb,emb_avg))
        print('embedding shape', x_emb.shape)      
    else:
        x_emb =emb
        print('embedding shape', x_emb.shape)
    
    ##### output label #####
    y1 = data[['y1']].values

    return x_emb, x_attr, y1

def OML_model(emb_size, attr_size, layers):
    # define two sets of inputs
    inputA = Input(shape=(emb_size,))
    inputB = Input(shape=(attr_size,))

    #the first branch uses fully connected layers to predict 
    for i in range(len(layers)-1):
        if i == 0:
            x = Dense(layers[i], activation="linear")(inputA)
        else:
            x = Dense(layers[i], activation="linear")(x)
    hidden_model = Model(inputs=inputA,outputs=x)
    # combined = Concatenate([x.outputs,inputB])
    combined = Concatenate(axis=1,name='attr_layer')([hidden_model.output,inputB])

    # apply a FC layer and then a regression prediction on the combined outputs
    z = Dense(1, activation="linear", name='final')(combined)

    model = Model(inputs=[inputA, inputB], outputs=z)

    return model

def OML_generate_fold(x_emb,x_attr,y,n_folds):
    kf = KFold(n_splits = n_folds)
    data_fold = dict()
    i=0
    for train_index, val_index in kf.split(x_emb,y):
        i = i+1
        train_input = x_emb[train_index]
        train_attr = x_attr[train_index]
        train_output = y[train_index]

        val_input = x_emb[val_index]
        val_attr = x_attr[val_index]
        val_output = y[val_index]
   
        data_fold['fold_train'+str(i)] = [[train_input,train_attr],train_output]
        data_fold['fold_test'+str(i)] = [[val_input,val_attr],val_output]


    return data_fold


def OML_CV(x_emb,x_attr,y,layers,param_grid,n_folds=5):
    best_loss = float('inf')
    best_batch_size = 0
    best_epochs = 0
    best_lr = 0
    best_loss = float("inf")

    data_fold = OML_generate_fold(x_emb,x_attr,y,n_folds)

    batch_size=param_grid['batch_size']
    epochs=param_grid['epochs']
    learn_rate=param_grid['learn_rate']
    best_model= OML_model(x_emb.shape[1], x_attr.shape[1], layers)

    for n_batch in batch_size:
        for n_epoch in epochs:
            for lr in learn_rate:
                avg_loss = 0
                for i in range(n_folds):
                    model = OML_model(x_emb.shape[1], x_attr.shape[1], layers)
                    optimizer = Adam(lr=lr)
                    model.compile(loss='mean_squared_error', optimizer=optimizer)

                     # fit model
                    train_data = data_fold['fold_train'+str(i+1)]
                    test_data = data_fold['fold_test'+str(i+1)]
                    es = EarlyStopping(monitor='val_loss',patience=5, mode='min', verbose=1)
                    history = model.fit(x=train_data[0],y = train_data[1],validation_data=(test_data[0], test_data[1]), epochs=n_epoch,batch_size=n_batch, verbose=0,callbacks=[es]) 

                    # evaluate the model
                    train_loss = model.evaluate(train_data[0], train_data[1], verbose=0)
                    test_loss = model.evaluate(test_data[0], test_data[1], verbose=0)
                    avg_loss = avg_loss + test_loss/n_folds
                print("Test Loss={0:.2f}".format(avg_loss))
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_batch_size = n_batch
                    best_epochs = n_epoch
                    best_lr = lr
                    best_model = model
                    
    return best_model,best_batch_size,best_epochs,best_lr

def OML_get_estimation_result(best_model,x_emb,x_attr, y,layers):
    
    attr_model = Model(best_model.input, best_model.get_layer('attr_layer').output) 

    # Get final predition results
    total_loss = best_model.evaluate([x_emb,x_attr], y, verbose=0)
    print('Total loss: %.3f' % (total_loss))

    weight_influ = best_model.get_layer('final').get_weights()[0][-1][0] # weights
    print('beta coefficient:', weight_influ)
    bias = best_model.get_layer('final').get_weights()[1] # bias

    ############# calculate se, p, and CI
    attr_output = attr_model.predict([x_emb,x_attr])
    intercept = np.ones([len(attr_output),1])
    X_design = np.hstack((intercept,attr_output))
    pred = best_model.predict([x_emb,x_attr])
    residual = y - pred
    var_res = np.mean(abs(residual - residual. mean())**2) 
    cov_coefficient = (inv(((X_design.T).dot(X_design))))*var_res 
    se = np.sqrt(cov_coefficient)
    se_influ = se[-1,-1]
    t_influ = weight_influ/se_influ

    df = len(X_design)-layers[-1]-x_attr.shape[1]-1
    pval_influ = stats.t.sf(np.abs(t_influ), df)*2 # two-sided pvalue = Prob(abs(t)>tt)

    
    return [weight_influ,se_influ,t_influ,pval_influ,total_loss]
    
