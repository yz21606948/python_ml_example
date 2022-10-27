# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:12:29 2019

@author: Mercuro
"""
#############  part 2  MLP 缺失填0   做训练和预测 
import re
import os
import copy 
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
import pandas as pd
import numpy as np
import sklearn
from utils.A01_analysisdata  import Runing_Selected_model,LG_run_univariable ,step_wise,AUC_calculate,auc_curve,Runing_step_wise
#from  A01_analysisdata  import LG_run_univariable ,step_wise,AUC_calculate,auc_curve
from utils.A00_cleandata  import fill_with_value
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split



from keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



f = open('./Analysis_cleaned_data.csv', encoding='UTF-8')
Raw_data = pd.read_csv(f )
#CT_data = Raw_data[Raw_data['CT_new']==1]
#Visa_data = Raw_data[Raw_data['Visa_new']==1]
#Skin_data = Raw_data[Raw_data['skin_new']==1]

Raw_data = Raw_data.drop(columns=["CT_new","Visa_new","skin_new"])
Raw_data_new = Raw_data.fillna(value=-99)

Raw_data_new = Raw_data_new+1
Raw_data_new[Raw_data_new==-98] = 0

from sklearn.preprocessing import MinMaxScaler as max_min_fit
Trans_fit = max_min_fit()
Trans_fit.fit(Raw_data_new)
Transed_data = pd.DataFrame(Trans_fit.transform(Raw_data_new))
Transed_data.columns = Raw_data_new.columns





def special_res(model_in,df_in,y_in,set_zero_list=None):
    data_need = copy.deepcopy(df_in)
    if(set_zero_list!=None):
        data_need.loc[:,set_zero_list] =0
    score_ = model_in.predict_proba( data_need )
#    score_ = model_in.predict_classes( data_need )
    return roc_auc_score(y_true=y_in.iloc[:,1], y_score=score_[:,1])
#    return accuracy_score(y_in[:,1], score_[:,1])
    

#df_X = Transed_data

def  repeat_out_res(df_X,seed_in=1234,repeat_time=1000): 
    np.random.seed(seed_in)
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import RMSprop
    import tensorflow as tf
    from sklearn import metrics
    from keras import backend as K
    import copy
 
    from keras.callbacks import EarlyStopping
    
    need_x_list = df_X.columns
    need_x_list_all = need_x_list.drop("诊断_new")
    
    for i in range(repeat_time):
        print('*'*20)
        print(i)
        print('*'*20)
        K.clear_session()
    
        early_stopping = EarlyStopping(monitor='loss', patience=50,
                                       mode = 'min', verbose=2)
        X_train, X_test, y_train, y_test = train_test_split(df_X.loc[:,need_x_list_all],
                                                                        df_X.loc[:,'诊断_new'], 
                                                                        train_size=4/5, 
                                                                        test_size=1/5,
                                                                        random_state=seed_in+i*100 )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
 
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)
        
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        
        model_fit = Sequential()
        model_fit.add(Dense(128, activation='relu', input_shape=(59,)))
#        model_fit.add(Dense(64, activation='relu', input_shape=(25,)))
        model_fit.add(Dropout(0.5))
        model_fit.add(Dense(128, activation='relu'))
        model_fit.add(Dropout(0.5))
        model_fit.add(Dense(2, activation='softmax'))
        model_fit.summary()
        
        model_fit.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        history = model_fit.fit(X_train.loc[:,need_x_list], y_train,
                            batch_size=16,
                            shuffle=True,
                            epochs=100,
                            verbose=0,
                            validation_data=(X_test.loc[:,need_x_list], y_test),
                            callbacks=[roc_callback(training_data=(X_train.loc[:,need_x_list], y_train),
                                      validation_data=(X_test.loc[:,need_x_list], y_test)),
                                    early_stopping])
        
#df_X = Transed_data
repeat_out_res(df_X=Transed_data,seed_in=1234,repeat_time=1000)
