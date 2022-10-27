# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:12:16 2019

@author: Mercuro
"""

#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
import statsmodels.api as sm
import pandas as pd 
import copy
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


#df = Raw_data
#Y_var = '诊断_new'
def LG_run_univariable(df,Y_var ,X_list_need,adjust_covr=False):
    res_need = df.copy(deep  =True)
#    Y_in = res_need[Y_var]
    X_ALL = res_need.drop(columns = Y_var)
#    X_list_need = X_ALL.columns
#    if(delete==True):
#        X_list_need = X_list_need.drop(delete_var)
    var_name,coef_ ,p_value = [],[],[]
    summary_ = []
#    res_need['intercept'] =1
    for i in range(len(X_list_need)):

        if(adjust_covr==False):
            print(X_list_need[i])
            temp_X =  res_need.loc[:,[ X_list_need[i],Y_var]]
            temp_X.dropna(inplace=True)
            lg_model = smf.logit( "{} ~ {}".format(Y_var,' + '.join([X_list_need[i]])),temp_X ).fit()
        else:
            print(X_list_need[i])
            temp_X =  res_need.loc[:,[ '年龄_new','性别_new',X_list_need[i],Y_var]]
            temp_X.dropna(inplace=True)
            lg_model = smf.logit( "{} ~ 性别_new + 年龄_new + {}".format(Y_var,' + '.join([X_list_need[i]])),temp_X ).fit()
        results = lg_model.summary()

        var_name.append(X_list_need[i])
        coef_.append(lg_model.params[X_list_need[i]])
        p_value.append(lg_model.pvalues[X_list_need[i]])
#        df[ X_list_need[i]].value_counts()
#        temp_X[ X_list_need[i]].value_counts()
#        pd.crosstab(temp_X[ X_list_need[i]],
#                    temp_X.loc[:,Y_var])
    temp_data = pd.DataFrame()
    temp_data['var_name'] = var_name
    temp_data['OR'] = np.exp(coef_)
    temp_data['p_value'] = p_value
    return temp_data


'''
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
'''

 


#df = multivariable_data
#Y_var = '诊断_new'
#always_in = ['年龄','性别']
#cut_off_in = 0.05
#cut_off_stay = 0.05
def step_wise(df,Y_var,always_in,cut_off_in,cut_off_stay,entry_var = True,show_var=False):
    temp_df = copy.deepcopy(df) 
    remain_var_list = set(temp_df.columns) 
    remain_var_list.discard(Y_var)
    if(entry_var):
        remain_var_list=remain_var_list.difference(set(always_in))
    selected = []    
    iter_len = []
    iter_num = 0
    while len(remain_var_list)>0:
    #    forward part
        forwart_p = [] 
    #    while  ########################
        for candidate in remain_var_list:
            
            selected = list(selected)
            if(entry_var):
                formula_f = "{} ~ {}".format(Y_var,' + '.join( selected +always_in + [candidate]))
            else:
                formula_f = "{} ~ {}".format(Y_var,' + '.join(selected + [candidate]))
            if show_var:
                print(formula_f)
            
            model_fit_f =  smf.logit(formula_f, data =temp_df).fit(disp=False,
                                    maxiter=200,method='bfgs' )
    #        model_fit_f.summary()
#    temp_df['分布_skin_弥散均匀'].value_counts()
            forwart_p.append( (model_fit_f.pvalues[candidate],candidate))
        
        forwart_p.sort()
        if(forwart_p[0][0]<cut_off_in): 
            selected.append(forwart_p[0][1])
    #    backward part
        back_selct_var = set(selected)
 
        while len(back_selct_var)>0: 
            if(entry_var):
                formula_b = "{} ~ {}".format(Y_var,' + '.join(list(back_selct_var) +
                             always_in ))
            else:
                formula_b = "{} ~ {}".format(Y_var,' + '.join(list(back_selct_var)))
            model_fit_b =  smf.logit(formula_b, data =temp_df).fit(disp=False, maxiter=200,method='bfgs')
#            model_fit_b.summary()
            back_pvalue = model_fit_b.pvalues[back_selct_var]
            need_delete = back_pvalue[back_pvalue>cut_off_stay]
            if(len(need_delete)==0):
                break
            else:
                sort_back_var = np.argmax(need_delete)
#                sort_back_var = np.argmax(back_pvalue)
                back_selct_var = list(back_selct_var)
                back_selct_var.remove(sort_back_var)
        selected = back_selct_var
        remain_var_list = remain_var_list.difference(set(selected))
        
        iter_num = iter_num +1
        formula_Final = "{} ~ {}".format(Y_var,' + '.join(list(selected) +always_in ))
#        print(formula_Final)
#        print(iter_num)
        iter_len.append(len(remain_var_list))
        if(iter_num>10):
            if(iter_len[-1]==iter_len[-4]):
                break
#    formula_b =   formula_Final
    print(formula_Final)
    return formula_Final
        
def auc_curve(y,prob,roc_auc,filg_save=False,
          file_add_name = './GA_results.Train.png'):
    fpr,tpr,threshold = roc_curve(y,prob) ###计算真正率和假正率
#    roc_auc = auc(fpr,tpr) ###计算auc的值
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, 
             label='ROC curve area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if filg_save:
        plt.savefig(file_add_name,dpi=100,format = 'jpg')
    plt.show() 
#    plt.savefig()
    
def AUC_calculate(formula_in,df_train,df_test,Y_var,plot_res = False):
    model_fit_b =  smf.logit(formula_in, 
                      data =df_train).fit(disp=False)
    predic_train = model_fit_b.predict(df_train)
    predic_test = model_fit_b.predict(df_test)
    ROC_train = roc_auc_score(y_true =df_train[Y_var] ,y_score = predic_train)
    ROC_test = roc_auc_score(y_true =df_test[Y_var] ,y_score = predic_test)
    auc_curve(df_train[Y_var],predic_train,ROC_train)
    auc_curve(df_test[Y_var],predic_test,ROC_test)
    return ROC_train,ROC_test




def Runing_step_wise(df_train_x,df_train_y,df_test_x,df_test_y,
                     addr = r'D:\Python_project\Chen_skin\results\20190929\Ct_'):
    
    combine_res_all_var_in_Train = step_wise(df=pd.concat((df_train_x ,
                                             df_train_y),
                                                    axis=1),
                                       Y_var = '诊断_new',
                                       always_in = ['年龄_new','性别_new'],
                                       cut_off_in = 0.10,
                                       cut_off_stay = 0.05,
                                       entry_var = True)
    
    model_tradition = smf.logit(combine_res_all_var_in_Train, 
                      data =pd.concat((df_train_x ,
                                            df_train_y),
                                                   axis=1)).fit(disp=False,
                        maxiter=200,method='bfgs')

    model_tradition.summary()
    model_tradition.params
    model_tradition.pvalues
    CI_OR = round(np.exp(model_tradition.conf_int()),2)
    OR = round(np.exp(model_tradition.params),2)
    
    res_logistic_t1 =  pd.concat((OR,
                                CI_OR),axis=1)
    
    res_logistic_t2 = pd.concat((model_tradition.params,
                                model_tradition.pvalues),axis=1)
    
    res_logistic_t3 = pd.concat((res_logistic_t1,
                                res_logistic_t2),axis=1)
    res_logistic_t3.columns = ['OR','LOR','UOR','coef','P']
    res_logistic_t3.to_csv(addr + '_train_model.csv',encoding = 'utf_8_sig')
    print('#'*12 + 'Train' + '#'*12)
    print(res_logistic_t3)
    del res_logistic_t3
    
    
    model_tradition = smf.logit(combine_res_all_var_in_Train, 
                      data =pd.concat((df_test_x ,
                                            df_test_y),
                                                   axis=1)).fit(disp=False,
                        maxiter=200,method='bfgs')

    model_tradition.summary()
    model_tradition.params
    model_tradition.pvalues
    CI_OR = round(np.exp(model_tradition.conf_int()),2)
    OR = round(np.exp(model_tradition.params),2)
    
    res_logistic_t1 =  pd.concat((OR,
                                CI_OR),axis=1)
    
    res_logistic_t2 = pd.concat((model_tradition.params,
                                model_tradition.pvalues),axis=1)
    
    res_logistic_t3 = pd.concat((res_logistic_t1,
                                res_logistic_t2),axis=1)
    res_logistic_t3.columns = ['OR','LOR','UOR','coef','P']
    res_logistic_t3.to_csv(addr + '_test_model.csv',encoding = 'utf_8_sig')
    
    
    print('#'*12 + 'Test' + '#'*12)
    print(res_logistic_t3)
    #res_logistic_t3.to_csv('./Chen_skin/results/Lgistic_res.csv',
    #                       encoding = 'utf_8_sig')
    prob_train = model_tradition.predict(pd.concat((df_train_x ,
                                                df_train_y),
                                                       axis=1))
    prob_test = model_tradition.predict(pd.concat((df_test_x ,
                                                df_test_y),
                                                       axis=1))
    
#    roc_auc_score(y_true =df_train_y ,y_score= prob_train)
    
#    roc_auc_score(y_true =df_test_y ,y_score= prob_test) 
    
    auc_curve(df_train_y,
              prob=prob_train,
              roc_auc =  roc_auc_score(y_true =df_train_y ,y_score= prob_train),
              filg_save=True,
              file_add_name = addr + '_train_.pdf')
    
    auc_curve(df_test_y,
              prob=prob_test,
              roc_auc =  roc_auc_score(y_true =df_test_y ,y_score= prob_test) ,
              filg_save=True,
              file_add_name = addr + '_test_.pdf')
    
    
    
    
    
    
    
  

def Runing_Selected_model(df_train_x,df_train_y,df_test_x,df_test_y,
                          model_fit = '诊断_new ~ 血流速度_CT_new + 角质层_CT_new + 年龄_new + 性别_new',
                     addr = r'D:\Python_project\Chen_skin\results\20190929\Ct_'):
    
    
    model_tradition = smf.logit(model_fit, 
                      data =pd.concat((df_train_x ,
                                            df_train_y),
                                                   axis=1)).fit(disp=False,
                        maxiter=200,method='bfgs')

    model_tradition.summary()
    model_tradition.params
    model_tradition.pvalues
    CI_OR = round(np.exp(model_tradition.conf_int()),2)
    OR = round(np.exp(model_tradition.params),2)
    
    res_logistic_t1 =  pd.concat((OR,
                                CI_OR),axis=1)
    
    res_logistic_t2 = pd.concat((model_tradition.params,
                                model_tradition.pvalues),axis=1)
    
    res_logistic_t3 = pd.concat((res_logistic_t1,
                                res_logistic_t2),axis=1)
    res_logistic_t3.columns = ['OR','LOR','UOR','coef','P']
    res_logistic_t3.to_csv(addr + '_train_model.csv',encoding = 'utf_8_sig')
    print('#'*12 + 'Train' + '#'*12)
    print(res_logistic_t3)
    del res_logistic_t3
    
 
    #res_logistic_t3.to_csv('./Chen_skin/results/Lgistic_res.csv',
    #                       encoding = 'utf_8_sig')
    prob_train = model_tradition.predict(pd.concat((df_train_x ,
                                                df_train_y),
                                                       axis=1))
    prob_test = model_tradition.predict(pd.concat((df_test_x ,
                                                df_test_y),
                                                       axis=1))
    
#    roc_auc_score(y_true =df_train_y ,y_score= prob_train)
    
#    roc_auc_score(y_true =df_test_y ,y_score= prob_test) 
    
    auc_curve(df_train_y,
              prob=prob_train,
              roc_auc =  roc_auc_score(y_true =df_train_y ,y_score= prob_train),
              filg_save=True,
              file_add_name = addr + '_train_.jpg')
    
    auc_curve(df_test_y,
              prob=prob_test,
              roc_auc =  roc_auc_score(y_true =df_test_y ,y_score= prob_test) ,
              filg_save=True,
              file_add_name = addr + '_test_.jpg')
    
    
    
    
    
    
    
    
    
    
