# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:14:52 2019

@author: Mercuro
"""
# %reset -f
import os 
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score,roc_curve
import pandas as pd
import numpy as np
import sklearn
from utils.A01_analysisdata  import LG_run_univariable ,step_wise,AUC_calculate,auc_curve
#from  A01_analysisdata  import LG_run_univariable ,step_wise,AUC_calculate,auc_curve
#from utils.A00_cleandata  import *

import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

f = open('./Analysis_cleaned_data.csv', encoding='UTF-8')
Raw_data = pd.read_csv(f )

#Raw_data['分支_Skin'].value_counts()

#pd.crosstab(Raw_data['分支_Skin'],Raw_data['诊断_new'])

#Raw_data['围绕附属器_Skin'].value_counts(dropna=False)

#from scipy import stats
#Raw_data['年龄'].describe()
#Raw_data['年龄'].value_counts(dropna=False)
#
#Raw_data['性别_new'].describe()
#Raw_data['性别_new'].value_counts(dropna=False)
#
#case_age = Raw_data['年龄'][Raw_data['诊断_new']==1]
#
#case_age.dropna(inplace = True)
#case_age.describe()
#
#control_age = Raw_data['年龄'][Raw_data['诊断_new']==0]
#
#control_age.dropna( inplace = True)
#control_age.describe()
#stats.ttest_ind( case_age, 
#                 control_age,equal_var = False)



#from  scipy.stats import chi2_contingency
#import numpy as np
#kf_data = np.array([[38,359], [26,93]])
#kf = chi2_contingency(kf_data)




#model_tradition = smf.logit('诊断_new ~ 年龄', 
#                      data =pd.concat((X_train ,
#                                            y_train),
#                                                   axis=1)).fit(disp=False)
#
#model_tradition.summary()
  


#pd.crosstab(Raw_data.loc[:,'诊断_new'],Raw_data.loc[:,'性别'])


#
#all_columns = pd.DataFrame(Raw_data.columns)
#all_columns.columns = ['check_var']
#cleaned_data = search_key_word1(df = all_columns,
#                           target_var = 'check_var',
#                          pattern_in1 = 'CT',
#                         new_var =  'CT_count')
#cleaned_data['CT_count'].value_counts()
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = 'check_var',
#                          pattern_in1 = 'Skin',
#                         new_var =  'Skin_count')
#
#cleaned_data['Skin_count'].value_counts()
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = 'check_var',
#                          pattern_in1 = 'VISIA',
#                         new_var =  'VISIA_count')
#
#cleaned_data['VISIA_count'].value_counts()
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = 'check_var',
#                          pattern_in1 = 'nan',
#                         new_var =  'nan_count')
#cleaned_data['nan_count'].value_counts()


''' 传统分析  单因素 有意义  逐步法跑多因素 '''

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Raw_data['其他_Skin_屑'].value_counts()
#Raw_data.drop(columns = ['其他_Skin_屑','脓疱_Skin_new','其他_Skin_毛囊_红晕',
#                         '围绕附属器_Skin',
#                         '角质层_CT_new', '血管形态_CT_线状','其他_Skin_脓疱_红晕'
#                         ,'苯环_Skin','其他_Skin_眼外侧'], 
#                inplace = True)

#Raw_data['网格_Skin_new'].value_counts()

#for ii in Raw_data.columns:
#    print(Raw_data[ii].value_counts())

#pd.crosstab(Raw_data['苯环_Skin'],Raw_data['诊断_new'])
#Raw_data.dropna(axis=1,inplace=True,thresh=262)

#Raw_data.dropna( subset = ['性别'],inplace = True) ##########

#Raw_data.fillna(value=dict(zip(Raw_data.columns,np.nanmean(Raw_data,axis=0 ))),
#          inplace = True,
#          axis=0)  

#Raw_data.fillna(value=0,
#          inplace = True,
#          axis=0)  
#Raw_data.fillna(value=dict(zip(Raw_data.columns,np.nanmean(Raw_data,axis=0 ))),
#          inplace = True,
#          axis=0)  

#Raw_data['形态_Skin_多角形'].value_counts()
#Raw_data['围绕毛囊_CT_有无'].value_counts()
#Raw_data['表皮海绵水肿_CT_nan'].value_counts()

from sklearn.model_selection import train_test_split
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Raw_data['形态_Skin_多角形'].value_counts()


#Raw_data.drop(columns = ['形态_Skin_多角形','表皮海绵水肿_CT_nan','围绕毛囊_CT_有无' ], 
#                inplace = True)
#Raw_data_X = Raw_data.drop(columns = ['诊断_new' ,'血管形态_Skin_线状','苯环_Skin',
#                                      '其他_Skin_眼外侧_苯环',
#                                      '角质层_CT_new',
#                                      '血管形态_CT_线状','其他_Skin_脓疱_红晕',
#                                      '围绕附属器_Skin','其他_Skin_眼外侧'], 
#                inplace = False)

#Raw_data_X = Raw_data.drop(columns = ['诊断_new' ,'血管形态_Skin_线状'
##                                      '其他_Skin_眼外侧_苯环',
#                                       ], 
#                inplace = False)
#pd.crosstab(Raw_data['诊断_new'],Raw_data['其他_Skin_眼外侧'])
#pd.crosstab(Raw_data['诊断_new'],Raw_data_X['苯环_Skin'])
#pd.crosstab(Raw_data['诊断_new'],Raw_data['其他_Skin_眼外侧_苯环'])

#Raw_data_X['浅层受累毛囊百分比'] = Raw_data_X['浅层受累毛囊百分比']*10
#Raw_data_X['深层受累毛囊百分比'] = Raw_data_X['深层受累毛囊百分比']*10



Raw_data  = Raw_data.drop(columns = ['CT_new', 'Visa_new', 'skin_new'])
#            ,
#                                     '表皮海绵水肿_CT_new_块缺失'])

#sss = Raw_data_X.corr('spearman')
#sss1 = sss.describe()
Raw_data_X = Raw_data.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(Raw_data_X,
                                                    Raw_data.loc[:,'诊断_new'], 
                                                    train_size=0.75, test_size=0.25,random_state=12345678)


'''
    #########################################################
    #########################################################
    #########################################################
'''
# =============================================================================
# univariable_res = LG_run_univariable(df =Raw_data,
#                                      Y_var='诊断_new')
# 
# univariable_res_select = univariable_res[univariable_res['p_value']<0.05]
# 
# univariable_res_select_list = univariable_res_select['var_name']
# 
# analysis_need = set(univariable_res_select_list ).union(set(['诊断_new','性别','年龄']))
# 
# multivariable_data = Raw_data.loc[:,analysis_need]
# 
# combine_res = step_wise(df=multivariable_data,
#           Y_var = '诊断_new',
#           always_in = ['年龄','性别'],
#           cut_off_in = 0.05,
#           cut_off_stay = 0.05,entry_var = True)
# =============================================================================



'''
    #########################################################
    #########################################################
    #########################################################
'''



# =============================================================================
# univariable_res_train = LG_run_univariable(df =pd.concat((X_train,y_train),
#                                                    axis=1),
#                                      Y_var='诊断_new')
# 
# univariable_res_select = univariable_res_train[univariable_res_train['p_value']<0.05]
# univariable_res_select_list = univariable_res_select['var_name']
# 
# #univariable_res_test = LG_run_univariable(df =pd.concat((X_test,y_test),
# #                                                   axis=1),
# #                                     Y_var='诊断_new')
# 
# analysis_need = set(univariable_res_select_list ).union(set([ '性别','年龄']))
# 
# 
# combine_res = step_wise(df=pd.concat((X_train.loc[:,analysis_need],
#                                             y_train),
#                                                    axis=1),
#                                       Y_var = '诊断_new',
#                                       always_in = ['年龄','性别'],
#                                       cut_off_in = 0.05,
#                                       cut_off_stay = 0.05,entry_var = True)
# 
# 
# model_tradition = smf.logit(combine_res, 
#                       data =pd.concat((X_train ,
#                                             y_train),
#                                                    axis=1)).fit(disp=False)
# 
# 
# prob_train = model_tradition.predict(pd.concat((X_train ,
#                                             y_train),
#                                                    axis=1))
# prob_test = model_tradition.predict(pd.concat((X_test ,
#                                             y_test),
#                                                    axis=1))
# 
# auc_curve(y_train,
#           prob=prob_train,
#           roc_auc =  roc_auc_score(y_true =y_train ,y_score= prob_train),
#           filg_save=True,
#           file_add_name = './Tradition.Train.pdf')
# 
# auc_curve(y_test,
#           prob=prob_test,
#           roc_auc =  roc_auc_score(y_true =y_test ,y_score= prob_test),
#           filg_save=True,
#           file_add_name = './Tradition.Test.pdf')
# 
#  
# =============================================================================
combine_res_all_var_in_Train = step_wise(df=pd.concat((X_train ,
                                             y_train),
                                                    axis=1),
                                       Y_var = '诊断_new',
                                       always_in = ['年龄_new','性别_new'],
                                       cut_off_in = 0.10,
                                       cut_off_stay = 0.05,
                                       entry_var = True)

#'诊断_new ~ 面颊_VISIA + 耳前_VISIA + 血管形态_CT_多角形_网格状 + 
#血流速度_CT_new + 粗细_Skin_粗细不均 + 年龄 + 性别'


#combine_res_all_var_in_test = step_wise(df=pd.concat((X_test ,
#                                             y_test),
#                                                    axis=1),
#                                       Y_var = '诊断_new',
#                                       always_in = ['年龄','性别'],
#                                       cut_off_in = 0.10,
#                                       cut_off_stay = 0.05,
#                                       entry_var = True)

#诊断_new ~ 面颊_VISIA + 耳前_VISIA + 血管形态_CT_多角形_网格状 + 血流速度_CT_new + 
#粗细_Skin_粗细不均 + 年龄 + 性别

#df['分支_Skin'].value_counts()

#X_train['其他_Skin_眼外侧'].value_counts()

#formula_b = '''诊断_new ~ 血管形态_Skin_点状 + 数量_Skin_致密中等 + 单个毛囊最多毛囊虫数目_CT_sh_new + 
#        角质层_屏障表现 + 眼睑血管_VISIA + 分布_Skin_弥散均匀 + 角质层_CT_new + 年龄 + 
#        性别 +苯环_Skin '''
#formula_b ='''诊断_new ~ 血管形态_Skin_点状 + 眼睑血管_VISIA + 分布_Skin_弥散均匀 + 
#角质层_CT_new + 年龄 + 性别 + 苯环_Skin'''
#
#formula_b = '诊断_new ~ 苯环_Skin'
#model_fit_b =  smf.logit(formula_b, data =pd.concat((X_train ,
#                                             y_train), axis=1)).fit(disp=False,
#                                                method='bfgs')
#model_fit_b.params
#model_fit_b.pvalues


#sss = data.loc[:,['诊断_new','血管形态_Skin_点状','数量_Skin_致密中等',
#            '单个毛囊最多毛囊虫数目_CT_sh_new','角质层_屏障表现','眼睑血管_VISIA',
#            '分布_Skin_弥散均匀','角质层_CT_new','年龄','性别','苯环_Skin']].corr('spearman')




#X_train['围绕附属器_Skin'].value_counts()

#X_train['数量_Skin_中等'].value_counts()



model_tradition = smf.logit(combine_res_all_var_in_Train, 
                      data =pd.concat((X_train ,
                                            y_train),
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
#res_logistic_t3.to_csv('./Chen_skin/results/Lgistic_res.csv',
#                       encoding = 'utf_8_sig')
prob_train = model_tradition.predict(pd.concat((X_train ,
                                            y_train),
                                                   axis=1))

prob_train = prob_train[np.isnan( prob_train)==False]
#loci_train = np.where(np.isnan( prob_train)==False)
 


#X_train['红斑整体程度_VISIA_new'].value_counts(dropna = False)
#
#X_train['围绕附属器_Skin_new'].value_counts(dropna = False)
#
#X_train['苯环_网格_Skin_new'].value_counts(dropna = False)


prob_test = model_tradition.predict(pd.concat((X_test ,
                                            y_test),
                                                   axis=1))


prob_test = prob_test[np.isnan(prob_test)==False]
#loci_test = np.where(np.isnan( prob_test))


roc_auc_score(y_true =y_train[prob_train.index] ,
              y_score= prob_train)

roc_auc_score(y_true =y_test[prob_test.index] ,
              y_score= prob_test) 



auc_curve(y_train,
          prob=prob_train,
          roc_auc =  roc_auc_score(y_true =y_train ,y_score= prob_train),
          filg_save=True,
          file_add_name = './Tradition.Train.pdf')

auc_curve(y_test,
          prob=prob_test,
          roc_auc =  roc_auc_score(y_true =y_test ,y_score= prob_test) ,
          filg_save=True,
          file_add_name = './Tradition.Test.pdf')


####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
from tpot import TPOTClassifier  
import numpy as np
 
import xgboost
tpot_res = TPOTClassifier(generations=100,
                      random_state = 1324,n_jobs=2,
                      cv = 3 ,
                      population_size=100, 
                      verbosity=2)
#scoring = 'roc_auc'
tpot_res.fit(X_train, y_train)
print(tpot_res.score(X_train, y_train))   #  
print(tpot_res.score(X_test, y_test))     # 
tpot_res.export('Tpot_Skin_res.py')
 
auc_curve(y_train,
          prob=tpot_res.predict_proba(X_train)[:,1],
          roc_auc = tpot_res.score(X_train, y_train),
          filg_save=True,
          file_add_name = './GA_results.Train.pdf')

auc_curve(y_test,
          prob=tpot_res.predict_proba(X_test)[:,1],
          roc_auc = tpot_res.score(X_test, y_test),
          filg_save=True,
          file_add_name = './GA_results.Test.pdf')








# =============================================================================
# =============================================================================
# # ##################################################
# =============================================================================
# =============================================================================
from gplearn.genetic import SymbolicTransformer,SymbolicClassifier
#
#
from gplearn.functions import make_function
def _And(x1, x2):
    temp = np.logical_and(x1>0,x2>0)
    return np.where(temp,1,0)
     
And = make_function(function=_And,
                        name='And',
                        arity=2)

def _Or(x1, x2):
    temp = np.logical_or(x1>0,x2>0)
    return np.where(temp,1,0)
     
Or = make_function(function=_Or,
                        name='Or',
                        arity=2)
##
gp1 = SymbolicTransformer(generations=100, 
                          init_depth = (2,4),
#                          metric='spearman',
                          population_size=1000,
                          const_range = None,
                         hall_of_fame=100, 
                         n_components=10,
                         function_set=[And,Or],
                         parsimony_coefficient='auto',
                         max_samples=0.5, verbose=1,
                         random_state=100, n_jobs=1)

sample_weight_in = y_train
sample_weight_in[y_train==1] = 1 - sum(y_train)/len(y_train)
sample_weight_in[y_train==0] =  sum(y_train)/len(y_train)
gp1.fit(X_train, y_train,sample_weight=sample_weight_in)
print(gp1)
#
Train = gp1.transform(X_train)
X_train.columns[[2,37]]
#X_train.columns[[16,59,37]]
#X_train.columns[[30,36,37,2]]
#X_train.columns[[31,37]] ## Index(['眉弓_VISIA', '红斑整体程度_VISIA'], dtype='object')
#
##Train[:,0]
##X_train.iloc[:,2]
##Train[:,0] == X_train.iloc[:,2]
#
#X_train['红斑整体程度_VISIA'].value_counts()
#X_train['眉弓_VISIA'].value_counts()





