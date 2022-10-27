# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:25:06 2019

@author: Mercuro
"""

####  factor variable
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
import re



def fill_with_value(df):
    temp_ = np.nanmedian(df,axis=0) 
    list_ = dict(zip(df.columns,temp_))
    df = df.fillna(value=list_)
    return df
 


def factor(Vector,label_in):
    temp_d = Vector.copy(deep=True)
    temp_d.fillna(np.nan,inplace = True)
    j = 0
    for i in label_in:
        temp_d[temp_d==i] = j
        j=j+1
    return(temp_d )
    

     

def show_crass_table(df,var_,addr='./delete.csv' ,save_ = True,
                     col_show = True):
    if(len(var_)==2):
        res = pd.crosstab(df[var_[0]],df[var_[1]],margins=True,
                          rownames=[var_[0]],
                          colnames=[var_[1]],
                         dropna=False  )
    else:
        res =  df[var_[0]].value_counts(dropna = False)
        res = pd.DataFrame(res  )
        if(col_show):
            res = pd.DataFrame(np.transpose(res))
        res['频数'] = res.index
#        res = res.loc[:,['频数',var_]]
        res = res.iloc[:,[1,0]]
        res.columns = [var_[0],'频数']
    if(save_):
        res.to_csv(addr,encoding = 'utf_8_sig')
    print(res.shape)
    return res

#show_crass_table(Raw_data,['诊断'])
    

def wordcloud_(df,var_,addr='delete.tiff',stopwords = {'NaN':0}):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt  #绘制图像的模块
    import  jieba                    #jieba分词
    try_= df[var_]
    try_ = try_.fillna(' ')
    try_ = try_.to_string(index = False)
    try_ = try_.replace(' ','')
    try_ = try_.replace('\n',' ')
    
    cut_text = jieba.cut(try_,cut_all=True,HMM = True)
    
    
    result = " ".join(cut_text)
    wordcloud = WordCloud( width = 450,height = 250,
                            min_font_size = 10,
                           font_path = 'C:\Windows\Fonts\msyh.ttc', # 字体
                           background_color = 'black', # 背景色
                           max_words = 200, # 最大显示单词数
#                           max_font_size = 60, # 频率最大单词字体大小
                           stopwords = stopwords # 过滤噪声词
                         ).generate(result)
    
#    wordcloud.to_image()
    wordcloud.to_file(addr)
    
#wordcloud_(Raw_data,
#          '备注_CISIA',
#          addr='delete.tiff',stopwords = {'NaN':0  })

    
def subset(df,target_var,check_list,inver=True):
    res = df.copy(deep = True)
    need_loci = []
    if(inver):  
        for ii in df[target_var]:
            need_loci.append(ii not in check_list)
    else:
        for ii in df[target_var]:
            need_loci.append(ii in check_list)       
    res = df[need_loci]
    return res
    
def Turn_to_NaN(df,target_var,check_list):
    res = df.copy(deep=True)
    res.reset_index(drop=True,inplace = True)
    for ii in res.index:
        if res.loc[ii,target_var] in check_list:
            res.loc[ii,target_var] = np.nan
    return res
            

def ONEHOT(df,target_var,prefix_var,Nan_label = ['块缺失','NaN',np.nan]):
    res = df.copy(deep = True)
    res = Turn_to_NaN(df=res,target_var=target_var,check_list=Nan_label)
    temp = pd.get_dummies(df[target_var], 
                          dummy_na=False ,
                          prefix =prefix_var )
    index_row  = pd.isna(res[target_var])
    temp.loc[list(index_row),temp.columns] = np.nan
    res = pd.concat((res,temp),axis=1)
#    res.drop(columns =target_var,inplace=True )
    return res
    
def search_key_word1(df,target_var,pattern_in1,new_var,pattern_nan ='不清楚|块缺失'):
    res = df.copy(deep=True)
    res[new_var] = np.nan
    for ii in res.index:
        if pd.isna( res.loc[ii,target_var ]):
            pass
        else:
            if re.search(pattern_nan,res.loc[ii,target_var ]):
                pass
            else:
                flag_res = re.search(pattern_in1,res.loc[ii,target_var ])
                if flag_res:
                    res.loc[ii,new_var ] = 1
                else:
                    res.loc[ii,new_var ] = 0
    return res
    
def search_key_word2(df,target_var,pattern_in1,pattern_in2,new_var,Nan_target = '块缺失|未知'):
    res = df.copy(deep=True)
    res[new_var] = np.nan
    for ii in res.index:
        if pd.isna( res.loc[ii,target_var ] ):
            pass
        else:
            flag_res1 = re.search(pattern_in1,res.loc[ii,target_var ])
            flag_res2 =  re.search(pattern_in2,res.loc[ii,target_var ] )
            flag_res3 =  re.search(Nan_target,res.loc[ii,target_var ] )
            if flag_res2:
                res.loc[ii,new_var ] = 0
            elif flag_res1:
                res.loc[ii,new_var ] = 1
            elif flag_res3:
                res.loc[ii,new_var ] = np.nan
            else:
                res.loc[ii,new_var ] = np.nan
    return res



def extract_number(df,target_var,pattern_nan,new_var,pattern_in=r'\d+',count_=2):
    res = df.copy(deep = True)
    res[new_var] = np.nan
    for ii in res.index:
        if pd.isna( res.loc[ii,target_var ]):
            pass
        else:
            if re.search(pattern_nan,res.loc[ii,target_var ]):
                pass
            else:
                temp0  = res.loc[ii,target_var ].strip()
                temp = re.findall(pattern_in,temp0,
                                  flags= re.S)
#                 pattern_in=r'\d.\S*'
#                 re.findall(pattern_in,res.loc[ii,target_var ],
#                                  flags= re.S)
#                 ii = 0
#                 pattern_in=r'\d*'
#                 print(res.loc[ii,target_var ])
#
#                 print(re.findall(pattern_in,res.loc[ii,target_var ],
#                                  flags= re.S))
#                 ii = ii+1
                temp_num = []
                for jj in temp:
                    if jj != '':
                        temp_num.append(int(jj))
                
                if len(temp_num)<=1:
                    pass
                else:
                    res.loc[ii,new_var ] = int(temp[count_])
    return res

#order_list = [['无'],['1','2','3'],['3_10'],['10_20'],['20_30'],['30_50'],
#              ['大于50']]


def rank_variable(df,target_var,order_list,new_var,pattern_nan= '不清楚'):
    res = df.copy(deep = True)
    flag = 0 
    ii = 0
    for i in order_list:
        if np.nan in i:
            flag =1 
            var_index = ii
        ii = ii+1
    
    res[new_var] = np.nan
    for ii in res.index:
#        res[target_var]
        if pd.isna( res.loc[ii,target_var ] ):
            if(flag==0):
                pass
            else:
                res.loc[ii,new_var] = var_index
        else:
            if re.search(pattern_nan,res.loc[ii,target_var ]):
                pass
            else:
                for i in range(len(order_list)):
                    if res.loc[ii,target_var ] in order_list[i]:
                        res.loc[ii,new_var] = i
    return res
                
def map_results_keep_min(list_pattern,target_str):
    def tenp_return(x):
        if re.search(x[0],target_str):
            return 1 
        else:
            return 0
    res0 = list(map(tenp_return,list_pattern))
    res1 = [i for i in range(len(res0)) if res0[i]==1 ]
    if len(res1)==0:
        return np.nan
    else:
        return min(res1)
    

#map_results_keep_min(list_pattern = [['(无|未见).*色素环', np.nan], 
#                                      ['(少量|个别|散在).*色素环'], 
#                                      ['色素环|上皮角']],
#                     target_str = '色素环')

def rank_variable_perl(df,target_var,order_list,new_var,pattern_nan= '不清楚'):
    res = df.copy(deep = True)
    flag = 0 
    ii = 0
    for i in order_list:
        if np.nan in i:
            flag =1 
            var_index = ii
        ii = ii+1
    res[new_var] = np.nan
    for ii in res.index: 
        if pd.isna( res.loc[ii,target_var ] ):
            if(flag==0):
                pass
            else:
                res.loc[ii,new_var] = var_index
        else:
            if re.search(pattern_nan,res.loc[ii,target_var ]):
                pass
            else:
                for iii in range(len(order_list)): 
                    need_var = map_results_keep_min(list_pattern=order_list,
                                                    target_str=res.loc[ii,target_var ])
                    res.loc[ii,new_var] = need_var
#    res[new_var].value_counts(dropna = False)
    return res
    


    



#def Combine_var(df,target_var1,target_var2, new_var,
#                pattern_nan= '不清楚'):
#    res = df.copy(deep = True)
#    res[new_var] = np.nan
#    
#    for i in range(len(res[new_var])):
        
    
    
def combine_var(df,target_combine_list,new_var_name):
    res = df.copy(deep = True)
    def temp_(x):
        if(all(pd.isna(x))):
            return np.nan
        elif any(x>0):
            return 1
        else:
            return 0
    res[new_var_name] = res.loc[:,target_combine_list].apply(temp_,axis=1)
    res.drop(columns = target_combine_list,inplace = True)
    return res
#    res.loc[19,target_combine_list]
    

def Cut_var_two(df,target_var,new_var,cut_off,Nan_partain = '块缺失'):
    res= df.copy(deep = True)
    res[new_var] = np.nan 
    res.loc[res[target_var]==Nan_partain,target_var]=np.nan
    group1 = np.argwhere( res[target_var]<cut_off)
    group2 = np.argwhere(res[target_var] >=cut_off)
    res.loc[group1[:,0],new_var] =0
    res.loc[group2[:,0],new_var] =1
    
    res.loc[df[target_var]==Nan_partain,new_var] = np.nan
    return res
    
    
    
    
    
    
    



