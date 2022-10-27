# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:21:03 2019

@author: Mercuro
"""
#%reset -f
#package and function import
import pandas as pd
import numpy as np
from utils.A00_cleandata  import *

from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
import re
import copy

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
f = open('./CT和VISIA和皮肤镜clean_202008_used.csv')
Raw_data = pd.read_csv(f)


'''  data clean begain '''

#    pre-rosacea  玫瑰痤疮 大田痣 依照建议剔除
#    合并诊断含有玫瑰痤疮的也建议剔除

ss = show_crass_table(Raw_data,['诊断'],col_show = False)
cleaned_data = subset(df = Raw_data,target_var = '诊断',
                  check_list =['pre-rosacea', '玫瑰痤疮？','太田痣'],
                     inver = True)
cleaned_data = cleaned_data[pd.isna(cleaned_data['诊断'])==False]
#
#show_colnum = pd.DataFrame(cleaned_data.columns)
#
# 依照建议，合并疾病有玫瑰痤疮 剔除
ss = show_crass_table(Raw_data,['合并疾病'],col_show = False)
cleaned_data = subset(df = cleaned_data,target_var = '合并疾病',
                  check_list =['玫瑰痤疮／脂溢性皮炎','玫瑰痤疮','pre-rosacea？',
                               'pre-rosacea'],
                     inver = True)

#cleaned_data['诊断'].value_counts(dropna = False)
cleaned_data.drop(columns = ['采集部位_CT'],inplace = True)
cleaned_data.drop(columns = ['时间','姓名','皮肤镜编号',
                         'visa编号','编号','合并疾病'],inplace = True)
#
#cleaned_data.drop(columns = ['时间','皮肤镜编号',
#                         'visa编号','编号','合并疾病'],inplace = True)


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '诊断',
                          pattern_in1 = '玫瑰',
                         new_var =  '诊断_new')
cleaned_data.drop(columns = ['诊断'],inplace = True)
cleaned_data['诊断_new'].value_counts(dropna = False)

cleaned_data = cleaned_data.reset_index(drop=True)
#pd.crosstab(cleaned_data['性别'],cleaned_data['诊断_new'])

#cleaned_data['角质层_CT'] = factor(Vector = Raw_data['角质层_CT'] ,
#                                label_in = ['无','不规整','欠规整','规整',
#                                            '较规整'])

## 表示缺失状态  CT整缺  VISIA整缺  Skin整缺

loci_CT =cleaned_data.iloc[:,2:18].apply(lambda x:all(pd.isna(x)),axis=1) 
loci_Visa =cleaned_data.iloc[:,18:33].apply(lambda x:all(pd.isna(x)),axis=1)
loci_skin =cleaned_data.iloc[:,33:47].apply(lambda x:all(pd.isna(x)),axis=1)

cleaned_data['CT_new'] = 1
cleaned_data['Visa_new'] = 1
cleaned_data['skin_new'] = 1

cleaned_data.iloc[np.where(loci_CT)[0],48]  =0
cleaned_data.iloc[np.where(loci_Visa)[0],49] =0
cleaned_data.iloc[np.where(loci_skin)[0],50] =0

cleaned_data.iloc[np.where(loci_CT)[0],2:18] = '块缺失'
#cleaned_data.iloc[[10,16,196],2:18] = '块缺失'
cleaned_data.iloc[np.where(loci_Visa)[0],18:33] = '块缺失' 
cleaned_data.iloc[np.where(loci_skin)[0],33:47] = '块缺失' 

#
#皮肤镜：474
#CT：374
#Visia：421

#AB = set.union(set(np.where(loci_CT)[0]),set(np.where(loci_skin)[0])) # 175
#AC = set.union(set(np.where(loci_skin)[0]),set(np.where(loci_Visa)[0])) # 130
#BC = set.union(set(np.where(loci_CT)[0]),set(np.where(loci_Visa)[0])) # 130
#ALL_ = BC.union(set(np.where(loci_skin)[0]))


cleaned_data['性别'].value_counts(dropna=False)   ## 8人性别不详
cleaned_data['性别_new'] = factor(Vector = cleaned_data['性别'] ,label_in = ['女','男'])
cleaned_data['性别_new'].value_counts(dropna=False)

cleaned_data['角质层_CT'].value_counts(dropna = False)



cleaned_data['隆突部位有对称固定红斑'].value_counts(dropna = False)
cleaned_data['隆突部位肥大增生'].value_counts(dropna = False)
cleaned_data['隆突部位肥大增生'][cleaned_data['隆突部位肥大增生']==0] = '无'

cleaned_data['隆突部位有对称固定红斑_new'] = factor(Vector = cleaned_data['隆突部位有对称固定红斑'] ,
            label_in = ['无','有'])
cleaned_data['隆突部位肥大增生_new'] = factor(Vector = cleaned_data['隆突部位肥大增生'] ,
            label_in = ['无','有'])

#cleaned_data['分型'].value_counts(dropna = False)


#cleaned_data =rank_variable(df=cleaned_data,
#                                new_var = '角质层_CT_new',
#                                target_var = '角质层_CT',
#                         order_list = [['规整','无',np.nan],
#                                       ['较规整'],
#                                       ['欠规整'],
#                                       ['不规整'] ],
#                                       pattern_nan= '不清楚|块缺失')
#res_ = cleaned_data['角质层_CT_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]



cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '角质层_CT_new',
                                target_var = '角质层_CT',
                         order_list = [['较规整','规整','无',np.nan,'欠规整'],  
                                       ['不规整'] ],
                                       pattern_nan= '不清楚|块缺失')
res_ = cleaned_data['角质层_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]

#cleaned_data['角质层_CT_new'].value_counts(dropna=False)
#cleaned_data = ONEHOT(cleaned_data, '角质层_CT','角质层_CT')
#
#cleaned_data['角质层_CT_new'].value_counts(dropna = False) 

#cleaned_data['屏障表现_CT'].value_counts(dropna = False)
#cleaned_data =rank_variable(df=cleaned_data,
#                                new_var = '屏障表现_CT_new',
#                                target_var = '屏障表现_CT',
#                         order_list = [['边界清','无', np.nan],
#                                       ['边界较清'],
#                                       ['边界欠清'],
#                                       ['不清楚','边界不清']],
#                                       pattern_nan= '块缺失') 
##cleaned_data.loc[:,['角质层_CT_new','屏障表现_CT_new']].corr('spearman')
#cleaned_data['屏障表现_CT_new'].value_counts(dropna = False)
#res_ = cleaned_data['屏障表现_CT_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]


cleaned_data['屏障表现_CT'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '屏障表现_CT_new',
                                target_var = '屏障表现_CT',
                         order_list = [['边界较清','边界清','无', np.nan,'边界欠清',], 
                                       ['不清楚','边界不清']],
                                       pattern_nan= '块缺失') 
#cleaned_data.loc[:,['角质层_CT_new','屏障表现_CT_new']].corr('spearman')
cleaned_data['屏障表现_CT_new'].value_counts(dropna = False)
res_ = cleaned_data['屏障表现_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]

#cleaned_data['角质层_屏障表现'] = np.nan
#for i in  cleaned_data.index:
##    print(i)
#    if (cleaned_data.loc[i,'角质层_CT_new']==0 and cleaned_data.loc[i,'屏障表现_CT_new']==0):
#        cleaned_data.loc[i,'角质层_屏障表现'] =0
#    elif (cleaned_data.loc[i,'角质层_CT_new']==1 and cleaned_data.loc[i,'屏障表现_CT_new']==1):
#        cleaned_data.loc[i,'角质层_屏障表现'] =3
#    elif (cleaned_data.loc[i,'角质层_CT_new'] in [1,2]) and (cleaned_data.loc[i,'屏障表现_CT_new'] in [1,2]):
#        cleaned_data.loc[i,'角质层_屏障表现'] =1
#    else:
#        cleaned_data.loc[i,'角质层_屏障表现'] =2
#        
#cleaned_data['角质层_屏障表现'].value_counts(dropna = False)

#res_ = cleaned_data['角质层_屏障表现'].value_counts(dropna = False)
#res_[np.sort(res_.index)]

#表皮海绵水肿

cleaned_data['表皮海绵水肿_CT'].value_counts(dropna = False)

#df = cleaned_data
#target_var = '表皮海绵水肿_CT'
#prefix_var = '表皮海绵水肿_CT'
#Nan_label = '块缺失|NaN'


#cleaned_data = ONEHOT(df = cleaned_data, target_var = '表皮海绵水肿_CT',
#                      prefix_var = '表皮海绵水肿_CT_new')
#
#cleaned_data['表皮海绵水肿_CT'].value_counts(dropna = False)
#
##'表皮海绵水肿_CT_多灶性', '表皮海绵水肿_CT_局灶性', '表皮海绵水肿_CT_无', '表皮海绵水肿_CT_有',
##       '表皮海绵水肿_CT_nan'
#cleaned_data['表皮海绵水肿_CT_new_局灶性'].value_counts(dropna = False)
#
#cleaned_data['表皮海绵水肿_CT_new_有'].value_counts(dropna = False)
#cleaned_data['表皮海绵水肿_CT_new_多灶性'].value_counts(dropna = False)
#
#
#cleaned_data = combine_var(df = cleaned_data,
#                           target_combine_list =['表皮海绵水肿_CT_new_有','表皮海绵水肿_CT_new_多灶性'] ,
#                           new_var_name = '表皮海绵水肿_CT_new_有_多灶性')
#cleaned_data['表皮海绵水肿_CT_new_有_多灶性'].value_counts(dropna = False)
 
cleaned_data['表皮海绵水肿_CT'].value_counts(dropna = False)

cleaned_data =rank_variable(df = cleaned_data,
                                new_var = '表皮海绵水肿_CT_new',
                                target_var = '表皮海绵水肿_CT',
                         order_list = [['边界清','无', np.nan],
                                       ['局灶性'],
                                       ['多灶性','有']],
                                       pattern_nan= '块缺失') 
#cleaned_data.loc[:,['角质层_CT_new','屏障表现_CT_new']].corr('spearman')
cleaned_data['表皮海绵水肿_CT_new'].value_counts(dropna = False)
res_ = cleaned_data['表皮海绵水肿_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]



cleaned_data = Turn_to_NaN(df = cleaned_data,
                       target_var = '表皮深度_CT',
                       check_list = ['灶性','nan','无法测量','块缺失',
                                     '大于20',
                                     '深度无法测量',
                                     '无法测量深度','NaN','大于55',
                                     '41，起始部位过深无法测量',
                                     '85，34','60，96',
                                     '无法测量 49？',
                                     '大于61，无法测量深度',
                                     '大于60','大于48',
                                     '大于9 ',
                                     '-53',
                                     '大于9','大于49',
                                     '无'])

cleaned_data['表皮深度_CT_new'] = cleaned_data['表皮深度_CT']
#cleaned_data['表皮深度_CT'].describe()
##### 基底层变量

#cleaned_data['基底层色素改变_CT'].value_counts(dropna = False)
#cleaned_data = search_key_word2(df = cleaned_data,
#                           target_var = '基底层色素改变_CT',
#                          pattern_in1 = '色素环|上皮角',
#                          pattern_in2 = '(无|未见).*色素环',
#                         new_var =  '基底层_色素环有无_CT_new')
#cleaned_data['基底层_色素环有无_CT_new'].value_counts(dropna = False)
#


##cleaned_data['基底层色素改变_CT'].value_counts(dropna = False)
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '基底层色素改变_CT',
#                          pattern_in1 = '大量.*色素环',
#                         new_var =  '基底层_色素环大量_CT_new')
#cleaned_data['基底层_色素环大量_CT_new'].value_counts(dropna = False)
#
##cleaned_data = search_key_word1(df = cleaned_data,
##                           target_var = '基底层色素改变_CT',
##                          pattern_in1 = '少量.*色素环',
##                         new_var =  '基底层_色素环少量_CT')
##cleaned_data['基底层_色素环少量_CT'].value_counts(dropna = False)
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '基底层色素改变_CT',
#                          pattern_in1 = '(少量|个别|散在).*色素环',
#                         new_var =  '基底层_色素环少量_个别_CT_new')
#cleaned_data['基底层_色素环少量_个别_CT_new'].value_counts(dropna = False)
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '基底层色素改变_CT',
#                          pattern_in1 = '色素环.*破坏',
#                         new_var =  '基底层_色素环破坏_CT_new')
#cleaned_data['基底层_色素环破坏_CT_new'].value_counts(dropna = False)
#
###############
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '基底层色素改变_CT',
#                          pattern_in1 = '(折光|炎)',
#                         new_var =  '基底层_折光_CT_new')
#cleaned_data['基底层_折光_CT_new'].value_counts(dropna = False)


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '基底层色素改变_CT',
                          pattern_in1 = '(条索|树突)',
                         new_var =  '基底层_条索_CT_new')
cleaned_data['基底层_条索_CT_new'].value_counts(dropna = False)


cleaned_data['基底层色素改变_CT'].value_counts(dropna = False)
cleaned_data =rank_variable_perl(df=cleaned_data,
                                new_var = '基底层色素改变_CT_new',
                                target_var = '基底层色素改变_CT',
                         order_list = [['(无|未见).*色素环', np.nan],
                                       ['(少量|个别|散在).*色素环'],
                                       ['色素环|上皮角']],
                                       pattern_nan= '块缺失') 
#cleaned_data.loc[:,['角质层_CT_new','屏障表现_CT_new']].corr('spearman')
cleaned_data['基底层色素改变_CT_new'].value_counts(dropna = False)
res_ = cleaned_data['基底层色素改变_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]




cleaned_data['界面改变_CT'].value_counts(dropna = False)

cleaned_data =rank_variable_perl(df=cleaned_data,
                                new_var = '界面改变_CT_new',
                                target_var = '界面改变_CT',
                         order_list = [['无', np.nan],
                                       ['活跃|有'] ],
                                       pattern_nan= '块缺失') 
#cleaned_data.loc[:,['角质层_CT_new','屏障表现_CT_new']].corr('spearman')
cleaned_data['界面改变_CT_new'].value_counts(dropna = False)
res_ = cleaned_data['界面改变_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]

 

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '界面改变_CT',
#                          pattern_in1 = '活跃|有',
#                          pattern_nan = '不清楚|块缺失',
#                         new_var =  '界面改变_CT_有无_new')
#
#cleaned_data['界面改变_CT_有无_new'].value_counts(dropna = False) 
#
#cleaned_data = combine_var(df = cleaned_data,
#                           target_combine_list =['基底层_色素环破坏_CT_new',
#                                                 '基底层_折光_CT_new',
#                                                 '界面改变_CT_有无_new'] ,
#                           new_var_name = '基底色素环破坏_折光_界面改变_new')
#
#cleaned_data['基底色素环破坏_折光_界面改变_new'].value_counts(dropna = False)

 


#炎症细胞浸润


#cleaned_data = combine_var(df = cleaned_data,
#                           target_combine_list =['基底层_色素环个别_CT','基底层_色素环少量_CT'] ,
#                           new_var_name = '基底层_色素环少量_个别_CT')
#cleaned_data['基底层_色素环少量_个别_CT'].value_counts(dropna = False)






#受累毛囊/毛囊个数  浅层



cleaned_data = extract_number(df = cleaned_data,
                          target_var = '受累毛囊/毛囊个数_CT_sh',
                          pattern_nan = '未见毛囊结构|无|图像不清|不清楚|块缺失',
                          new_var = '受累毛囊_sh_new',
                          pattern_in=r'\d+',
                          count_=0) 
cleaned_data['受累毛囊_sh_new'].value_counts(dropna = False)

cleaned_data['受累毛囊/毛囊个数_CT_sh'].value_counts(dropna = False)

cleaned_data = extract_number(df = cleaned_data,
                          target_var = '受累毛囊/毛囊个数_CT_sh',
                          pattern_nan = '未见毛囊结构|无|图像不清|不清楚|块缺失',
                          new_var = '毛囊个数_sh_new',
                          pattern_in=r'\d+',
                          count_=1) 
cleaned_data['毛囊个数_sh_new'].value_counts(dropna = False)

cleaned_data = extract_number(df = cleaned_data,
                          target_var = '受累毛囊/毛囊个数_CT_dh',
                          pattern_nan = '未见毛囊结构|无|图像不清|不清楚|块缺失',
                          new_var = '受累毛囊_dh_new',
                          pattern_in=r'\d+',
                          count_=0) 
cleaned_data['受累毛囊_dh_new'].value_counts(dropna = False)

cleaned_data['受累毛囊/毛囊个数_CT_dh'].value_counts(dropna = False)

#cleaned_data['受累毛囊_CT_无有_sh_new'] = np.nan
#cleaned_data['受累毛囊_CT_无有_dh_new'] = np.nan
#
#
## =============================================================================
## =============================================================================
## # 
## =============================================================================
## =============================================================================
#for ii in cleaned_data.index:
#    if cleaned_data.loc[ii,'受累毛囊_dh_new'] >0:
#        cleaned_data.loc[ii,'受累毛囊_CT_无有_dh_new'] = 1
#    elif cleaned_data.loc[ii,'受累毛囊_dh_new'] ==0:
#        cleaned_data.loc[ii,'受累毛囊_CT_无有_dh_new'] = 0
#        
#    if cleaned_data.loc[ii,'受累毛囊_sh_new'] >0:
#        cleaned_data.loc[ii,'受累毛囊_CT_无有_sh_new'] = 1
#    elif cleaned_data.loc[ii,'受累毛囊_sh_new'] ==0:
#        cleaned_data.loc[ii,'受累毛囊_CT_无有_sh_new'] = 0

#cleaned_data['受累毛囊_CT_无有_sh'].value_counts(dropna = False)
cleaned_data = extract_number(df = cleaned_data,
                          target_var = '受累毛囊/毛囊个数_CT_dh',
                          pattern_nan = '未见毛囊结构|无|图像不清|不清楚|块缺失',
                          new_var = '毛囊个数_dh_new',
                          pattern_in=r'\d+',
                          count_=1) 
cleaned_data['毛囊个数_dh_new'].value_counts(dropna = False)



#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '受累毛囊/毛囊个数_CT_dh',
#                          pattern_in1 = '无|0',
#                          pattern_nan = '看不清|不清楚|不显著|可',
#                         new_var =  '受累毛囊_CT_无有_dh')
# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================
#cleaned_data['浅层深层都有_没_new'] = np.nan
#
#for i in cleaned_data.index:
#    if cleaned_data.loc[i,'受累毛囊_dh_new'] >0 and cleaned_data.loc[i,'受累毛囊_sh_new'] >0:
#        cleaned_data.loc[i,'浅层深层都有_没_new'] = 1
##    elif : 
##        cleaned_data.loc[i,'浅层深层都有'] = 0
#        
#    if cleaned_data.loc[i,'受累毛囊_dh_new']==0 and cleaned_data.loc[i,'受累毛囊_sh_new'] ==0:
#        cleaned_data.loc[i,'浅层深层都有_没_new'] = 0
##    else: 
##        cleaned_data.loc[i,'浅层深层都没有'] = 0   
# 
#                
#cleaned_data['浅层深层都有_没_new'].value_counts(dropna = False) 

#cleaned_data['单个毛囊最多毛囊虫数目_CT_sh']

cleaned_data['浅层受累毛囊百分比_new'] =  cleaned_data['受累毛囊_sh_new']/cleaned_data['毛囊个数_sh_new']
cleaned_data['深层受累毛囊百分比_new'] =  cleaned_data['受累毛囊_dh_new']/cleaned_data['毛囊个数_dh_new']

cleaned_data = cleaned_data.drop(columns = ['受累毛囊_sh_new','受累毛囊_dh_new',
                             '毛囊个数_sh_new','毛囊个数_dh_new'])

#for i in cleaned_data.index:
#    if pd.isna( cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_sh'] ):
#        continue
#    else:
#        print(cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_sh'])
#        res_temp = re.search( r'/|月',cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_sh'] )
#        if res_temp:
#            temp_res_num = re.findall(r'\d*',cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_sh'])
#            temp_num = []
#            for ii in temp_res_num:
#                if ii !='':
#                    temp_num.append(int(ii)) 
#            cleaned_data.loc[i,'浅层受累毛囊百分比'] = temp_num[0]/temp_num[1]
#
##cleaned_data['浅层受累毛囊百分比'].value_counts(dropna = False)
#
##cleaned_data['受累毛囊/毛囊个数_CT_sh'].value_counts(dropna = False)
#
#
#
#cleaned_data['深层受累毛囊百分比'] = np.nan
#
#for i in cleaned_data.index:
#    if pd.isna( cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_dh'] ):
#        continue
#    else:
##        print(cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_dh'])
#        res_temp = re.search( r'/|月',cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_dh'] )
#        if res_temp:
#            temp_res_num = re.findall(r'\d*',cleaned_data.loc[i,'受累毛囊/毛囊个数_CT_dh'])
#            temp_num = []
#            for ii in temp_res_num:
#                if ii !='':
#                    temp_num.append(int(ii)) 
#            cleaned_data.loc[i,'深层受累毛囊百分比'] = temp_num[0]/temp_num[1]
#
#cleaned_data['深层受累毛囊百分比'].value_counts(dropna = False)

cleaned_data['单个毛囊最多毛囊虫数目_CT_sh'].describe()

cleaned_data['单个毛囊最多毛囊虫数目_CT_sh'].value_counts(dropna = False)

cleaned_data  = Cut_var_two( df = cleaned_data ,
                                target_var = '单个毛囊最多毛囊虫数目_CT_sh',
                                cut_off = 3,
                                new_var = '单个毛囊最多毛囊虫数目_CT_sh_new')
cleaned_data['单个毛囊最多毛囊虫数目_CT_sh_new'].value_counts(dropna = False)
cleaned_data['单个毛囊最多毛囊虫数目_CT_dh'].describe()
cleaned_data  = Cut_var_two( df = cleaned_data ,
                                target_var = '单个毛囊最多毛囊虫数目_CT_dh',
                                cut_off = 3 ,
                                new_var = '单个毛囊最多毛囊虫数目_CT_dh_new' )

cleaned_data['单个毛囊最多毛囊虫数目_CT_dh_new'].value_counts(dropna = False)


cleaned_data['围绕毛囊_CT'].value_counts(dropna=False)

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '围绕毛囊_CT',
#                          pattern_in1 = '有',
#                          pattern_nan = '不清楚|块缺失',
#                         new_var =  '围绕毛囊_CT_有无_new')
#
#cleaned_data['围绕毛囊_CT_有无_new'].value_counts(dropna = False)


#cleaned_data = ONEHOT(cleaned_data, '血管数量_CT','血管数量_CT')

cleaned_data['血管数量_CT'].value_counts(dropna = False)

cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '血管数量_CT_new',
                                target_var = '血管数量_CT',
                         order_list = [['个别'],
                                       ['数条','少量'],
                                       ['较多'],
                                       ['大量'] ],
                                       pattern_nan= '不清楚|块缺失')

cleaned_data['血管数量_CT_new'].value_counts(dropna = False)
res_ = cleaned_data['血管数量_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]
#血管形态_CT

cleaned_data = search_key_word1(   df = cleaned_data,
                           target_var = '血管形态_CT',
                          pattern_in1 = '点状|点球状|球状|肾小球样',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_CT_点状_new')
res_ = cleaned_data['血管形态_CT_点状_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]

cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '血管形态_CT',
                          pattern_in1 = '线状|短线状|棒状',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_CT_线状_new')
res_ = cleaned_data['血管形态_CT_线状_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '血管形态_CT',
#                          pattern_in1 = '长线状',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '血管形态_CT_长线_new')
#res_ = cleaned_data['血管形态_CT_长线_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '血管形态_CT',
                          pattern_in1 = '细|纤细',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_CT_细_new')
res_ = cleaned_data['血管形态_CT_细_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '血管形态_CT',
                          pattern_in1 = '粗|粗大',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_CT_粗_new')
res_ = cleaned_data['血管形态_CT_粗_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]



#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '血管形态_CT',
#                          pattern_in1 = '分枝状',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '血管形态_CT_分枝状_new')
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '血管形态_CT',
#                          pattern_in1 = '弯曲|环',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '血管形态_CT_弯曲_new')
#res_ = cleaned_data['血管形态_CT_弯曲_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#res_ = cleaned_data['血管形态_CT_分枝状_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]





#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '血管形态_CT',
#                          pattern_in1 = '线状|短线状|棒状',
#                          pattern_nan = '图像不清',
#                         new_var =  '血管形态_CT_线状')
#res_ = cleaned_data['血管形态_CT_线状'].value_counts(dropna = False)
#res_[np.sort(res_.index)]




cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '血管形态_CT',
                          pattern_in1 = '分|多角形|网格',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_CT_分枝状_new')
res_ = cleaned_data['血管形态_CT_分枝状_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]

cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '血管形态_CT',
                          pattern_in1 = '弯曲|环',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_CT_弯曲_new')
res_ = cleaned_data['血管形态_CT_弯曲_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]



cleaned_data['血管形态_CT_线状_非弯曲_非分支_new'] = np.nan
for ii in cleaned_data.index:
    if cleaned_data.loc[ii,'血管形态_CT_线状_new'] ==1 and cleaned_data.loc[ii,'血管形态_CT_弯曲_new'] ==0 and cleaned_data.loc[ii,'血管形态_CT_分枝状_new']:
        cleaned_data.loc[ii,'血管形态_CT_线状_非弯曲_非分支_new'] =1
    elif all([np.isnan(cleaned_data.loc[ii,'血管形态_CT_线状_new'] ) , np.isnan(cleaned_data.loc[ii,'血管形态_CT_弯曲_new']) ] ):
        cleaned_data.loc[ii,'血管形态_CT_线状_非弯曲_非分支_new'] = np.nan
    else:
        cleaned_data.loc[ii,'血管形态_CT_线状_非弯曲_非分支_new'] =0

res_ = cleaned_data['血管形态_CT_线状_非弯曲_非分支_new'].value_counts(dropna=False)
res_[np.sort(res_.index)]



cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '血管形态_CT',
                          pattern_in1 = '成网|网格|网|多角',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_CT_网格状_多角_new')
res_ = cleaned_data['血管形态_CT_网格状_多角_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '血管形态_CT',
#                          pattern_in1 = '多角形',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '血管形态_CT_多角形_new')
#res_ = cleaned_data['血管形态_CT_多角形_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]




#cleaned_data = combine_var(df = cleaned_data,
#                           target_combine_list =['血管形态_CT_网格状_new','血管形态_CT_多角形_new'] ,
#                           new_var_name = '血管形态_CT_多角形_网格状_new')
#
#
#res_ = cleaned_data['血管形态_CT_多角形_网格状_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]

 


cleaned_data['血管扩张程度_CT'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '血管扩张程度_CT_new',
                                target_var = '血管扩张程度_CT',
                         order_list = [['无扩张'],
                                       ['稍扩张'],
                                       ['扩张'],
                                       ['显著扩张']],
                                       pattern_nan= '不清楚|块缺失')
res_ = cleaned_data['血管扩张程度_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]


cleaned_data['血流速度_CT'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '血流速度_CT_new',
                                target_var = '血流速度_CT',
                         order_list = [['较慢'],
                                       ['慢'],
                                       ['一般'],
                                       ['稍快'],
                                       ['快']],
                                       pattern_nan= '不清楚|块缺失')
 
res_ = cleaned_data['血流速度_CT_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]



cleaned_data['额部_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '额部_VISIA_new',
                                target_var = '额部_CISIA',
                         order_list = [['无',np.nan],
                                       ['轻'],
                                       ['中','重']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['额部_VISIA_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]




cleaned_data['眉间_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '眉间_VISIA_new',
                                target_var = '眉间_CISIA',
                         order_list = [['无',np.nan],
                                       ['轻'],
                                       ['中','重']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['眉间_VISIA_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]


cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '眉弓_VISIA_new',
                                target_var = '眉弓_CISIA',
                         order_list = [['无',np.nan],
                                       ['轻'],
                                       ['中','重']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['眉弓_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]



cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '面颊_VISIA_new',
                                target_var = '面颊_CISIA',
                         order_list = [['无',np.nan],
                                       ['轻'],
                                       ['中','重']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['面颊_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]



cleaned_data['口周_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '口周_VISIA_new',
                                target_var = '口周_CISIA',
                         order_list = [['无',np.nan],
                                       ['轻'],
                                       ['中','重']],
                                       pattern_nan= '不清楚|块缺失')
 
res_ = cleaned_data['口周_VISIA_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]




cleaned_data['仅有颏部_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '仅有颏部_VISIA_new',
                                target_var = '仅有颏部_CISIA',
                         order_list = [ ['无',np.nan],
                                       ['轻','中','重'] ],
                                       pattern_nan= '不清楚|块缺失')
res_ = cleaned_data['仅有颏部_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]


cleaned_data['眶周_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '眶周_VISIA_new',
                                target_var = '眶周_CISIA',
                         order_list = [['无',np.nan],
                                       ['轻'],
                                       ['中','重']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['眶周_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]



cleaned_data.loc[list(pd.isna(cleaned_data['眼睑血管_CISIA'])),'眼睑血管_CISIA'] = '无'

cleaned_data.loc[cleaned_data['眼睑血管_CISIA']=='块缺失','眼睑血管_CISIA'] = np.nan

cleaned_data['血管显露_VISIA_temp'] = factor(Vector = cleaned_data['眼睑血管_CISIA'] ,label_in = ['无','有'])
res_ = cleaned_data['血管显露_VISIA_temp'].value_counts(dropna = False)
res_[np.sort(res_.index)]


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '其他_Skin',
                          pattern_in1 = '血管',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '其他_Skin_血管_temp')
#show_res = cleaned_data[cleaned_data['其他_Skin_眼外侧']==1]
cleaned_data['其他_Skin_血管_temp'].value_counts(dropna = False)


cleaned_data = combine_var(df = cleaned_data,
                           target_combine_list =['其他_Skin_血管_temp',
                                                 '血管显露_VISIA_temp'] ,
                           new_var_name = '血管显露_VISIA_new')

res_ = cleaned_data['血管显露_VISIA_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]



#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '血管形态_CT',
#                          pattern_in1 = '成网|网格|网|多角',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '血管形态_CT_网格状_多角_new')
#res_ = cleaned_data['血管形态_CT_网格状_多角_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]














#cleaned_data.loc[list(pd.isna(cleaned_data['眼睑血管_CISIA'])),'眼睑血管_CISIA'] = '无'
#
#cleaned_data.loc[cleaned_data['眼睑血管_CISIA']=='块缺失','眼睑血管_CISIA'] = np.nan
#
#cleaned_data['眼睑血管_VISIA_new'] = factor(Vector = cleaned_data['眼睑血管_CISIA'] ,label_in = ['无','有'])
#
#res_ = cleaned_data['眼睑血管_VISIA_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]


cleaned_data['红斑整体程度_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '红斑整体程度_VISIA_new',
                                target_var = '红斑整体程度_CISIA',
                         order_list = [['无'],
                                       ['轻'],
                                       ['中'],
                                       ['重']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['红斑整体程度_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]


cleaned_data['丘疹_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '丘疹_VISIA_new',
                                target_var = '丘疹_CISIA',
                         order_list = [['无'],
                                       ['数个'],
                                       ['少量'],
                                       ['中等'],
                                       ['大量']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['丘疹_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index, )]



cleaned_data['脓疱_CISIA'].value_counts(dropna = False)
cleaned_data = rank_variable(df=cleaned_data,
                         new_var = '脓疱_VISIA_new',
                         target_var = '脓疱_CISIA',
                         order_list = [['无'],
                                       ['1','2'],
                                       ['3','3_10','10_20','20_30','30_50', '大于50']],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['脓疱_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]


cleaned_data['鼻部_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '鼻部_VISIA_new',
                                target_var = '鼻部_CISIA',
                         order_list = [['无'],
                                       ['轻'],
                                       ['中','重'] ],
                                       pattern_nan= '不清楚|块缺失')

res_ = cleaned_data['鼻部_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]


cleaned_data['耳前_CISIA'].value_counts(dropna = False)
cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '耳前_VISIA_new',
                                target_var = '耳前_CISIA',
                         order_list = [['无'],
                                       ['轻','中','重'] ],
                                       pattern_nan= '无法辨别|块缺失')

res_ = cleaned_data['耳前_VISIA_new'].value_counts(dropna = False) 
res_[np.sort(res_.index)]



#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '备注_CISIA',
#                          pattern_in1 = '血管',
#                          pattern_nan = '图像不清',
#                         new_var =  '备注_VISIA')
#
#res_ = cleaned_data['备注_CISIA'].value_counts(dropna = False)
#res_ = cleaned_data['备注_VISIA'].value_counts(dropna = False)
#res_[np.sort(res_.index)]

#cleaned_data['备注_VISIA'].value_counts(dropna = False)


'''
    skin
'''



cleaned_data['背景_Skin'].value_counts(dropna = False)
cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '背景_Skin',
                          pattern_in1 = '黄|棕',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '背景_Skin_黄_new')

cleaned_data['背景_Skin_黄_new'].value_counts(dropna = False)


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '背景_Skin',
                          pattern_in1 = '白',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '背景_Skin_白_new')
cleaned_data['背景_Skin_白_new'].value_counts(dropna = False)

#show_res = cleaned_data[cleaned_data['背景_Skin']==1]
#show_res['背景_Skin_白_new'].value_counts(dropna = False)


#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '背景_Skin',
#                          pattern_in1 = '粉',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '背景_Skin_粉_new')
#cleaned_data['背景_Skin_粉_new'].value_counts(dropna = False)

#show_res = cleaned_data[cleaned_data['背景_Skin_粉']==1]
#show_res['背景_Skin'].value_counts(dropna = False)


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '背景_Skin',
#                          pattern_in1 = '[((?!白).*(粉|红|紫))|((粉|红|紫).*(?!白))]',
                            pattern_in1 = '(粉|红|紫)',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '背景_Skin_粉红紫_new')
cleaned_data['背景_Skin_粉红紫_new'].value_counts(dropna = False)
#
#show_res = cleaned_data[cleaned_data['背景_Skin_粉红紫']==1]
#show_res['背景_Skin'].value_counts(dropna = False)


#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '背景_Skin',
##                          pattern_in1 = '[((?!白).*(粉|红|紫))|((粉|红|紫).*(?!白))]',
#                            pattern_in1 = '白',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '背景_Skin_白_new')
#cleaned_data['背景_Skin_白_new'].value_counts(dropna = False)

#show_res = cleaned_data[cleaned_data['背景_Skin_白']==1]
#show_res['背景_Skin'].value_counts(dropna = False)


cleaned_data['背景_Skin_粉红紫_非白_new'] = np.nan
for ii in cleaned_data.index :
    if cleaned_data.loc[ii,'背景_Skin_白_new'] == 1 :
        cleaned_data.loc[ii,'背景_Skin_粉红紫_非白_new'] ==0
    else:
        cleaned_data.loc[ii,'背景_Skin_粉红紫_非白_new'] = cleaned_data.loc[ii,'背景_Skin_粉红紫_new']
       

show_res = cleaned_data[cleaned_data['背景_Skin_粉红紫_非白_new']==1]
show_res['背景_Skin'].value_counts(dropna = False)

cleaned_data['背景_Skin_粉红紫_非白_new'].value_counts(dropna = False)
#########################################################################################

cleaned_data = cleaned_data.drop(columns = '背景_Skin_粉红紫_new')


#show_res.shape
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '背景_Skin',
#                          pattern_in1 = '(粉).*白',
#                          pattern_nan = '图像不清',
#                         new_var =  '背景_Skin_粉白')
#cleaned_data['背景_Skin_粉白'].value_counts(dropna = False)
#
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '背景_Skin',
#                          pattern_in1 = '红|暗红',
#                          pattern_nan = '图像不清',
#                         new_var =  '背景_Skin_红')
#cleaned_data['背景_Skin_红'].value_counts(dropna = False)
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '背景_Skin',
#                          pattern_in1 = '紫',
#                          pattern_nan = '图像不清',
#                         new_var =  '背景_Skin_紫')
#
#cleaned_data['背景_Skin_紫'].value_counts(dropna = False)





cleaned_data['分布_Skin'].value_counts(dropna = False)

cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '分布_Skin',
                          pattern_in1 = '弥漫均匀',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '分布_Skin_弥散均匀_new')
cleaned_data['分布_Skin_弥散均匀_new'].value_counts(dropna = False)

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '分布_Skin',
#                          pattern_in1 = '弥漫不均|弥漫不均匀',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '分布_Skin_弥漫不均_new')
#cleaned_data['分布_Skin_弥漫不均_new'].value_counts(dropna = False)
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '分布_Skin',
#                          pattern_in1 = '多灶性不均',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '分布_Skin_多灶性不均_new')
#cleaned_data['分布_Skin_多灶性不均_new'].value_counts(dropna = False)
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '分布_Skin',
#                          pattern_in1 = '局灶不均|局灶性不均',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '分布_Skin_局灶不均_new')
#cleaned_data['分布_Skin_局灶不均_new'].value_counts(dropna = False)

 

cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '分布_Skin',
                          pattern_in1 = '不均|局灶性不均|多灶性不均|弥漫不均|弥漫不均匀',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '分布_skin_不均_new')
cleaned_data['分布_skin_不均_new'].value_counts(dropna = False)

#cleaned_data['分布_Skin'].value_counts(dropna = False)


cleaned_data['粗细_Skin'].value_counts(dropna = False)
cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '粗细_Skin',
                          pattern_in1 = '粗细不均|粗细不等|部分细小部分中等',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '粗细_Skin_粗细不均_new')
cleaned_data['粗细_Skin_粗细不均_new'].value_counts(dropna = False)


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '粗细_Skin',
                          pattern_in1 = '较粗|稍粗|稍粗大|较粗大',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '粗细_Skin_粗大_new')
cleaned_data['粗细_Skin_粗大_new'].value_counts(dropna = False)


#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '粗细_Skin',
#                          pattern_in1 = '中等',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '粗细_Skin_中等_new')
#cleaned_data['粗细_Skin_中等_new'].value_counts(dropna = False)


#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '粗细_Skin',
#                          pattern_in1 = '较细|稍细|纤细|细小',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '粗细_Skin_细_new')
#cleaned_data['粗细_Skin_细_new'].value_counts(dropna = False)
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '粗细_Skin',
#                          pattern_in1 = '模糊|点状|细碎|纤细细碎',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '粗细_Skin_点状_new')
#
#cleaned_data['粗细_Skin_点状_new'].value_counts(dropna = False)
 


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '粗细_Skin',
                          pattern_in1 = '较细|稍细|纤细|细小|模糊|点状|细碎|纤细细碎',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '粗细_Skin_细_点状_new')
cleaned_data['粗细_Skin_细_点状_new'].value_counts(dropna = False)


  
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

#
#cleaned_data['分支_Skin'].value_counts(dropna=False)
cleaned_data['苯环_Skin'].value_counts(dropna = False)
cleaned_data.loc[list(pd.isna(cleaned_data['苯环_Skin'])),'苯环_Skin'] = '无'

cleaned_data.loc[ cleaned_data['苯环_Skin']=='块缺失','苯环_Skin'] = np.nan 

cleaned_data['苯环_Skin_new']= factor(cleaned_data['苯环_Skin'],label_in=['无','有'])
cleaned_data['苯环_Skin_new'].value_counts(dropna=False)

cleaned_data['分支_Skin'].value_counts(dropna = False)
cleaned_data.loc[list(pd.isna(cleaned_data['分支_Skin'])),'分支_Skin'] = '无'


cleaned_data = rank_variable(df=cleaned_data,
                                new_var = '分支_Skin_new',
                                target_var = '分支_Skin',
                         order_list = [['无'],
                                       ['有'] ],
                                       pattern_nan= '不清楚|块缺失')
cleaned_data['分支_Skin_new'].value_counts(dropna=False)

#cleaned_data['分支_Skin'].value_counts(dropna = False)
cleaned_data['形态_Skin'].value_counts(dropna = False)


cleaned_data['网格_Skin'].value_counts(dropna = False)
cleaned_data.loc[list(pd.isna(cleaned_data['网格_Skin'])),'网格_Skin'] = '无'

cleaned_data =rank_variable(df=cleaned_data,
                                new_var = '网格_Skin_new',
                                target_var = '网格_Skin',
                         order_list = [['无'],
                                       ['有','小','不完全','大'] ],
                                       pattern_nan= '不清楚|块缺失')
 
res_ = cleaned_data['网格_Skin_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]



cleaned_data = combine_var(df = cleaned_data,
                           target_combine_list =['网格_Skin_new',
                                                 '分支_Skin_new'] ,
                           new_var_name = '苯环_网格_Skin_new')

cleaned_data['苯环_网格_Skin_new'].value_counts(dropna = False)




cleaned_data = rank_variable(df=cleaned_data,
                                new_var = '分支_Skin_new',
                                target_var = '分支_Skin',
                         order_list = [['无'],
                                       ['有'] ],
                                       pattern_nan= '不清楚|块缺失')
cleaned_data['分支_Skin_new'].value_counts(dropna=False)

#'围绕丘疹_Skin', '围绕脓疱_Skin', '围绕毛囊_Skin', '毛囊_Skin', '脓疱_Skin'


# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================
cleaned_data['围绕丘疹_Skin'].value_counts(dropna = False)
cleaned_data.loc[list(pd.isna(cleaned_data['围绕丘疹_Skin'])),'围绕丘疹_Skin'] = 0
cleaned_data.loc[cleaned_data['围绕丘疹_Skin']=='块缺失','围绕丘疹_Skin'] = np.nan
cleaned_data['围绕丘疹_Skin_temp']= factor(cleaned_data['围绕丘疹_Skin'],label_in=['无','有'])

cleaned_data['围绕脓疱_Skin'].value_counts(dropna = False) 
cleaned_data.loc[list(pd.isna(cleaned_data['围绕脓疱_Skin'])),'围绕脓疱_Skin'] = 0
cleaned_data.loc[cleaned_data['围绕脓疱_Skin']=='块缺失','围绕脓疱_Skin'] = np.nan
cleaned_data['围绕脓疱_Skin_temp']= factor(cleaned_data['围绕脓疱_Skin'],label_in=['无','有'])

cleaned_data['围绕毛囊_Skin'].value_counts(dropna = False)
cleaned_data.loc[list(pd.isna(cleaned_data['围绕毛囊_Skin'])),'围绕毛囊_Skin'] = 0
cleaned_data.loc[cleaned_data['围绕毛囊_Skin']=='块缺失','围绕毛囊_Skin'] = np.nan
cleaned_data['围绕毛囊_Skin_temp']= factor(cleaned_data['围绕毛囊_Skin'],label_in=['无','有'])



#cleaned_data['围绕毛囊_Skin_new'].value_counts(dropna = False)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
cleaned_data = combine_var(df = cleaned_data,
                           target_combine_list =['围绕丘疹_Skin_temp',
                                                 '围绕脓疱_Skin_temp',
                                                 '围绕毛囊_Skin_temp'] ,
                           new_var_name = '围绕附属器_Skin_new')

cleaned_data['围绕附属器_Skin_new'].value_counts(dropna = False)


# =============================================================================
# 
# =============================================================================
 
#cleaned_data['数量_Skin'].value_counts(dropna = False)
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '数量_Skin',
#                          pattern_in1 = '致密|大量密集|大量|中等',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '数量_Skin_致密中等_new')
#cleaned_data['数量_Skin_致密中等_new'].value_counts(dropna = False)
#
##cleaned_data = search_key_word1(df = cleaned_data,
##                           target_var = '数量_Skin',
##                          pattern_in1 = '中等',
##                          pattern_nan = '图像不清',
##                         new_var =  '数量_Skin_中等')
##cleaned_data['数量_Skin_中等'].value_counts(dropna = False)
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '数量_Skin',
#                          pattern_in1 = '稀疏',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '数量_Skin_稀疏_new')
#cleaned_data['数量_Skin_稀疏_new'].value_counts(dropna = False)
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '数量_Skin',
#                          pattern_in1 = '少量|个别',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '数量_Skin_少量_new')
#cleaned_data['数量_Skin_少量_new'].value_counts(dropna = False)

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '数量_Skin',
#                          pattern_in1 = '大量',
#                          pattern_nan = '图像不清',
#                         new_var =  '数量_Skin_大量')
#cleaned_data['数量_Skin_大量'].value_counts(dropna = False)

# =============================================================================
# 
# =============================================================================



# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

cleaned_data =rank_variable_perl(df=cleaned_data,
                                new_var = '数量_Skin_new',
                                target_var = '数量_Skin',
                         order_list = [['少量|个别', np.nan],
                                       ['稀疏'],
                                       ['致密|密集|大量|中等']],
                                       pattern_nan= '块缺失') 
cleaned_data['数量_Skin_new'].value_counts(dropna = False)


 

cleaned_data['毛囊_Skin'].value_counts(dropna = False)

cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '毛囊_Skin',
                          pattern_in1 = '油滴',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '毛囊_Skin_油滴_new')

cleaned_data['毛囊_Skin_油滴_new'].value_counts(dropna = False)

cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '毛囊_Skin',
                          pattern_in1 = '粉刺',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '毛囊_Skin_粉刺_new')

cleaned_data['毛囊_Skin_粉刺_new'].value_counts(dropna = False)


#cleaned_data['脓疱_Skin'].value_counts(dropna = False)
#cleaned_data =rank_variable(df=cleaned_data,
#                                new_var = '脓疱_Skin_new',
#                                target_var = '脓疱_Skin',
#                         order_list = [['无'],
#                                       ['个别','数个','个别脓疱'],
#                                       ['少量','少量脓疱'],
#                                       ['较多','大量','多，散在','较多脓疱']],
#                                       pattern_nan= '不清楚')
#
#res_ = cleaned_data['脓疱_Skin'].value_counts(dropna = False)
#res_ = cleaned_data['脓疱_Skin_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]



cleaned_data['脓疱_Skin'].value_counts(dropna = False)
cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '脓疱_Skin',
                          pattern_in1 = '个别|数个|个别脓疱|少量|少量脓疱|较多|大量|多，散在|较多脓疱',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '脓疱_Skin_new')
cleaned_data['脓疱_Skin_new'].value_counts(dropna = False)



#
#cleaned_data['其他_Skin'].value_counts(dropna = False)
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '其他_Skin',
#                          pattern_in1 = '脱屑|鳞屑|痂屑|碎屑',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '其他_Skin_屑_new')
#cleaned_data['其他_Skin_屑_new'].value_counts(dropna = False)

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '其他_Skin',
#                          pattern_in1 = '(眼外侧).*(紫色结构|蜂窝状结构|血管网)',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '其他_Skin_眼外侧_new')
##show_res = cleaned_data[cleaned_data['其他_Skin_眼外侧']==1]
#cleaned_data['其他_Skin_眼外侧_new'].value_counts(dropna = False)

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '其他_Skin',
#                          pattern_in1 = '(蜂窝状).*(紫色|血管)',
#                          pattern_nan = '图像不清',
#                         new_var =  '其他_Skin_蜂窝状')
#show_res = cleaned_data[cleaned_data['其他_Skin_蜂窝状']==1]
#cleaned_data['其他_Skin_蜂窝状'].value_counts(dropna = False)

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '其他_Skin',
#                          pattern_in1 = '(毛囊).*(红晕|红斑)',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '其他_Skin_毛囊_红晕_new')
#cleaned_data['其他_Skin_毛囊_红晕_new'].value_counts(dropna = False)
#
##show_res = cleaned_data[cleaned_data['其他_Skin_毛囊_红晕']==1]
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '其他_Skin',
#                          pattern_in1 = '(脓疱).*(红晕|红斑)',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '其他_Skin_脓疱_红晕_new')
#cleaned_data['其他_Skin_脓疱_红晕_new'].value_counts(dropna = False)

#show_res = cleaned_data[cleaned_data['其他_Skin_脓疱_红晕']==1]

#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '其他_Skin',
#                          pattern_in1 = '(点状).*(血管)',
#                          pattern_nan = '图像不清',
#                         new_var =  '其他_Skin_点状血管')
#cleaned_data['其他_Skin_点状血管'].value_counts(dropna = False)

#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '其他_Skin',
#                          pattern_in1 = '(丘疹|脓疱).*(线状|纤细|放射)',
#                          pattern_nan = '图像不清',
#                         new_var =  '其他_Skin_丘脓周血管')
#cleaned_data['其他_Skin_丘脓周血管'].value_counts(dropna = False)


#show_res = cleaned_data[cleaned_data['其他_Skin_丘脓周血管']==1]

 

# =============================================================================
# cleaned_data['形态_Skin'].value_counts(dropna = False)
# =============================================================================




#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '线状|短线状|棒状',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_短线')
#res_ = cleaned_data['形态_Skin_短线'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '长线状',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_长线')
#res_ = cleaned_data['形态_Skin_长线'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '细|纤细',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_细')
#res_ = cleaned_data['形态_Skin_细'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '粗|粗大',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_粗')
#res_ = cleaned_data['形态_Skin_粗'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '弯曲|环',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_弯曲')
#res_ = cleaned_data['形态_Skin_弯曲'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '分枝状',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_分枝状')
#res_ = cleaned_data['形态_Skin_分枝状'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '成网|网格状|网',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_网格状')
#res_ = cleaned_data['形态_Skin_网格状'].value_counts(dropna = False)
#res_[np.sort(res_.index)]
#
#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '多角形',
#                          pattern_nan = '图像不清',
#                         new_var =  '形态_Skin_多角形')
#res_ = cleaned_data['形态_Skin_多角形'].value_counts(dropna = False)
#res_[np.sort(res_.index)]


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '形态_Skin',
                          pattern_in1 = '点状|点球状|球状|肾小球样',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_Skin_点状_new')
res_ = cleaned_data['血管形态_Skin_点状_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]


#cleaned_data['形态_Skin'].value_counts(dropna = False)
cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '形态_Skin',
                          pattern_in1 = '线状|短线状|棒状|分',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_Skin_线状_temp')
res_ = cleaned_data['血管形态_Skin_线状_temp'].value_counts(dropna = False)
res_[np.sort(res_.index)]


cleaned_data = search_key_word1(df = cleaned_data,
                           target_var = '形态_Skin',
                          pattern_in1 = '弯|环',
                          pattern_nan = '图像不清|块缺失',
                         new_var =  '血管形态_Skin_弯曲_new')
res_ = cleaned_data['血管形态_Skin_弯曲_new'].value_counts(dropna = False)
res_[np.sort(res_.index)]





#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '分',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '血管形态_Skin_分支_new')
#res_ = cleaned_data['血管形态_Skin_分支_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]

#分支_Skin_new

cleaned_data['血管形态_Skin_线状_非弯曲_非分支_new'] = np.nan
for ii in cleaned_data.index:
    if cleaned_data.loc[ii,'血管形态_Skin_线状_temp'] ==1 and cleaned_data.loc[ii,'血管形态_Skin_弯曲_new'] ==0 and cleaned_data.loc[ii,'分支_Skin_new'] ==0:
        cleaned_data.loc[ii,'血管形态_Skin_线状_非弯曲_非分支_new'] =1
    elif all([np.isnan(cleaned_data.loc[ii,'血管形态_Skin_线状_temp'] ) , np.isnan(cleaned_data.loc[ii,'血管形态_Skin_弯曲_new']) ] ):
        cleaned_data.loc[ii,'血管形态_Skin_线状_非弯曲_非分支_new'] = np.nan
    else:
        cleaned_data.loc[ii,'血管形态_Skin_线状_非弯曲_非分支_new'] =0

res_ = cleaned_data['血管形态_Skin_线状_非弯曲_非分支_new'].value_counts(dropna=False)
res_[np.sort(res_.index)]



#cleaned_data = search_key_word1(df = cleaned_data,
#                           target_var = '形态_Skin',
#                          pattern_in1 = '分',
#                          pattern_nan = '图像不清|块缺失',
#                         new_var =  '血管形态_Skin_线状_分支_new')
#res_ = cleaned_data['血管形态_Skin_线状_分支_new'].value_counts(dropna = False)
#res_[np.sort(res_.index)]











#cleaned_data['其他_Skin_眼外侧_new'].value_counts(dropna = False)

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#cleaned_data['苯环_Skin'].value_counts(dropna = False)
#cleaned_data.loc[pd.isna(cleaned_data['苯环_Skin']),'苯环_Skin'] = 0
#cleaned_data.loc[cleaned_data['苯环_Skin']=='块缺失','苯环_Skin'] = np.nan
#cleaned_data['苯环_Skin_new'] = factor(Vector = cleaned_data['苯环_Skin'] ,label_in = ['无','有'])

#cleaned_data = combine_var(df = cleaned_data,
#                           target_combine_list =['其他_Skin_眼外侧_new','苯环_Skin_new'] ,
#                           new_var_name = '其他_Skin_眼外侧_苯环_new')
#
#cleaned_data['其他_Skin_眼外侧_苯环_new'].value_counts(dropna = False)


# =============================================================================
# 
# =============================================================================

#
#col_num_need = ['诊断_new','年龄',
#                '其他_Skin_眼外侧_苯环',
#                '性别',
#                '角质层_CT_new',
#                '屏障表现_CT_new',
#                '角质层_屏障表现',
#                '表皮海绵水肿_CT_有_多灶性',
#                '表皮海绵水肿_CT_局灶性',
#                '表皮深度_CT',
#                '基底层_色素环有无_CT',
#                '基底层_色素环大量_CT',
#                '基底层_色素环少量_个别_CT',
#                '基底层_条索_CT',
#                '基底色素环破坏_折光_界面改变',
#                '受累毛囊_CT_无有_sh',
#                '浅层受累毛囊百分比',
#                '单个毛囊最多毛囊虫数目_CT_sh_new',
#                '受累毛囊_CT_无有_dh',
#                '深层受累毛囊百分比',
#                '浅层深层都没有',
#                '浅层深层都有',
#                '单个毛囊最多毛囊虫数目_CT_dh_new',
#                '围绕毛囊_CT_有无',
#                '血管数量_CT_new',
#                '血管形态_CT_点状',
#                '血管形态_CT_线状',
#                '血管形态_CT_弯曲_分枝状',
#                '血管形态_CT_线状_非弯曲',
#                '血管形态_CT_多角形_网格状',
#                '血管扩张程度_CT_new',
#                '血流速度_CT_new',
#                '额部_VISIA',
#                '眉间_VISIA',
#                '眉弓_VISIA',
#                '面颊_VISIA',
#                '口周_VISIA',
#                '仅有颏部_VISIA',
#                '眶周_VISIA',
#                '眼睑血管_VISIA',
#                '红斑整体程度_VISIA',
#                '丘疹_VISIA',
#                '脓疱_VISIA',
#                '鼻部_VISIA',
#                '耳前_VISIA',
##                '背景_Skin_白',
#                '背景_Skin_粉红紫_非白',
#                '分布_Skin_弥散均匀',
#                '分布_skin_不均',
#                '粗细_Skin_粗细不均',
#                '粗细_Skin_粗大',
#                '粗细_Skin_中等',
#                '粗细_Skin_细',
#                '粗细_Skin_点状',
#                '苯环_Skin',
#                '分支_Skin_new',
#                '网格_Skin_new',
#                '围绕附属器_Skin',
#                '数量_Skin_致密中等',
#                '数量_Skin_稀疏',
#                '数量_Skin_少量',
#                '毛囊_Skin_油滴',
#                '毛囊_Skin_粉刺',
#                '脓疱_Skin_new',
#                '其他_Skin_屑',
#                '其他_Skin_眼外侧',
#                '其他_Skin_毛囊_红晕',
#                '其他_Skin_脓疱_红晕',
#                '血管形态_Skin_点状',
#                '血管形态_Skin_线状',
#                '血管形态_Skin_弯曲_分枝状',
#                '血管形态_Skin_线状_非弯曲']


#cleaned_data_new = cleaned_data.loc[:,col_num_need]
#
#cleaned_data_new['围绕附属器_Skin'].value_counts(dropna=False)

#
#cleaned_data.drop(columns = ['背景_Skin','数量_Skin','粗细_Skin','网格_Skin','毛囊_Skin',
#                             '脓疱_Skin','其他_Skin','脓疱_CISIA',
#                             '围绕毛囊_CT','形态_Skin',
#                             '角质层_CT','额部_CISIA','眉间_CISIA','眶周_CISIA',
#                             '部位_CISIA','鼻部_CISIA','耳前_CISIA','备注_CISIA',
#                             '眉弓_CISIA','面颊_CISIA','口周_CISIA','仅有颏部_CISIA',
#                             '红斑整体程度_CISIA','眼睑血管_CISIA','丘疹_CISIA',
#                             '血流速度_CT',
#                             '分布_Skin',
#                             '血管扩张程度_CT',
#                             '血管形态_CT',
#                             '屏障表现_CT',
#                             '基底层色素改变_CT',
#                             '炎症细胞浸润_CT',
#                             '界面改变_CT',
#                             '受累毛囊/毛囊个数_CT_sh',
#                             '受累毛囊/毛囊个数_CT_dh',
#                             '血管数量_CT' ],inplace = True)

#cleaned_data['性别_new'] = cleaned_data['性别'] 
cleaned_data['年龄_new'] = cleaned_data['年龄']
Var_list = cleaned_data.columns

Var_need_list =  list(map(lambda x: re.search('_new',x)!=None,Var_list))

cleaned_data_new = cleaned_data.loc[:,Var_need_list]
#面颊	口周	仅有颏部	眶周	眼睑血管	红斑整体程度	丘疹	脓疱	部位	鼻部	耳前




# 新加了三列变量

cleaned_data_new.to_csv('./Analysis_cleaned_data.csv',
                    index = False,encoding = 'utf_8_sig')



#cleaned_data['形态_Skin'].value_counts()

#
#Var_list = pd.DataFrame(cleaned_data_new.columns)
#Var_list.to_csv('delete.csv',index = False,encoding = 'utf_8_sig') 

#Var_need_list =  list(map(lambda x: re.search('VISIA',x)!=None,Var_list))
#sum(Var_need_list)
#
#
#Var_list = cleaned_data_new.columns
#Var_need_list =  list(map(lambda x: re.search('Skin',x)!=None,Var_list))
#sum(Var_need_list)
#
#Var_list = cleaned_data_new.columns
#Var_need_list =  list(map(lambda x: re.search('CT',x)!=None,Var_list))
#sum(Var_need_list)

 
