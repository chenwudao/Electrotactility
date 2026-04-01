# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_props_jiehe_xghxg.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd

thr = 0.7
f = 'pearsonr_0.1'
df_= pd.read_excel(resolve_input_file('X_' + f + '_互相关性分析.xlsx'),index_col=0)
df_corr = pd.read_excel(resolve_input_file(f + '.xlsx'),index_col=0)

col_names = df_.columns
col_set_p = set()

for count, col_name in enumerate(col_names):
    for i in range(count):
        col_name_1 = col_names[i]
        correlation = df_.loc[col_name,col_name_1]
        if correlation >= thr: 
            if col_name_1 in col_set_p:
                continue
            else:
                corr1 = df_corr.loc[col_name_1,'pearsonr_corr']
                corr2 = df_corr.loc[col_name,'pearsonr_corr']
                if corr1 > corr2:  #比较皮尔森相关系数
                    col_set_p.add(col_name)
                    break
                else:
                    col_set_p.add(col_name_1)

df_ = pd.read_excel(resolve_input_file('X_' + f + '.xlsx'),index_col=0)
col_names = set(df_.columns.tolist())
out_f = df_.drop(list(col_names & col_set_p),axis = 1)

f = 'srocc_0.3_max_mic_0.1'
df_= pd.read_excel(resolve_input_file('X_' + f + '_互相关性分析_spe.xlsx'),index_col=0)
df_corr = pd.read_excel(resolve_input_file(f + '.xlsx'),index_col=0)

col_names = df_.columns
col_set = set()

for count, col_name in enumerate(col_names):
    for i in range(count):
        col_name_1 = col_names[i]
        correlation = df_.loc[col_name,col_name_1]
        if correlation >= thr:
            if col_name_1 in col_set:
                continue
            else:
                #spe优先
                corr1 = df_corr.loc[col_name_1,'spearman']
                corr2 = df_corr.loc[col_name,'spearman']
                if abs(corr1 - corr2) <= 0.05:
                    corr1 = df_corr.loc[col_name_1,'max_mic']
                    corr2 = df_corr.loc[col_name,'max_mic']
                    if abs(corr1 - corr2) <= 0.03:
                        corr1 = df_corr.loc[col_name_1,'srocc_corr']
                        corr2 = df_corr.loc[col_name,'srocc_corr']
                        if abs(corr1 - corr2) <= 0.05:
                            corr1 = df_corr.loc[col_name_1,'kendall']
                            corr2 = df_corr.loc[col_name,'kendall']
                            
                if corr1 > corr2:
                    col_set.add(col_name)
                    break
                else:
                    col_set.add(col_name_1)

df_ = pd.read_excel(resolve_input_file('X_' + f + '.xlsx'),index_col=0)
col_names = set(df_.columns.tolist())
col_set_p = set(out_f.columns.tolist())
col_set_drop = col_names & (col_set | col_set_p)
out_f_s = df_.drop(list(col_set_drop),axis = 1)

out_f = pd.concat([out_f,out_f_s],axis = 1)

out_f.to_excel(resolve_output_file('X_pea_sro_结合相关互相关_'+str(thr)+'.xlsx'))
