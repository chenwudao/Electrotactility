# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_select_mean.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
"""
df_= pd.read_excel(resolve_input_file('X_结合相关互相关_相关性分析.xlsx'),index_col=0)
col_names = set(df_.columns)
s_max_mic = set(df_.sort_values(axis = 1,by=['max_mic','spearman','srocc_corr','pearsonr_corr'],ascending = False).iloc[:,:50].columns)
s_spearman = set(df_.sort_values(axis = 1,by=['spearman','max_mic','srocc_corr','pearsonr_corr'],ascending = False).iloc[:,:50].columns)
s_srocc_corr = set(df_.sort_values(axis = 1,by=['srocc_corr','max_mic','spearman','pearsonr_corr'],ascending = False).iloc[:,:50].columns)
s_pearsonr_corr = set(df_.sort_values(axis = 1,by=['pearsonr_corr','max_mic','spearman','srocc_corr'],ascending = False).iloc[:,:50].columns)
s_rfr_importance = set(df_.sort_values(axis = 1,by=['rfr_importance','pearsonr_corr','max_mic','spearman','srocc_corr'],ascending = False).iloc[:,:100].columns)

s_nl = s_max_mic | s_spearman | s_srocc_corr | s_pearsonr_corr #求并集
s_l = set(df_.sort_values(axis = 1,by=['pearsonr_corr','max_mic','spearman','srocc_corr'],ascending = False).iloc[:,:len(s_nl)].columns.tolist())

s_all_selected = s_rfr_importance | s_nl | s_l

df_ = pd.read_excel(resolve_input_file('X_inbody_相关性分析.xlsx'),index_col=0)
col_names = set(df_.columns)

s_drop = col_names - s_all_selected
df_.drop(list(s_drop),axis = 1).to_excel(resolve_output_file('X_inbody_相关性分析_筛后.xlsx'))
"""
    
df_= pd.read_excel(resolve_input_file('X_纹理特征结果.xlsx'),index_col=0)
col_names = set(df_.columns)
df_= pd.read_excel(resolve_input_file('结果_补充.xlsx'),index_col=0)

s_drop = set(df_.columns) - col_names

df_.drop(list(s_drop),axis = 1).to_excel(resolve_output_file('结果_补充_筛后.xlsx'))
