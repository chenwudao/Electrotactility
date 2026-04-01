# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_select.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd

#filelist = ['X','X_电阻抗谱数据','X_纹理特征结果']
filelist = ['X_结合相关互相关']
for f in filelist:
    df_= pd.read_excel(resolve_input_file(f +'_相关性分析.xlsx'), index_col=0)
    col_names = set(df_.columns)
    s_max_mic = set(df_.sort_values(axis = 1,by=['max_mic','spearman','srocc_corr','pearsonr_corr'],ascending = False).iloc[:,:50].columns)
    s_spearman = set(df_.sort_values(axis = 1,by=['spearman','max_mic','srocc_corr','pearsonr_corr'],ascending = False).iloc[:,:50].columns)
    s_srocc_corr = set(df_.sort_values(axis = 1,by=['srocc_corr','max_mic','spearman','pearsonr_corr'],ascending = False).iloc[:,:50].columns)
    s_pearsonr_corr = set(df_.sort_values(axis = 1,by=['pearsonr_corr','max_mic','spearman','srocc_corr'],ascending = False).iloc[:,:50].columns)
    s_rfr_importance = set(df_.sort_values(axis = 1,by=['rfr_importance','pearsonr_corr','max_mic','spearman','srocc_corr'],ascending = False).iloc[:,:100].columns)
    
    s_nl = s_max_mic | s_spearman | s_srocc_corr | s_pearsonr_corr
    s_l = set(df_.sort_values(axis = 1,by=['pearsonr_corr','max_mic','spearman','srocc_corr'],ascending = False).iloc[:,:len(s_nl)].columns.tolist())

    s_nl_rfr = s_rfr_importance & s_nl
    s_l_rfr = s_rfr_importance & s_l
    
    l_set = [s_l, s_nl, s_l_rfr, s_nl_rfr]
    name_list = ['_线性优先.xlsx','_非线性优先.xlsx','_线性优先_rfr.xlsx','_非线性优先_rfr.xlsx']
    
    df = pd.read_excel(resolve_input_file(f + '.xlsx'))
    
    for i in range(len(l_set)):
        s_drop = col_names - l_set[i]
        s_drop.remove('手指（优食-0 非优中-1 非优尾-2 非优食-3）')
        df_.drop(list(s_drop),axis = 1).to_excel(resolve_output_file(f + '_相关性分析' + name_list[i]))
        df.drop(list(s_drop),axis = 1).to_excel(resolve_output_file(f + name_list[i]))
