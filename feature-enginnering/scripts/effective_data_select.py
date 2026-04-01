# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：effective_data_select.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd

col_names = ['受试者编号','手指','手指（优食-0 非优中-1 非优尾-2 非优食-3）']
#df_ = pd.read_excel(resolve_input_file('pearsonr_0.1.xlsx'),index_col=0)
#features_pearsonr = set(df_.index.tolist())
df_ = pd.read_excel(resolve_input_file('srocc_0.3_max_mic_0.1.xlsx'),index_col=0)
features_srocc_mic = set(df_.index.tolist())

df_ = pd.read_excel(resolve_input_file('X_人体成分.xlsx'),index_col=0)
#out_pearsonr = df_[col_names]
out_srocc_mic = df_[col_names]
filelist = ['X_人体成分','X_电阻抗','X_纹理特征_去常数项']
for f in filelist:
    df_= pd.read_excel(resolve_input_file(f +'.xlsx'), index_col=0)
    col_names = set(df_.columns.tolist())
    
    #new_frame = df_.drop(list(col_names - features_pearsonr),axis = 1)
    #out_pearsonr = pd.concat([out_pearsonr,new_frame],axis = 1)
    
    new_frame = df_.drop(list(col_names - features_srocc_mic),axis = 1)
    out_srocc_mic = pd.concat([out_srocc_mic,new_frame],axis = 1)
    
#out_pearsonr.to_excel(resolve_output_file('X_pearsonr_0.1.xlsx'))
out_srocc_mic.to_excel(resolve_output_file('X_srocc_0.3_max_mic_0.1.xlsx'))
