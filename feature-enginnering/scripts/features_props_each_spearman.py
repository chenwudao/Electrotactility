# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_props_each_spearman.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd

#filelist = ['X_人体成分','X_电阻抗','X_纹理特征_去常数项']
filelist = ['X_人体成分_new']
#filelist = ['X_人体成分','X_纹理特征_去常数项']
for f in filelist:
    col_names = ['受试者编号','手指','手指（优食-0 非优中-1 非优尾-2 非优食-3）']
    df_= pd.read_excel(resolve_input_file(f +'.xlsx'), index_col=0).drop(col_names,axis = 1).astype(float)
    col_names = df_.columns.tolist()
    correlations = pd.DataFrame()
    
    for col_name in col_names:
        correlations[col_name] = df_.corrwith(df_[col_name],method='spearman').abs()
        #correlations[col_name] = df_.corrwith(df_[col_name],method='spearman')
    correlations.to_excel(resolve_output_file(f + '_互相关性分析_spe.xlsx'))
