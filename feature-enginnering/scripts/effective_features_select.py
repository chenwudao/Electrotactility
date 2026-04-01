# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：effective_features_select.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
filelist = ['X_人体成分','X_电阻抗','X_纹理特征_去常数项']
out_f = pd.DataFrame()
for f in filelist:
    df = pd.read_excel(resolve_input_file(f +'_相关性分析.xlsx'), index_col=0)
    new_frame = df[(df['normaltest_p'] > 0.05) & (df['pearsonr_p'] < 0.05)]
    #new_frame = df[(df['srocc_corr'] > 0.3) | (df['spearman'] > 0.3) | (df['kendall'] > 0.3) | (df['max_mic'] > 0.1)]
    out_f = pd.concat([out_f,new_frame],axis = 0)
out_f = out_f.drop('rfr_importance',axis = 1)
out_f.to_excel(resolve_output_file('pearsonr_0.05.xlsx'))
#out_f.to_excel(resolve_output_file('srocc_0.3_max_mic_0.1.xlsx'))
