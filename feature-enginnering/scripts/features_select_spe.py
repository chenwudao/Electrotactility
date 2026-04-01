# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_select_spe.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd

num = 20
#filelist = ['X','X_电阻抗谱数据','X_纹理特征结果']
filelist = ['X_结合相关互相关_spe_0.7']
for f in filelist:
    df_= pd.read_excel(resolve_input_file(f +'_相关性分析.xlsx'), index_col=0)
    col_names = set(df_.index)
    s_spearman = set(df_.sort_values(axis = 0,by=['spearman','max_mic','srocc_corr','pearsonr_corr'],ascending = False).iloc[:num,:].index)
    
    df = pd.read_excel(resolve_input_file(f +'.xlsx'), index_col=0)
    

    s_drop = col_names - s_spearman
    #s_drop.remove('手指（优食-0 非优中-1 非优尾-2 非优食-3）')

    df_.drop(list(s_drop),axis = 0).to_excel(resolve_output_file(f + '_Selspe_'+str(num)+'_相关性分析.xlsx'))
    df.drop(list(s_drop),axis = 1).to_excel(resolve_output_file(f + '_Selspe_'+str(num)+'.xlsx'))
