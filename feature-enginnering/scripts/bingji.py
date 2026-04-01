# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：bingji.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
import datetime

#filelist = ['X_人体成分','X_电阻抗','X_纹理特征_去常数项']

#未经过有效性筛选
#filelist = ['X_人体成分_结合相关互相关_spe_0.7','X_电阻抗_结合相关互相关_spe_0.7','X_纹理特征_去常数项_结合相关互相关_spe_0.7']
#filelist = ['X_人体成分_结合相关互相关_spe_0.8','X_电阻抗_结合相关互相关_spe_0.8','X_纹理特征_去常数项_结合相关互相关_spe_0.8']
#filelist = ['X_人体成分_结合相关互相关_spe_0.9','X_电阻抗_结合相关互相关_spe_0.9','X_纹理特征_去常数项_结合相关互相关_spe_0.9']
filelist = ['X_人体成分_new_结合相关互相关_spe_0.9','X_电阻抗_结合相关互相关_spe_0.9','X_纹理特征_去常数项_结合相关互相关_spe_0.9']
#filelist = ['X_人体成分_结合相关互相关_spe_0.9','X_电阻抗_结合相关互相关_spe_0.9','X_纹理特征_去常数项_结合相关互相关_spe_0.9', 'X_pearsonr_0.1_结合相关互相关_0.9']

#经过有效性筛选
#filelist = ['X_人体成分_结合相关互相关_spe_0.9_rfr_0.9','X_电阻抗_结合相关互相关_spe_0.9_rfr_0.9','X_纹理特征_去常数项_结合相关互相关_spe_0.9_rfr_0.9']
#filelist = ['X_人体成分_结合相关互相关_spe_0.9_rfr_0.8','X_电阻抗_结合相关互相关_spe_0.9_rfr_0.4','X_纹理特征_去常数项_结合相关互相关_spe_0.9_rfr_0.4']

#filelist = ['X_人体成分_结合相关互相关_spe_0.9_rfr_0.9','X_电阻抗_结合相关互相关_spe_0.9_rfr_0.9','X_纹理特征_去常数项_结合相关互相关_spe_0.9_rfr_0.9', 'X_pearsonr_0.1_结合相关互相关_0.9_rfr_0.9']
#filelist = ['X_人体成分_结合相关互相关_spe_0.9_rfr_0.8','X_电阻抗_结合相关互相关_spe_0.9_rfr_0.4','X_纹理特征_去常数项_结合相关互相关_spe_0.9_rfr_0.4', 'X_pearsonr_0.1_结合相关互相关_0.9_rfr_0.8']
out_f = pd.DataFrame()
col_set = set()

for f in filelist:
    df_= pd.read_excel(resolve_input_file(f + '.xlsx'),index_col=0)
    col_names = set(df_.columns.tolist())
    df_ = df_.drop(list(col_names & col_set),axis = 1)
    col_set = col_set | col_names
    out_f = pd.concat([out_f,df_],axis = 1)
    
    
#out_f.to_excel(resolve_output_file('X.xlsx'))    
    

#未经过有效性筛选
#out_f.to_excel(resolve_output_file('X_结合相关互相关_spe_0.7.xlsx'))
#out_f.to_excel(resolve_output_file('X_结合相关互相关_spe_0.8.xlsx'))
#out_f.to_excel(resolve_output_file('X_结合相关互相关_spe_0.9.xlsx'))
out_f.to_excel(resolve_output_file('X_结合相关互相关_spe_0.9_new.xlsx'))
#out_f.to_excel(resolve_output_file('X_结合相关互相关_0.9.xlsx'))

#经过有效性筛选
#out_f.to_excel(resolve_output_file('X_人体成分0.9_电阻抗0.9_纹理特征0.9_结合相关互相关_0.9 _rfr.xlsx'))
#out_f.to_excel(resolve_output_file('X_人体成分0.8_电阻抗0.4_纹理特征0.4_结合相关互相关_0.9 _rfr.xlsx'))

#out_f.to_excel(resolve_output_file('X_人体成分0.9_电阻抗0.9_纹理特征0.9_pearsonr0.9_结合相关互相关_0.9 _rfr.xlsx'))
#out_f.to_excel(resolve_output_file('X_人体成分0.8_电阻抗0.4_纹理特征0.4_pearsonr0.8_结合相关互相关_0.9 _rfr.xlsx'))
