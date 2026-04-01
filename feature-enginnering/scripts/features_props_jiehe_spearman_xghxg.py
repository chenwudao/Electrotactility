# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_props_jiehe_spearman_xghxg.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd

thr = 0.9
#filelist = ['srocc_0.3_max_mic_0.1']
#filelist = ['人体成分','电阻抗','纹理特征_去常数项']
filelist = ['人体成分_new']
for f in filelist:
    df_= pd.read_excel(resolve_input_file('X_' + f + '_互相关性分析_spe.xlsx'),index_col=0)
    #df_corr = pd.read_excel(resolve_input_file(f + '.xlsx'),index_col=0)
    df_corr = pd.read_excel(resolve_input_file('X_' + f + '_相关性分析.xlsx'),index_col=0)
    
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
                    """
                    #mic优先 和spe优先的结果一样
                    corr1 = df_corr.loc[col_name_1,'max_mic']
                    corr2 = df_corr.loc[col_name,'max_mic']
                    if abs(corr1 - corr2) <= 0.03:
                        corr1 = df_corr.loc[col_name_1,'spearman']
                        corr2 = df_corr.loc[col_name,'spearman']
                        if abs(corr1 - corr2) <= 0.05:
                            corr1 = df_corr.loc[col_name_1,'srocc_corr']
                            corr2 = df_corr.loc[col_name,'srocc_corr']
                            if abs(corr1 - corr2) <= 0.05:
                                corr1 = df_corr.loc[col_name_1,'kendall']
                                corr2 = df_corr.loc[col_name,'kendall']
                    """
                    if corr1 > corr2:
                        col_set.add(col_name)
                        break
                    else:
                        col_set.add(col_name_1)
    
    df_ = pd.read_excel(resolve_input_file('X_' + f + '.xlsx'),index_col=0)
    col_names = set(df_.columns.tolist())
    df_ = df_.drop(list(col_names & col_set),axis = 1)          
    df_.to_excel(resolve_output_file('X_' + f + '_结合相关互相关_spe_'+ str(thr)+'.xlsx'))
