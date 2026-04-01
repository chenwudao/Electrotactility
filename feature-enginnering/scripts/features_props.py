# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_props.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
from MIC import CyrusMIC
from scipy.stats import pearsonr,normaltest
from sklearn.ensemble import RandomForestRegressor

col_num = 3
tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
tags = tf.loc[:,:].values[:,3].astype(float).ravel()
log_tags = tf.loc[:,:].values[:,5].astype(float).ravel()
y_ord = tags.argsort()
#filelist = ['X_人体成分','X_电阻抗','X_纹理特征_去常数项']
#filelist = ['Y']
filelist = ['X_人体成分_new']
for f in filelist:
    df_= pd.read_excel(resolve_input_file(f +'.xlsx'), index_col=0)
    data = df_.loc[:,:].values[:,col_num:].astype(float)
    #print("读取数据：\n{0}".format(data))
    col_names = df_.columns
    result = pd.DataFrame([])
    result = pd.DataFrame(index = df_.columns[3:])
    for i in range(data.shape[1]):
        x = data[:,i]
        x_ord = x.argsort()
        
        nt = normaltest(x) #正态检验
        log_psr = pearsonr(x,log_tags) #皮尔森系数 -1 ~ 1 (趋近于1或-1相关性越大)
        
        mic_tool = CyrusMIC() 
        max_mic = mic_tool.cal_MIC(x,tags) #最大信息系数0 ~ 1(趋近于1相关性越大)
        srocc = pearsonr(x_ord,y_ord) #秩序相关系数 -1 ~ 1 (趋近于1或-1相关性越大)
        
        result.loc[col_names[i+col_num], 'normaltest_stat'] = nt[0]
        result.loc[col_names[i+col_num], 'normaltest_p'] = nt[1] #p值越大 越显著
        result.loc[col_names[i+col_num], 'pearsonr_corr'] = abs(log_psr[0])
        result.loc[col_names[i+col_num],'pearsonr_p'] = log_psr[1] #p值越小 越显著
        
        result.loc[col_names[i+col_num],'max_mic'] = max_mic
        result.loc[col_names[i+col_num], 'srocc_corr'] = abs(srocc[0])
        result.loc[col_names[i+col_num], 'srocc_p'] = srocc[1] #p值越小 越显著
        """
        result.loc['normaltest_stat', col_names[i+col_num]] = nt[0]
        result.loc['normaltest_p', col_names[i+col_num]] = nt[1] #p值越大 越显著
        result.loc['pearsonr_corr', col_names[i+col_num]] = abs(log_psr[0])
        result.loc['pearsonr_p', col_names[i+col_num]] = log_psr[1] #p值越小 越显著
        
        result.loc['max_mic', col_names[i+col_num]] = max_mic
        result.loc['srocc_corr', col_names[i+col_num]] = abs(srocc[0])
        result.loc['srocc_p', col_names[i+col_num]] = srocc[1] #p值越小 越显著
        """
    df = pd.DataFrame(data)
    df['y'] = tags
    correlations = df.corrwith(df.y,method='spearman').abs()#等级差数法
    #result.loc['spearman'] = correlations[:(len(correlations)-1)].tolist()
    result['spearman'] = correlations[:(len(correlations)-1)].tolist()
    correlations = df.corrwith(df.y,method='kendall').abs() #肯德尔和谐系数
    #result.loc['kendall'] = correlations[:(len(correlations)-1)].tolist()
    result['kendall'] = correlations[:(len(correlations)-1)].tolist()
    
    #定义随机森林回归模型
    model_rfr = RandomForestRegressor()
    #拟合模型
    model_rfr.fit(data,tags)
    importances = model_rfr.feature_importances_
    result['rfr_importance'] = importances.tolist()
    
    result.to_excel(resolve_output_file(f + '_相关性分析.xlsx'))
