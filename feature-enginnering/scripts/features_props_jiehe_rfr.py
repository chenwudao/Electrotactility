# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_props_jiehe_rfr.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

num = 35
col_num = 3
tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
tags = tf.loc[:,:].values[:,3].astype(float).ravel()
filelist = ['X_结合相关互相关_spe_0.9_rfr_50_']
#filelist = ['X_结合相关互相关_spe_0.7_结合相关互相关_spe_0.7','X_结合相关互相关_spe_0.8','X_结合相关互相关_spe_0.9']
#filelist = ['X_pea_sro_结合相关互相关_0.9','X_pea_sro_结合相关互相关_0.95','X_结合相关互相关_0.9','X_结合相关互相关_spe_0.9']
#filelist = ['X_结合相关互相关']
for f in filelist:
    df_= pd.read_excel(resolve_input_file(f + '.xlsx'), index_col=0)
    data = df_.loc[:,:].values[:,col_num:].astype(float)
    #print("读取数据：\n{0}".format(data))
    col_names = df_.columns.tolist()[3:]
    
    
    #定义随机森林回归模型
    model_rfr = RandomForestClassifier(random_state = 610,criterion='gini',n_estimators = 92,max_depth = 7,max_features = 7,min_samples_leaf = 1,min_samples_split = 2)

    #拟合模型
    model_rfr.fit(data,tags)
    importances = model_rfr.feature_importances_
    result = pd.DataFrame(importances,columns=['rfr_importance'],index = col_names)
    s_rfr_importance = set(result.sort_values(by=['rfr_importance'],ascending = False).iloc[:num,:].index)
    df_ = df_.drop(list(set(col_names) - s_rfr_importance),axis=1)
    df_.to_excel(resolve_output_file(f + '_rfr_'+str(num)+'.xlsx'))
