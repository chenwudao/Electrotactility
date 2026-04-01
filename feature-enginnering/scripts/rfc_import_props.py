# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：rfc_import_props.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
from MIC import CyrusMIC
from scipy.stats import pearsonr,normaltest
from sklearn.ensemble import RandomForestClassifier

col_num = 3
tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
tags = tf.loc[:,:].values[:,3].astype(float).ravel()
log_tags = tf.loc[:,:].values[:,5].astype(float).ravel()
y_ord = tags.argsort()
#filelist = ['X_人体成分','X_电阻抗','X_纹理特征_去常数项']
#filelist = ['Y']
filelist = ['X_结合相关互相关_spe_0.9_rfr_50_']
for f in filelist:
    df_= pd.read_excel(resolve_input_file(f + '.xlsx'), index_col=0)
    data = df_.loc[:,:].values[:,col_num:].astype(float)
    #print("读取数据：\n{0}".format(data))
    col_names = df_.columns
    result = pd.DataFrame(index = col_names[col_num:])
    
    #定义随机森林回归模型
    # 非跨被试3477 二分类 中位数阈值 0.671875 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅电学
    #model_rfc = RandomForestClassifier(random_state = 231,criterion='gini',n_estimators = 11,max_depth = 12,max_features = 2,min_samples_leaf = 1,min_samples_split = 53)
    # 非跨被试3477 二分类 中位数阈值 0.75 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅解剖
    #model_rfc = RandomForestClassifier(random_state = 1440,criterion='gini',n_estimators = 4,max_depth = 4,max_features = 1,min_samples_leaf = 3,min_samples_split = 2)
    # 非跨被试3477 二分类 中位数阈值 0.6875 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅图像
    #model_rfc = RandomForestClassifier(random_state = 917,criterion='gini',n_estimators = 40,max_depth = 11,max_features = 13,min_samples_leaf = 6,min_samples_split = 2)

    # 跨被试2832 二分类 中位数阈值 0.6640625 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅电学
    #model_rfc = RandomForestClassifier(random_state = 21,criterion='gini',n_estimators = 150,max_depth = 14,max_features = 2,min_samples_leaf = 1,min_samples_split = 2)
    # 跨被试2832 二分类 中位数阈值 0.6875 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅整体
    #model_rfc = RandomForestClassifier(random_state = 24,criterion='gini',n_estimators = 13,max_depth = 9,max_features = 2,min_samples_leaf = 33,min_samples_split = 2)
    # 跨被试2832 二分类 中位数阈值 0.703125 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅局部
    #model_rfc = RandomForestClassifier(random_state = 21,criterion='gini',n_estimators = 11,max_depth = 6,max_features = 23,min_samples_leaf = 1,min_samples_split = 2)
    #↓ 跨被试 2832 二分类 0.734375 中位数阈值 五折 X_结合相关互相关_spe_0.9_rfr_50 或 50_
    model_rfc = RandomForestClassifier(random_state = 83,criterion='gini',n_estimators = 66,max_depth = 9,max_features = 7,min_samples_leaf = 8,min_samples_split = 2)

    # 非跨被试3477 二分类 中位数阈值 0.671875 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅电学
    #model_rfc = RandomForestClassifier(random_state = 231,criterion='gini',n_estimators = 11,max_depth = 12,max_features = 2,min_samples_leaf = 1,min_samples_split = 53)
    # 非跨被试3477 二分类 中位数阈值 0.765625 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅整体
    #model_rfc = RandomForestClassifier(random_state = 570,criterion='gini',n_estimators = 10,max_depth = 3,max_features = 1,min_samples_leaf = 1,min_samples_split = 2)
    # 非跨被试3477 二分类 中位数阈值 0.71875 五折 X_pea_sro_结合相关互相关_0.7 X_结合相关互相关_spe_0.9_rfr_50_仅局部
    #model_rfc = RandomForestClassifier(random_state = 20,criterion='gini',n_estimators = 11,max_depth = 9,max_features = 5,min_samples_leaf = 1,min_samples_split = 2)
    #↓ 非跨被试 3477 二分类 0.765625 中位数阈值 五折 X_结合相关互相关_spe_0.9_rfr_50_
    model_rfc = RandomForestClassifier(random_state = 610,criterion='gini',n_estimators = 92,max_depth = 7,max_features = 7,min_samples_leaf = 1,min_samples_split = 2)
    

    #拟合模型
    model_rfc.fit(data,tags)
    importances = model_rfc.feature_importances_
    result['rfc_importance'] = importances.tolist()
    
    result.to_excel(os.path.join(RESULT_DIR, '图_new',f + '_rfc重要程度_非跨被试.xlsx'))
