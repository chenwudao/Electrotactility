# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：PEARSONR.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
df_= pd.read_excel(resolve_input_file('X_电阻抗谱数据.xlsx'))
data = df_.loc[:,:].values[:,2:].astype(float)
#print("读取数据：\n{0}".format(data))
tf = pd.read_excel(resolve_input_file('Y.xlsx'))
tags = tf.loc[:,:].values[:,2:].astype(float).ravel()
y = tags
#print("读取数据：\n{0}".format(tags))
col_names = df_.columns
result = pd.DataFrame([])
max_result = 0
max_col_num = 0
min_p = 1
min_col_num = 0
for i in range(data.shape[1]):
    x = data[:,i]
    #plt.scatter(x,tags,c = 'g')
    r = pearsonr(x,y)
    result.loc[0, col_names[i+2]] = r[0]
    result.loc[1, col_names[i+2]] = r[1]
    if(r[0] > max_result):
        max_result = r[0]
        max_col_num = i+2
    if(r[1] < min_p):
        min_p = r[1]
        min_col_num = i+2
        
    #plt.title(str(col_names[i+2])+'_'+str(max_mic))
    #plt.show()
x = data[:,max_col_num - 2]
plt.scatter(x,tags,c = 'g')
plt.ylabel('pw')
plt.xlabel(str(col_names[max_col_num]))
plt.title('pearsonr_corr_'+str(result[col_names[max_col_num]][0]))
plt.show()
x = data[:,min_col_num - 2]
plt.scatter(x,tags,c = 'g')
plt.ylabel('pw')
plt.xlabel(str(col_names[min_col_num]))
plt.title('pearsonr_p_'+str(result[col_names[min_col_num]][1]))
plt.show()
