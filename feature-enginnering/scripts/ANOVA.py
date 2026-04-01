# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：ANOVA.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import numpy as np
from sklearn.feature_selection import f_classif
import pandas as pd

df_= pd.read_excel(resolve_input_file('X.xlsx'))
data = df_.loc[:,:].values[:,2:].astype(np.float)
#print("读取数据：\n{0}".format(data))
tf = pd.read_excel(resolve_input_file('Y.xlsx'))
tags = tf.loc[:,:].values[:,2:].astype(np.float)
#print("读取数据：\n{0}".format(tags))

fval = f_classif(data, tags)
fval = pd.Series(fval[0], index=range(data.shape[1]))
fval.plot.bar()
