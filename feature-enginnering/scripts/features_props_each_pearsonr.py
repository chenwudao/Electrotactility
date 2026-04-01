# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：features_props_each_pearsonr.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import numpy as np
import pandas as pd

filelist = ['X_pearsonr_0.1']
for f in filelist:
    df_= pd.read_excel(resolve_input_file(f +'.xlsx'), index_col=0)
    data = df_.loc[:,:].values[:,3:].astype(float)
    #print("读取数据：\n{0}".format(data))
    col_names = df_.columns.tolist()[3:]
    
    correlations = np.abs(np.corrcoef(np.transpose(data)))
    
    df = pd.DataFrame(correlations,index=col_names,columns=col_names)
    
    df.to_excel(resolve_output_file(f + '_互相关性分析.xlsx'))
