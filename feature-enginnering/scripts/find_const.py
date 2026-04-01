# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：find_const.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd

df_= pd.read_excel(resolve_input_file('X_纹理特征.xlsx'), index_col=0)

const_cols = [c for c in df_.columns if df_[c].nunique(dropna=False) == 1]

df_ = df_.drop(const_cols,axis = 1)

df_.to_excel(resolve_output_file('X_纹理特征_去常数项.xlsx'))
