# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：fenji.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class my_KFold(object):
    def __init__(self,n_splits = 10,shuffle= True,random_state = 4073,tf=pd.DataFrame(),drop_sub_l = [],cross_sub = True,res=None,log= True):
        if res:
            self.res = res
            self.n_splits = len(res)
        else:
            self.n_splits = n_splits
            print('筛除受试者：',drop_sub_l)
            kf = KFold(n_splits = n_splits,shuffle = shuffle,random_state = random_state)
            self.res = []
            if cross_sub:
                subs = np.array(list(set(tf['受试者编号'])-set(drop_sub_l)))
                #print(subs)
                for train_i,test_i in kf.split(subs):
                    sl = subs.take(test_i).tolist()
                    train_index=(tf[~(tf['受试者编号'].isin(sl+drop_sub_l))].index.tolist())
                    test_index=(tf[(tf['受试者编号'].isin(sl))].index.tolist())
                    if log:
                        print('受试者编号：',sl)
                        print(test_index)
                        print('训练集数量：',len(train_index))
                    self.res.append([train_index,test_index])
                    #if 1 in train_index:
                    #    print('error')
            else:
                data = tf[~(tf['受试者编号'].isin(drop_sub_l))].index
                #print(data.shape)
                for train_i,test_i in kf.split(data):
                    train_index = data.take(train_i).tolist()
                    test_index = data.take(test_i).tolist()
                    if log:
                        print(test_index)
                        print('训练集数量：',len(train_index))
                    self.res.append([train_index,test_index])
    def split(self,X,y=None,group = None):
        return ((train_index,test_index) for train_index,test_index in self.res)
    def get_n_splits(self,X=None,y=None,group = None):
        return self.n_splits

tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
y = tf.loc[:,:].values[:,3].ravel()
drop_sub_l = []  
#获取训练和测试集对应索引数组
cross_sub = 0 #是否跨被试
shuffle = True #是否打乱顺序
total_fold = 5
log = True #是否打印分组日志
#自定义分组
res = []
typ = '真实标签'
        
if cross_sub:
   kf_r_state = 2832 #跨被试
   pf = pd.read_excel(resolve_input_file('跨被试分类结果值.xlsx'),index_col=0)
else:
   kf_r_state = 3477 #非跨被试
   pf = pd.read_excel(resolve_input_file('非跨被试分类结果值.xlsx'),index_col=0)

kf = my_KFold(n_splits =total_fold,shuffle= shuffle,random_state = kf_r_state,tf=tf,drop_sub_l = drop_sub_l,cross_sub = cross_sub,res=res,log = log)

res_0 = []
res_1 = []
for train_i,test_i in kf.split(y):
    tmp = pf.loc[train_i,:]
    train_0 = tmp[tmp[typ].isin([0])].index.tolist()
    train_1 = tmp[tmp[typ].isin([1])].index.tolist()
    
    tmp = pf.loc[test_i,:]
    test_0 = tmp[tmp[typ].isin([0])].index.tolist()
    test_1 = tmp[tmp[typ].isin([1])].index.tolist()
    
    res_0.append([train_0,test_0])
    res_1.append([train_1,test_1])
print(res_0)
print(res_1)
