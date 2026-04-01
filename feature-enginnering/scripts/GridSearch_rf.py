# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：GridSearch_rf.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
 
data = load_breast_cancer()  #乳腺癌案例
print(data.data.shape)
 
xtrain,xtest,ytrain,ytest = train_test_split(data.data,data.target,test_size=0.3,random_state=42)
# 网格搜索
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(xtrain,ytrain)
score = rf.score(xtest,ytest)
cross_s = cross_val_score(rf,xtest,ytest,cv=5).mean()
print('rf:',score)
print('cv:',cross_s)
 
#调参第一步：n_estimators
cross = []
for i  in range(0,200,10):
    rf = RandomForestClassifier(n_estimators=i+1, n_jobs=-1,random_state=42)
    rf.fit(xtrain,ytrain)
    cross_score = cross_val_score(rf, xtest, ytest, cv=5).mean()
    cross.append(cross_score)
plt.plot(range(1,201,10),cross)
plt.xlabel('n_estimators')
plt.ylabel('acc')
plt.show()
print((cross.index(max(cross))*10)+1,max(cross))
# n_estimators缩小范围
cross = []
for i  in range(0,25):
    rf = RandomForestClassifier(n_estimators=i+1, n_jobs=-1,random_state=42)
    rf.fit(xtrain,ytrain)
    cross_score = cross_val_score(rf, xtest, ytest, cv=5).mean()
    cross.append(cross_score)
plt.plot(range(1,26),cross)
plt.xlabel('n_estimators')
plt.ylabel('acc')
plt.show()
print(cross.index(max(cross))+1,max(cross))
 
#调整max_depth
param_grid = {'max_depth' : np.arange(1,20,1)}
#一般根据数据大小进行尝试，像该数据集 可从1-10 或1-20开始
rf = RandomForestClassifier(n_estimators=11,random_state=42)
GS = GridSearchCV(rf,param_grid,cv=5)
GS.fit(data.data,data.target)
print(GS.best_params_,GS.best_score_)
max_depth = GS.best_params_['max_depth']

#调整max_features
param_grid = {'max_features' : np.arange(5,30,1)}
rf = RandomForestClassifier(n_estimators=11,max_depth = max_depth,random_state=42)
GS = GridSearchCV(rf,param_grid,cv=5)
GS.fit(data.data,data.target)
#print(GS.best_params_)
#print(GS.best_score_)
print(GS.best_params_,GS.best_score_)
max_features = GS.best_params_['max_features']

#调整min_samples_leaf
param_grid = {'min_samples_leaf' : np.arange(1,1+10,1)}
#一般是从其最小值开始向上增加10或者20
# 面对高维度高样本数据，如果不放心，也可以直接+50，对于大型数据可能需要增加200-300
# 如果调整的时候发现准确率怎么都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
rf = RandomForestClassifier(n_estimators=11,max_depth = max_depth,max_features = max_features,random_state=42)
GS = GridSearchCV(rf,param_grid,cv=5)
GS.fit(data.data,data.target)
print(GS.best_params_,GS.best_score_)
min_samples_leaf = GS.best_params_['min_samples_leaf']

#调整min_samples_split
param_grid = {'min_samples_split' : np.arange(2,2+20,1)}
#一般是从其最小值开始向上增加10或者20
# 面对高维度高样本数据，如果不放心，也可以直接+50，对于大型数据可能需要增加200-300
# 如果调整的时候发现准确率怎么都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
rf = RandomForestClassifier(n_estimators=11,max_depth = max_depth,max_features = max_features,min_samples_leaf = min_samples_leaf,random_state=42)
GS = GridSearchCV(rf,param_grid,cv=5)
GS.fit(data.data,data.target)
print(GS.best_params_,GS.best_score_)
min_samples_split = GS.best_params_['min_samples_split']
 
#调整criterion
param_grid = {'criterion' :['gini','entropy']}
#一般是从其最小值开始向上增加10或者20
# 面对高维度高样本数据，如果不放心，也可以直接+50，对于大型数据可能需要增加200-300
# 如果调整的时候发现准确率怎么都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
rf = RandomForestClassifier(n_estimators=11,max_depth = max_depth,max_features = max_features,min_samples_leaf = min_samples_leaf,min_samples_split=min_samples_split,random_state=42)
GS = GridSearchCV(rf,param_grid,cv=5)
GS.fit(data.data,data.target)
print(GS.best_params_,GS.best_score_)
criterion= GS.best_params_['criterion']

rf = RandomForestClassifier(n_estimators=11,criterion=criterion,max_depth = max_depth,max_features = max_features,min_samples_leaf = min_samples_leaf,min_samples_split=min_samples_split,random_state=42)
rf.fit(xtrain,ytrain)
score = rf.score(xtest,ytest)
cross_s = cross_val_score(rf,xtest,ytest,cv=5).mean()
print('rf:',score)
print('cv:',cross_s)
