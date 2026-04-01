# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：temp.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Net(nn.Module):
    def init(self):
        super(Net,self).init()
        self.fc1 = nn.Linear(10,5)
        self.fc2 = nn.Linear(5,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""
df1 = pd.read_excel(resolve_input_file('X_人体成分.xlsx'),index_col=0)
#data1 = df1.loc[:,:].values[:,3:]
df2 = pd.read_excel(resolve_input_file('X_电阻抗.xlsx'),index_col=0)
#data2 = df2.loc[:,:].values[:,3:]
df3 = pd.read_excel(resolve_input_file('X_纹理特征_去常数项.xlsx'),index_col=0)
#data3 = df3.loc[:,:].values[:,3:]
#data = [row1 + row2 + row3 for row1,row2,row3 in zip(data1,data2,data3)]
data = np.concatenate((df1.loc[:,:].values[:,3:],df2.loc[:,:].values[:,3:]),axis = 1)
data = np.concatenate((data,df3.loc[:,:].values[:,3:]),axis = 1)
"""

df = pd.read_excel(resolve_input_file('X_结合相关互相关_0.9_rfr_60.xlsx'),index_col=0)
data = df.loc[:,:].values[:,3:]
#print("读取数据：\n{0}".format(data))
tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
tags = tf.loc[:,:].values[:,3].ravel()
#print("读取数据：\n{0}".format(tags))
indexes = set(df.index.tolist())

#test_l = [0,3,17,24,50,53,57,68,86,98,109,110,123] #纹理特征优 随机森林
#test_l = [1,21,29,36,44,46,61,73,74,75,99,110,118] #纹理特征优 决策树
#test_l = [7,10,16,17,21,44,46,64,65,67,107,109,117] #人体成分优
#test_l = [23,30,38,42,55,66,75,76,78,85,103,118,119] #电阻抗v1-v2优
test_l = [20,25,32,37,50,53,65,67,75,88,95,107,110] #X_pearsonr_0.1_结合相关互相关_0.9.xlsx优

test_data = data.take(test_l,0)
test_tags = tags.take(test_l,0)
train_data = data.take(list(indexes - set(test_l)),0)
train_tags = tags.take(list(indexes - set(test_l)),0)

#train_data,test_data,train_tags,test_tags = train_test_split(data,tags,shuffle= True,test_size=0.1)
Standard = 0
if Standard:
    print('数据标准化...')
    scaler = StandardScaler()  
    scaler.fit(train_data)
    train_data = scaler.transform(train_data) 
    test_data = scaler.transform(test_data)
"""
train_data_1  = data[:108]
train_data_2  = data[108:108]
train_data = np.vstack((train_data_1,train_data_2))
train_tags_1 = tags[:108]
train_tags_2 = tags[108:108]
train_tags = np.hstack((train_tags_1,train_tags_2))
test_data = data[108:]
test_tags = tags[108:]
"""
print("测试标签：\n{0}".format(test_tags))

#定义决策树回归模型
model_dtr = DecisionTreeRegressor()
#拟合模型
model_dtr.fit(train_data,train_tags)
#预测
pred_tags_dtr = model_dtr.predict(test_data)

print("决策树预测标签：\n{0}".format(pred_tags_dtr))

abs_err = np.abs(test_tags-pred_tags_dtr)
rel_err = abs_err/test_tags

#print("绝对误差：",abs_err)
#print("相对误差：",rel_err)

mse = mean_squared_error(test_tags, pred_tags_dtr) 
rmse = mean_squared_error(test_tags, pred_tags_dtr, squared=False) 
mae = mean_absolute_error(test_tags, pred_tags_dtr)
r2s = r2_score(test_tags, pred_tags_dtr)

print("均方误差：",mse)
print("均方根误差：",rmse)
print("平均绝对误差：",mae)
print("R-squared：",r2s)

#定义随机森林回归模型
model_rfr = RandomForestRegressor()
#拟合模型
model_rfr.fit(train_data,train_tags)
#预测
pred_tags_rfr = model_rfr.predict(test_data)
print("随机森林预测标签：\n{0}".format(pred_tags_rfr))

abs_err_rfr = np.abs(test_tags-pred_tags_rfr)
rel_err_rfr = abs_err_rfr/test_tags

#print("绝对误差：",abs_err_rfr)
#print("相对误差：",rel_err_rfr)

mse = mean_squared_error(test_tags, pred_tags_rfr) 
rmse = mean_squared_error(test_tags, pred_tags_rfr, squared=False) 
mae = mean_absolute_error(test_tags, pred_tags_rfr)
r2s = r2_score(test_tags, pred_tags_rfr)

print("均方误差：",mse)
print("均方根误差：",rmse)
print("平均绝对误差：",mae)
print("R-squared：",r2s)

importances = model_rfr.feature_importances_

# 绘制特征重要性
plt.bar(range(data.shape[1]), importances)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.show()

#定义支持向量机回归模型
model_svr = SVR()
#拟合模型
model_svr.fit(train_data,train_tags)
#预测
pred_tags_svr = model_svr.predict(test_data)
print("支持向量机预测标签：\n{0}".format(pred_tags_svr))

abs_err_svr = np.abs(test_tags-pred_tags_svr)
rel_err_svr = abs_err_svr/test_tags

#print("绝对误差：",abs_err_rfr)
#print("相对误差：",rel_err_rfr)

mse = mean_squared_error(test_tags, pred_tags_svr) 
rmse = mean_squared_error(test_tags, pred_tags_svr, squared=False) 
mae = mean_absolute_error(test_tags, pred_tags_svr)
r2s = r2_score(test_tags, pred_tags_svr)

print("均方误差：",mse)
print("均方根误差：",rmse)
print("平均绝对误差：",mae)
print("R-squared：",r2s)
