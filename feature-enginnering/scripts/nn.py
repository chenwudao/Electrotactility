# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：nn.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

import pandas as pd
import glob
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from torch.nn import init
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 

r_state = 21
filelist = glob.glob(str(LINEAR_WORK_DIR / '*'))
filelist = [resolve_input_file('X_结合相关互相关_非线性优先_rfr.xlsx')]
for f in filelist:
    #df = pd.read_excel(f,index_col=0)
    #data_ = df.loc[:,:].values[:,3:].astype(float)
    df = pd.read_excel(f)
    data_ = df.loc[:,:].values[:,2:].astype(float)
    #print("读取数据：\n{0}".format(data))
    
    #tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
    #tags_ = tf.loc[:,:].values[:,3].astype(float)
    tf = pd.read_excel(resolve_input_file('Y.xlsx'))
    tags_ = tf.loc[:,:].values[:,2].astype(float)
    #print("读取数据：\n{0}".format(tags))
    
    isnan = np.isnan(data_) # 判断每个元素是不是nan,返回[False,False,False,False,True]
    if True in isnan:
        print('data_ 中存在nan') 
    isnan = np.isnan(tags_) # 判断每个元素是不是nan,返回[False,False,False,False,True]
    if True in isnan:
        print('tag_ 中存在nan') 
    num_input ,num_example = data_.shape[1],data_.shape[0]
    
    """
    num_input = 50
    num_example = 10000
    true_w = torch.ones(1,num_input)*0.0056
    true_b = 0.028
    data = torch.tensor(np.random.normal(0,0.001,size  = (num_example,num_input)),dtype = torch.float32)
    tags = torch.mm(data,true_w.t()) + true_b
    tags += torch.normal(0,0.001,tags.shape)
    """
    data = torch.tensor(data_,dtype = torch.float32)
    tags = torch.tensor(tags_,dtype = torch.float32)
    
    train_x,test_x,train_y,test_y = train_test_split(data,tags,shuffle= True,test_size=0.1,random_state = r_state)
    print(test_y.shape)
    
    batch_size = 12
    #batch_size = 12
    train_dataset = TensorDataset(train_x,train_y)
    train_iter = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
    )
    test_dataset = TensorDataset(test_x,test_y)
    test_iter = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
    )
    
    
    model = nn.Sequential(OrderedDict([
        ('linear1',nn.Linear(num_input,256)),
        ('linear2',nn.Linear(256,128)),
        ('linear3',nn.Linear(128,1)),
    ])
    )
    for param in model.parameters():
        init.normal_(param,mean = 0 ,std = 0.001)
    
    # for param in model.state_dict():
    #     print(param)
    #     print(model.state_dict()[param])
    
    lr = 0.00000001
    #lr = 0.001
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr)
    
    
    def train(model,train_iter,test_iter,loss,num_epochs,batch_size,lr):
        train_ls,test_ls = [],[]
        for epoch in range(num_epochs):
            train_ls_sum ,test_ls_sum = 0,0
            for x,y in train_iter:
                #y_pred = torch.relu(model(x))
                y_pred = model(x)
                p = y_pred.detach().numpy()
                if True in np.isnan(p):
                    print(p)
                l = loss(y_pred,y)
                #b = l.item()
                #print('%f'%(b))
                #c = y.detach().numpy() 
                #d = y_pred.detach().numpy() 
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_ls_sum += l.item()
            for x ,y in test_iter:
                #y_pred = torch.relu(model(x))
                y_pred = model(x)
                #e = y_pred.detach().numpy() 
                p = y_pred.detach().numpy()
                if True in np.isnan(p):
                    print(p)
                l = loss(y_pred,y)
                test_ls_sum +=l.item()
            train_ls.append(train_ls_sum)
            test_ls.append(test_ls_sum)
            #print('epoch %d,train_loss %.6f,test_loss %f'%(epoch+1, train_ls[epoch],test_ls[epoch]))
        return train_ls,test_ls
    
    #num_epochs = 200
    num_epochs = 100
    train_loss ,test_loss = train(model,train_iter,test_iter,loss,num_epochs,batch_size,lr)
    y_pred = model(test_x)
    e = y_pred.detach().numpy() 
    f = test_y.detach().numpy()
    mse = mean_squared_error(f, e) 
    rmse = mean_squared_error(f, e, squared=False) 
    mae = mean_absolute_error(f, e)
    r2s = r2_score(f, e)
    #print(f)
    #print(e)
    print("均方误差：",mse)
    print("均方根误差：",rmse)
    print("平均绝对误差：",mae)
    print("R-squared：",r2s)
    x = np.linspace(0,len(train_loss),len(train_loss))
    plt.plot(x,train_loss,label="train_loss",linewidth=1.5)
    plt.plot(x,test_loss,label="test_loss",linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
