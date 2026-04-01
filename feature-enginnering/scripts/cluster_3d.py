# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：cluster_3d.py
统一使用项目内的相对路径。
"""

import os
from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR, RESULT_DIR

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import glob
# https://blog.csdn.net/icefountain/article/details/129181949

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

from sklearn.decomposition import PCA

def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def plotMatrixPoint(Mat, Tag,Y,f):
    """
    输入二维点矩阵和标签，能够改变不同形状
    :param Mat:
    :param Label:
    :return:
    """
    x = Mat[:, 0]
    y = Mat[:, 1]
    map_size = {0: 15, 1: 15,2:15}
    size = list(map(lambda x: map_size[x], Tag))
    map_color = {0: 'r', 1: 'g',2:'b'}
    color = list(map(lambda x: map_color[x], Tag))
    map_marker = {0: 'o', 1: 's',2:'v'}
    markers = list(map(lambda x: map_marker[x], Tag))
    mscatter(np.array(x), np.array(y), s=size, c=color, m=markers)  # scatter函数只支持array类型数据
    #for i in range(data_pca.shape[0]):
    #    if Label[i] != Tag[i]:
    #        plt.text(data_pca[i][0]*1.1, data_pca[i][1]*1.1, Y[i], fontsize=8, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签
    plt.title(f)
    plt.show()

tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
y = tf.loc[:,:].values[:,3].ravel()
temp = y.copy()
temp.sort()
thr_t = [[130,230],[temp[int(0.33*len(temp))],temp[int(0.66*len(temp))]]]
thr1,thr2 = thr_t[0]
# tags = np.array([1 if tag > 190 else 0 for tag in y])
tags = np.array([2 if tag > thr2 else (1 if tag > thr1 else 0) for tag in y])
scaler = MinMaxScaler()
y=y.reshape(-1, 1)
scaler.fit(y)
y = scaler.transform(y).ravel()
filelist = glob.glob(str(LINEAR_WORK_DIR / '*'))
#filelist = [resolve_input_file('X_pea_sro_结合相关互相关_0.9.xlsx')]
for f in filelist:
    # 导入数据，默认第一行为索引，index_col设定第一列也为索引
    df_raw = pd.read_excel(f, index_col=0).iloc[:,3:]
    
    # 数据0-1标准化
    #df_normalized_data = df_raw.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # print(df_normalized_data)
    
    
    # ------------------------下面介绍如何绘制聚类散点图-----------------------------
    # 对二分类的散点图绘制，网上教程很多，此篇文章主要介绍多分类的散点图绘制问题
    # 首先，对原数据进行 PCA 降维处理，获得散点图的横纵坐标轴数据
    scaler = StandardScaler()  
    scaler.fit(df_raw)
    data_pca = scaler.transform(df_raw)
    components = 3
    pca = PCA(n_components=components)
    pca.fit(data_pca)
    data_pca = pca.transform(data_pca)
    
    scaler = MinMaxScaler()  
    scaler.fit(data_pca)   
    data_pca = scaler.transform(data_pca)
    
    if components == 2:
        plotMatrixPoint(data_pca,tags,y,f)
    
    
    else:
        data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
        
        data_pca.insert(data_pca.shape[1], 'tags', tags)
        
        data_pca = data_pca.values
        
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
        map_color = {0: 'r', 1: 'g',2:'b'}
        # 创建图形
        for count,d in enumerate(data_pca):
            a = ((d[0]**2 + d[1]**2 + d[2]**2)**0.5)/(3**0.5)
            ax.scatter3D(d[0], d[1], d[2], color=map_color[d[3]],s = a*80,alpha=1 - y[count])
        plt.title(f)
