# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:18:59 2019

@author: Winry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import font_manager

# 导入宋体
my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")



def myfind(x,y):
    return [ a for a in range(len(y)) if y[a] == x]


def myKMeans():
    kmeans = KMeans(n_clusters=5)
    
    data2017 = pd.read_csv('data2017.csv')
    data2018 = pd.read_csv('data2018.csv')
    
    tarinData2017 = data2017.sample(n=1200000)
    testData2018 = data2018.sample(n=1200000)
    tarinData2017 = tarinData2017.reset_index()
    testData2018 = testData2018.reset_index()
    
    
#    nameList = ["ArrDelay","CRSElapsedTime","AirTime","Distance"]
    nameList = ["ArrDelay"]
    newtarinData2017 = tarinData2017[nameList]
    
    
#    
#    peopleNum2017 = np.random.randint(187,330,[20000,1])
#    peopleNum2017 = pd.DataFrame(peopleNum2017)
#    peopleNum2017.rename(columns={0:'peopleNum'}, inplace=True)
#    
#    peopleNum2018 = np.random.randint(187,330,[20000,1])
#    peopleNum2018 = pd.DataFrame(peopleNum2017)
#    
#    peopleNum2018.rename(columns={0:'peopleNum'}, inplace=True) 
#    
#    tarinData2017 = pd.concat([tarinData2017,peopleNum2017], axis=1)
#    testData2018 = pd.concat([testData2018,peopleNum2018], axis=1)
#    
    
    kmeans.fit(newtarinData2017)
    y_kmeans = kmeans.predict(newtarinData2017)
    
    
    numOne = myfind(0,y_kmeans)
    numTwo = myfind(1,y_kmeans)
    numThree = myfind(2,y_kmeans)
    numFour = myfind(3,y_kmeans)
    numFive = myfind(4,y_kmeans)
    
    
    clusterOne = tarinData2017.iloc[numOne]['ArrDelay']
    clusterTwo = tarinData2017.iloc[numTwo]['ArrDelay']
    clusterThree = tarinData2017.iloc[numThree]['ArrDelay']
    clusterFour = tarinData2017.iloc[numFour]['ArrDelay']
    clusterFive = tarinData2017.iloc[numFive]['ArrDelay']
    
    plt.figure(figsize=(17,8),dpi=80)  # 创建画布
    plt.plot(np.arange(clusterOne.shape[0]), clusterOne) 
    plt.plot(np.arange(clusterTwo.shape[0]), clusterTwo) 
    plt.plot(np.arange(clusterThree.shape[0]), clusterThree)
    plt.plot(np.arange(clusterFour.shape[0]),clusterFour)
    plt.plot(np.arange(clusterFive.shape[0]),clusterFive)
    
    label = ["第一类","第二类","第三类","第四类","第五类"]
    plt.tick_params(labelsize=33)
    plt.xlabel('样本点',fontproperties=my_font,size=33)
    plt.ylabel('延误时间/分钟',fontproperties=my_font,size=33)
    plt.legend(label,prop =my_font)
    plt.show()
    
    print (40 * '-')
    print("第一类最大延误时间:%d,最小延误时间:%d" % (max(clusterOne),
                                      min(clusterOne)))
    print (40 * '-')
    print ("第二类最大延误时间:%d,最小延误时间:%d" % (max(clusterTwo),
                                       min(clusterTwo)))
    print (40 * '-')
    print("第三类最大延误时间:%d,最小延误时间:%d" % (max(clusterThree),
                                      min(clusterThree)))
    print (40 * '-')
    print ("第四类最大延误时间:%d,最小延误时间:%d" % (max(clusterFour),
                                       min(clusterFour)))
    print (40 * '-')
    print("第五类最大延误时间:%d,最小延误时间:%d" % (max(clusterFive),
                                      min(clusterFive)))
    print (40 * '-')
    
    y_kmeans = pd.DataFrame(y_kmeans)
    y_kmeans.rename(columns={0:'DelyaCluster'}, inplace=True)
    clusterData2017 = pd.concat([tarinData2017,y_kmeans], axis=1)
    return clusterData2017

if __name__ == '__main__':
    
    clusterData2017 = myKMeans()
    clusterData2017.to_csv('C:\\Users\\Winry\\.spyder-py3\\GraduationProject\\FeatureSelection\\clusterData2017.csv',index=False)




