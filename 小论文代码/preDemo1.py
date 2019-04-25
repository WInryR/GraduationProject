# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:46:49 2019

@author: Winry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import kstest
import seaborn as sns
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")

new2017 = pd.read_csv('clusterData2017.csv')
data = new2017["ArrDelay"]
u = new2017["ArrDelay"].mean()

std = new2017["ArrDelay"].std()

kstest(new2017["ArrDelay"],'norm',(u,std))
x_d = np.linspace(-238,1773,120000)
newdata = data.values
kde = KernelDensity(bandwidth=1.0,kernel='gaussian')
kde.fit(newdata[:,None])

logprob = kde.score_samples(newdata[:,None])

plt.fill_between(x_d,np.exp(logprob),alpha=0.5)
plt.plot(newdata,np.full_like(newdata,-0.01),'|k',markeredgewidth=1)



sns.set(color_codes=True)
sns.set_style("white")

p1 = sns.kdeplot(new2017["ArrDelay"], shade=True, bw=.5, color="red")
p1 = sns.kdeplot(new2017["DelyaCluster"], shade=True, bw= 5, color="blue")




fig = plt.figure(figsize = (10,6))
ax1 = fig.add_subplot(2,1,1)
data.plot(kind = 'kde',grid = True,style = 'k')
plt.axvline(3*std,hold=None,color='r',linestyle="--",alpha=0.8) #3倍的标准差
plt.axvline(-3*std,hold=None,color='r',linestyle="--",alpha=0.8) 


error = data[np.abs(data - u) > 3*std]  #超过3倍差的数据（即异常值）筛选出来
data_c = data[np.abs(data - u) < 3*std] 
print('异常值共%i条' % len(error))
ax2 = fig.add_subplot(2, 1, 2)

plt.scatter(data_c.index, data_c, color = 'k', marker = '.', alpha = 0.3)
plt.scatter(error.index, error, color = 'r', marker = '.', alpha = 0.7)
#plt.xlim([-10,10010])
plt.grid()


accuary = pd.DataFrame([data_c.index,data_c])
accuary = accuary.T

error = pd.DataFrame([error.index,error]).T



accuary.to_csv('C:\\Users\\Winry\\.spyder-py3\\GraduationProject\\FeatureSelection\\accuary.csv',index=False)
error.to_csv('C:\\Users\\Winry\\.spyder-py3\\GraduationProject\\FeatureSelection\\error.csv',index=False)





'''
检验


kkk = pd.read_csv('clusterData2017.csv')

import seaborn as sns
sns.kdeplot(kkk["DelyaCluster"],shade = True)


plt.hist(kkk["DelyaCluster"],normed = True ,alpha = 0.5)

from scipy.stats import kstest

kstest(kkk["DelyaCluster"], 'norm')
'''

