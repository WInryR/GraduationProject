# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:28:12 2019

@author: Winry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

# 导入宋体
my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")

data2017 = pd.read_csv("clusterData2017.csv")
data2018 = pd.read_csv("data2018.csv")
tarinData2017 = data2017.sample(n=30000)
testData2018 = data2018.sample(n=30000)


X_train = tarinData2017.iloc[:,1:-2]
y_train = tarinData2017["DelyaCluster"]


X_test = testData2018.iloc[:,0:-1]
y_test_num = testData2018.iloc[:,-1]


y_test = np.zeros_like(y_test_num)
for i in range(len(y_test)):
    if y_test_num.iloc[i] <= 3:
        y_test[i] = 0
    elif (y_test_num.iloc[i] > 3) & (y_test_num.iloc[i] <= 53):
        y_test[i] = 4
    elif (y_test_num.iloc[i] > 53) & (y_test_num.iloc[i] <= 155):
        y_test[i] = 2
    elif y_test_num.iloc[i] >= 473:
        y_test[i] = 1
    else:
        y_test[i] = 3





n_folds = 6  # 设置交叉检验的次数
model_gbr = GradientBoostingClassifier()
scores = cross_val_score(model_gbr, X_train, y_train, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验

pre_y_list = model_gbr.fit(X_train, y_train).predict(X_test)  # 将回归训练中得到的预测y存入列表



print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, pre_y_list))
print(' 交叉检验准确率为 : %.4g '% np.mean(scores))







data201718 = pd.concat([X_train,X_test])

pca = PCA().fit(data201718)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(0.95).fit(data201718)
pca.n_components_
newdata201718 = pca.transform(data201718)
filtered = pca.inverse_transform(newdata201718)

newX_train = filtered[0:30000,:]

newX_test = filtered[30000:,:]


n_folds = 6  # 设置交叉检验的次数
model_gbr = GradientBoostingClassifier()
scores = cross_val_score(model_gbr, newX_train, y_train, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验

newPre_y_list = model_gbr.fit(newX_train, y_train).predict(newX_test)  # 将回归训练中得到的预测y存入列表



print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, newPre_y_list))
print(' 交叉检验准确率为 : %.4g '% np.mean(scores))


