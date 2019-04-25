# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:44:36 2019

@author: Winry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# 导入宋体
my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")

data2017 = pd.read_csv("data20117.csv")
data2018 = pd.read_csv("data2018.csv")
tarinData2017 = data2017.sample(n=20000)
testData2018 = data2018.sample(n=20000)

X_train = tarinData2017.iloc[:,0:-1]
y_train = tarinData2017["ArrDelay"]


X_test = testData2018.iloc[:,0:-1]
y_test = testData2018.iloc[:,-1]


n_folds = 6  # 设置交叉检验的次数
model_gbr = GradientBoostingClassifier()
scores = cross_val_score(model_gbr, X_train, y_train, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验

pre_y_list = model_gbr.fit(X_train, y_train).predict(X_train)  # 将回归训练中得到的预测y存入列表


def model_count(X_test,y_test,model,num):
    counts = 0
    cc = model.predict(X_test)
    for i in range(len(y_test)):
        if (cc[i] <= y_test.iloc[i]+num)&(cc[i] >= y_test.iloc[i]-num):
            counts += 1
    return cc,counts

model_gbr.fit(X_train, y_train)
[gbr_cc,gbr_count] = model_count(X_test,y_test,model_gbr,3)


print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, pre_y_list))
print(' 交叉检验准确率为 : %.4g '% np.mean(scores))