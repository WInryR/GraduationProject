# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:59:09 2019

@author: Winry
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:23:24 2018

@author: Winry
"""

from sklearn.base import BaseEstimator,ClassifierMixin
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt
import time,datetime

class KDEClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
#        Weights = np.array([0.121,0.119,0.118,0.111,0.098,0.093,0.092,0.084,0.05,0.050,0.040,0.018,0.005])
        
#        for i in range(X.shape[1]):
#            logprobs[i] = logprobs[i]*Weights[i]
            
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
    
def secondsFrom1970(seq,name):
    temp = []
    newName = "New" + name
    for i in range(len(seq[name])):
        timeDateStr = seq[name][i]
        timeDateStr = datetime.datetime.strptime(timeDateStr, "%Y/%m/%d %H:%M")
        From1970 = time.mktime(timeDateStr.timetuple())
        temp.append(From1970)
    seq[newName] = temp
def model_count(X_test,y_test,model,num):
    cc = []
    counts = 0
    for i in range(len(X_test)):
        new_point = X_test.iloc[i,:]
        new_point = new_point.values
        new_point = np.array(new_point).reshape(1,-1)
        new_pre_y = model.predict(new_point)
        cc.append(list(np.around(new_pre_y))[0])
    for i in range(len(y_test)):
        if (cc[i] <= y_test.iloc[i]+num)&(cc[i] >= y_test.iloc[i]-num):
            counts += 1
    return cc,counts



bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
grid.fit(X_train, y_train)
scores = [(val.mean_validation_score+0.68) for val in grid.grid_scores_]


fig = plt.figure(figsize=(20,20),dpi=80)

plt.tick_params(labelsize=33)


plt.semilogx(bandwidths, scores)
plt.xlabel('窗口宽度参数',fontproperties=my_font,size=53)
plt.ylabel('准确率（%）',fontproperties=my_font,size=53)
def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

from matplotlib.ticker import FuncFormatter
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.save('1.png')

print(grid.best_params_)
print('accuracy =', grid.best_score_)

# best bandwidths = [5.09413801481638]
#grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
#y_pre = grid.predict(X_test)

model = KDEClassifier(bandwidth = 5)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
counts = 0
num = 3
for i in range(len(y_test)):
        if (y_pre[i] == y_test):
            counts += 1
counts

#y_pred = grid.predict(X_test)
#gbr_count = 0
#for i in range(len(y_test)):
#    if (y_pred[i] <= y_test.iloc[i]+2)&(y_pred[i] >= y_test.iloc[i]-2):
##    if (cc[i][0] == y_test.iloc[i]):
#        gbr_count += 1


from sklearn import naive_bayes
model =  naive_bayes.GaussianNB()  # 高斯贝叶斯
model.fit(X_train,y_train)
predict=model.predict(X_test)

a = 0
for i in range(len(predict)):

    if predict[i] == y_test[i]:
        a = a+1



score=float(a)/len(predict)


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)


predict=clf.predict(X_test)

a = 0
for i in range(len(predict)):

    if predict[i] == y_test[i]:
        a = a+1
        
score = float(a)/len(predict)
score













