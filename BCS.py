#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019/2/6 15:10 
# @Author : Patrick 
# @File : KNN_Digit.py 
# @Software: PyCharm


import time

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm

train_pect = 0.8
vaild_pect = 0.5


def normalize(a):
    a = a.drop(labels='Class', axis=1)
    a = preprocessing.minmax_scale(a)
    a = preprocessing.normalize(a)
    a = preprocessing.scale(a)
    return a


rows_to_skip = []
f = open('BCW.csv')
count = 0
for i in f.readlines():
    if '?' in i:
        rows_to_skip.append(count)
    count += 1
f.close()

data_set = pd.read_csv('BCW.csv', skiprows=rows_to_skip)
print(data_set)
# print(data)
# print(data.size)
print('------------------------------------------------')
label = data_set['Class']
# print(data.size)
# print(label)
m = 0
train_pect = 1
c=1
t=0
flag = False
while True:
    if not flag:
        train_pect = c/10

    X_train, X_port, y_train, y_port = train_test_split(data_set, label, random_state=1, train_size=train_pect,
                                                        test_size=1 - train_pect, stratify=label)
    X_vaild, X_test, y_vaild, y_test = train_test_split(X_port, y_port, random_state=1, train_size=vaild_pect,
                                                        test_size=1 - vaild_pect, stratify=y_port)

    X_train = normalize(X_train)
    X_vaild = normalize(X_vaild)
    X_test = normalize(X_test)

    print('Train shape:' + str(X_train.shape))
    print('Vaild shape:' + str(X_vaild.shape))
    print('Test shape:' + str(X_test.shape))
    t1 = time.time()
    classifier = svm.SVC()
    classifier.fit(X_train, y_train.values.ravel())
    if not flag:
        X = X_test
        Y = y_test
    else:
        X = X_vaild
        Y = y_vaild

    predicted = classifier.predict(X)
    t2 = time.time()
    count = 0
    for i in zip(predicted, Y):
        pred, real = i
        if real == pred:
            count += 1
    accu = count / len(Y)
    print('Training percentage: '+str(train_pect)+'. Accuracy is :'+str(accu))
    print('Training time is: '+str(t2-t1)+'\n')
    (m,t) = [(accu,train_pect),(m,t)][accu<m]
    if c==9:
        print('The max accuracy is ' + str(m) + ' the training percentage is ' + str(t)+'\n')
        train_pect = t
        flag = True
    c+=1
    if flag:
        print('The vaild accuracy is ' + str(accu))
        break


