import numpy as np
import csv
import pandas as pd
import re
import math 
import random
import matplotlib.pyplot as plt
from math import log, inf

def sigmoid(w, x):
    x = x.T
    ret = 0
    for i in range(3):
        ret += w[i]*x[i]
    
    ans = 0
  
    ans = np.exp(-ret)
   
    return (1/(1+ans))

def cross(data, w):
    g = np.zeros(len(w))
    for i in range(data.shape[0]):
        x = np.array(data[i])
        #print(x,'tt')
        error = sigmoid(w, x)
        for j in range(3):          
            g[j] += (error - x[3])*x[j] #x[3] is y
    
    return g/len(data)
    
def L2_norm(data, w):
    g = np.zeros(len(w))
    for i in range(len(data)):
        x = np.array(data[i])
        error = sigmoid(w, x)
        for j in range(3):
            g[j] += (x[3]-error)*error*(1-error)*x[j]

    return g

def log(name, df, df2):
    
    c = ['x', 'y']
    c2 = ['x2', 'y2']
    df.columns = c
    df2.columns = c2
    x = np.array([df['x']])
    y = np.array([df['y']])
    x2 = np.array([df2['x2']])
    y2 = np.array([df2['y2']])
    x = x.T
    y = y.T
    x2 = x2.T
    y2 = y2.T
    one = np.ones([1, len(df)])
    one = one.T
    zero = np.zeros([1, len(df)])
    zero = zero.T
    A = np.hstack((one, x, y, zero))
    A2 = np.hstack((one, x2, y2, one))
    data = np.vstack((A, A2))

    plt.plot(x, y, 'o',c = 'orange', label = '0')
    plt.plot(x2, y2,'o', c = 'blue', label = '1')
    w1 = np.array([0, 0, 0])
    limit = 0
    eta = 0.03
    costs = []
    current_cost = 0
    while 1:
        limit += 1
        cros = cross(data, w1)
        norm = L2_norm(data, w1)
        w1 = w1 + eta * norm
        if( abs(norm[1])<=0.001 and norm[2] <= 0.001):
            #print(limit)
            break

    X = np.arange(-6, 15)
    Y = (-w1[0]/w1[2] - w1[1]*X/w1[2])
    plt.plot(X, Y, c = 'r', label = 'L2_norm')

    w2 = np.array([0, 0, 0])
    limit = 0
    eta = 0.03
    costs = []
    current_cost = 0
    while 1:
        limit += 1
        cros = cross(data, w2)
        #print(cros)
        w2 = w2 - eta * cros
        if abs(cros[1]) <= 0.001 and abs(cros[2]) <= 0.001:
            #print(limit)
            break
    
    X = np.arange(-6, 15)
    Y = (-w2[0]/w2[2] - w2[1]*X/w2[2])
    plt.plot(X, Y, c = 'g', label = 'cross entropy')
    plt.legend(loc = 'lower right')
    plt.title(u'%s'%name, fontsize=17)
    #plt.show()
    plt.savefig('%s classify.png'%name)
    plt.close('all')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tmp_x1 = []
    tmp_y1 = []
    tmp_x0 = []
    tmp_y0 = []
    for i in range(len(data)):
        x = data[i]
        tmp = w1[0]*x[0]+w1[1]*x[1]+w1[2]*x[2]
        if tmp > 0:
            tmp_x1.append(x[1])
            tmp_y1.append(x[2])
            if x[3] == 1: tp += 1
            else: fp += 1 #guess 1, actual 0
        else:
            tmp_x0.append(x[1])
            tmp_y0.append(x[2])
            if x[3] == 1: fn += 1
            else: tn += 1 #guess 0, actual 0
    plt.plot(tmp_x1, tmp_y1, 'o')
    plt.plot(tmp_x0, tmp_y0, 'o')
    plt.savefig('%s L2 norm.png'%name)
    plt.close('all')
    res_L2 = np.array([[tp, fn], [fp, tn]])
    df_L2 = pd.DataFrame(res_L2, index = ['Actual 1', 'Actual 0'], columns = ['Predict 1', 'Predict 0'])
    print('L2_norm weight :', w1)
    print('L2_norm matrix')
    print(df_L2)
    acc = (tp+tn)/len(data)
    rec = tp/(tp+fn)
    pre = tp/(tp+fp)
    print('Accuracy :',acc)
    print('Recall :',rec)
    print('Precision :',pre)
    print('')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tmp_x1 = []
    tmp_y1 = []
    tmp_x0 = []
    tmp_y0 = []
    for i in range(len(data)):
        x = data[i]
        tmp = w2[0]*x[0]+w2[1]*x[1]+w2[2]*x[2]
        if tmp > 0:
            tmp_x1.append(x[1])
            tmp_y1.append(x[2])
            if x[3] == 1: tp += 1
            else: fp += 1 #guess 1, actual 0
        else:
            tmp_x0.append(x[1])
            tmp_y0.append(x[2])
            if x[3] == 1: fn += 1
            else: tn += 1 #guess 0, actual 0
    plt.plot(tmp_x1, tmp_y1, 'o')
    plt.plot(tmp_x0, tmp_y0, 'o')
    plt.savefig('%s Cross entropy.png'%name)
    plt.close('all')
    
    res_cross = np.array([[tp, fn], [fp, tn]])
    df_cross = pd.DataFrame(res_cross, index = ['Actual 1', 'Actual 0'], columns = ['Predict 1', 'Predict 0'])
    print('Cross entropy weight :', w2)
    print('Cross entropy matrix')
    print(df_cross)
    acc = (tp+tn)/len(data)
    rec = tp/(tp+fn)
    pre = tp/(tp+fp)
    print('Accuracy :',acc)
    print('Recall :',rec)
    print('Precision :',pre)
    print('')
    
df1_1 = pd.read_csv('Logistic_data1-1.txt', header = None)
df1_2 = pd.read_csv('Logistic_data1-2.txt', header = None)
print('For data1----------------------------------------------------------------')
print('')
log('log_data1', df1_1, df1_2)

df2_1 = pd.read_csv('Logistic_data2-1.txt', header = None)
df2_2 = pd.read_csv('Logistic_data2-2.txt', header = None)
print('For data2----------------------------------------------------------------')
print('')
log('log_data2', df2_1, df2_2)
