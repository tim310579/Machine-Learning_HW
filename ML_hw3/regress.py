import numpy as np
import csv
import pandas as pd
import re
import math
import random
import matplotlib.pyplot as plt

def X2():
    df = pd.read_csv('linear_data.txt', header = None)
    c = ['x', 'y']
    df.columns = c
    B = np.array([df['y']])
    dfx = np.array([df['x']])
    siz = len(df)
    one = np.ones([1, siz])
    one = one.T
    dfx = dfx.T
    A = np.hstack((one, dfx))
    #print(dfm)
    AT = A.T
    a00 = 0
    a01 = 0
    a10 = 0
    a11 = 0
    for i in range(0, siz):
        a00 += AT[0][i] * A[i][0]
        a01 += AT[0][i] * A[i][1]
        a10 += AT[1][i] * A[i][0]
        a11 += AT[1][i] * A[i][1]
    #print(siz)
    ATA = np.array([[a00, a01], [a10, a11]])
    #print(ATA)
    det = a00*a11 - a01*a10
    ai00 = a11/det
    ai01 = -a01/det
    ai10 = -a10/det
    ai11 = a00/det
    ATAI = np.array([[ai00, ai01],[ai10, ai11]])
    #print(ATA)
    #print(ATAI) #(AT A)-1

    ATAIAT = [[]*siz]*2
    ataiat0 = []
    ataiat1 = []
    ataiat10 = 0
    ataiat11 = 0
    for i in range(0, siz):
        ataiat0.append(ATAI[0][0]*AT[0][i]+ATAI[0][1]*AT[1][i])
        ataiat1.append(ATAI[1][0]*AT[0][i]+ATAI[1][1]*AT[1][i])

    B = B.T
    #print(B)
    tmp0 = 0
    tmp1 = 0
    for i in range(0, siz):
        tmp0 += ataiat0[i]*B[i][0]
        tmp1 += ataiat1[i]*B[i][0]
    #print(tmp0, tmp1)
    X = df['x']
    Y = df['y']
    plt.plot(X,Y, 'o')
    x = np.arange(-5, 5)
    y = (tmp0+tmp1*x)
    print('Fitting line:', tmp1, 'X^1 + ', tmp0)
    err = 0;
    for i in range(0, siz):
        err += pow(((tmp0+tmp1*X[i])-Y[i]), 2)
    print(err)
    plt.plot(x,y)

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("For X2")
    #plt.show()
def compute_det2(ATA):
    ret = ATA[0][0]*ATA[1][1] - ATA[1][0]*ATA[0][1]
    return ret
def compute_det3(ATA):
    pos = 0
    pos += ATA[0][0]*ATA[1][1]*ATA[2][2]
    pos += ATA[0][1]*ATA[1][2]*ATA[2][0]
    pos += ATA[0][2]*ATA[1][0]*ATA[2][1]
    neg = 0
    neg += ATA[0][2]*ATA[1][1]*ATA[2][0]
    neg += ATA[0][0]*ATA[1][2]*ATA[2][1]
    neg += ATA[0][1]*ATA[1][0]*ATA[2][2]
    det = pos - neg
    return det
def X3():
    df = pd.read_csv('linear_data.txt', header = None)
    c = ['x', 'y']
    df.columns = c
    dfx = np.array([df['x']])
    B = np.array([df['y']])
    dfx2 = pow(dfx, 2)
    dfx = dfx.T
    dfx2 = dfx2.T
    B = B.T
    siz = len(df)
    one = np.ones([1, siz])
    one = one.T
    A = np.hstack((one, dfx, dfx2))
    AT = A.T
    #print(A, AT)
    ATA = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype = np.float)
    #ATA = np.array([[0,0],[0,0]], dtype = np.float)
    #for i in range(0, siz):
    print(ATA.shape)
    for j in range(3):
        for k in range(3):
            for i in range(siz):
                ATA[j][k] += AT[j][i] * A[i][k]

    print(ATA)
    #print(compute_inv2(ATA).dot(ATA))
    det = compute_det2(ATA)
    print(det)
    #print(np.linalg.det(ATA))
    

X2()
X3()
