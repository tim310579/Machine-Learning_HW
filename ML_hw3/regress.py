import numpy as np
import csv
import pandas as pd
import re
import math
import random
import matplotlib.pyplot as plt

df = pd.read_csv('linear_data.txt', header = None)
c = ['x', 'y']
df.columns = c
#df.readlines()
#dfx = df['x'].values
B = np.array([df['y']])
dfx = np.array([df['x']])
siz = len(df)
one = np.ones([1, siz])
one = one.T
dfx = dfx.T
#print(one.shape, dfx.shape)
#print(one)
#dfm = dft*df
#print(dfx)
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
#    ATAIAT[0].append(ATAI[0][0]*AT[0][i]+ATAI[0][1]*AT[1][i])
#    ATAIAT[1].append(ATAI[1][0]*AT[0][i]+ATAI[1][1]*AT[1][i])
#print(ataiat0)
#print(ataiat1)

#ATAIATB = np.array([[0],[0]])

#print(ATAIAT)
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
plt.title("The Title")
plt.show()

