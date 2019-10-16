import numpy as np
import csv
import pandas as pd
import re
import math
c = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data', header = None)
df.columns = [c]
df_train = pd.DataFrame()
df_test = pd.DataFrame()
df = df.sample(frac=1).reset_index(drop = True)   #shuffle
count = len(df)
train = count*0.7
train = int(train)
df_train = df[0:train]  #train data
df_test = df[train:count]   #test data
df_set = pd.DataFrame()
df_ver = pd.DataFrame()
df_vir = pd.DataFrame()

for index, row in df_train.iterrows():
    if(row['class'].values == 'Iris-setosa'):   
        df_set = df_set.append(df_train[index:index+1])   #add to class1
    elif(row['class'].values == 'Iris-versicolor'):
        df_ver = df_ver.append(df_train[index:index+1])   #class2
    else:
        df_vir = df_vir.append(df_train[index:index+1])   #class3

df_set.index = range(0, len(df_set))
df_ver.index = range(0, len(df_ver))
df_vir.index = range(0, len(df_vir))
#print(df_set)
#print(df_ver)
#print(df_vir)
set_mean = []
ver_mean = []
vir_mean = []
set_devi = []
ver_devi = []
vir_devi = []
i = 0
colum = df_set.shape[1]-1
for col in df_set.columns:
    set_mean.append(df_set[col].mean(0))    #avg and deviation of setosa
    set_devi.append(df_set[col].std(0))
    i = i+1
    if(i == colum):
        break
i = 0
for col in df_ver.columns:
    ver_mean.append(df_ver[col].mean(0))    #ver...
    ver_devi.append(df_ver[col].std(0))
    i = i+1
    if(i == colum):
        break
i = 0
for col in df_vir.columns:
    vir_mean.append(df_vir[col].mean(0))    #the same in vir..
    vir_devi.append(df_vir[col].std(0))
    i = i+1
    if(i == colum):
        break
#print(df_set)
#print(set_mean, set_devi)
#print(ver_mean, ver_devi)
#print(vir_mean, vir_devi)
pi = math.pi
sqrt_2pi = math.sqrt(2*pi)
df_test.index = range(0, len(df_test))
#print(df_test.ix[0,0], df_test.ix[0, 4])
pre_set = [0, 0, 0] #[0] mean predict setosa, and actual setosa  [1] means predict setosa, and actual versi..   [2] mean pre set, and actual virgi..
pre_ver = [0, 0, 0]
pre_vir = [0, 0, 0]
for j in range(0, len(df_test)):
    pro1 = 1
    pro2 = 1
    pro3 = 1
    for i in range(0, 4):
        tmp1 = 1/(set_devi[i]*sqrt_2pi)*math.exp((-(df_test.ix[j,i]-set_mean[i])**2)/(2*set_devi[i]**2))
        tmp2 = 1/(ver_devi[i]*sqrt_2pi)*math.exp((-(df_test.ix[j,i]-ver_mean[i])**2)/(2*ver_devi[i]**2))
        tmp3 = 1/(vir_devi[i]*sqrt_2pi)*math.exp((-(df_test.ix[j,i]-vir_mean[i])**2)/(2*vir_devi[i]**2))
        pro1 = pro1 * tmp1  #probability in setosa
        pro2 = pro2 * tmp2  #ver..
        pro3 = pro3 * tmp3  #vir..
    #print(pro1, pro2, pro3)
    maxx = max(pro1, pro2, pro3)
    if(pro1 == maxx):
        if(df_test.ix[j, 4] == "Iris-setosa"):
            pre_set[0] += 1
        elif(df_test.ix[j, 4] == "Iris-versicolor"):
            pre_set[1] += 1
        else:
            pre_set[2] += 1
        #print("Iris-setosa", df_test.ix[j, 4])
    elif(pro2 == maxx):
        if(df_test.ix[j, 4] == "Iris-setosa"):
            pre_ver[0] += 1
        elif(df_test.ix[j, 4] == "Iris-versicolor"):
            pre_ver[1] += 1
        else:
            pre_ver[2] += 1
        #print("Iris-versicolor", df_test.ix[j, 4])
    else:
        if(df_test.ix[j, 4] == "Iris-setosa"):
            pre_vir[0] += 1
        elif(df_test.ix[j, 4] == "Iris-versicolor"):
            pre_vir[1] += 1
        else:
            pre_vir[2] += 1
        #print("Iris-virginica", df_test.ix[j, 4])
outcome = pd.DataFrame(index = ['Act Iris-setosa', 'Act versicolor', 'Act virginica'], columns = ['Pre Iris-setosa', 'Pre Iris-versicolor', 'Pre Iris-virginica'])
outcome['Pre Iris-setosa'] = [pre_set[0], pre_set[1], pre_set[2]]
outcome['Pre Iris-versicolor'] = [pre_ver[0], pre_ver[1], pre_ver[2]]
outcome['Pre Iris-virginica'] = [pre_vir[0], pre_vir[1], pre_vir[2]]
print("Confusion Matrix, Pre->Predict, Act->Actual" )

print(outcome)
acc = (pre_set[0] + pre_ver[1] + pre_vir[2])/len(df_test)
print("Accuracy: ", acc)
rec = []
rec.append(pre_set[0]/(pre_set[0] + pre_ver[0] + pre_vir[0]))
rec.append(pre_ver[1]/(pre_set[1] + pre_ver[1] + pre_vir[1]))
rec.append(pre_vir[2]/(pre_set[2] + pre_ver[2] + pre_vir[2]))
pre = []
pre.append(pre_set[0]/(pre_set[0] + pre_set[1] + pre_set[2]))
pre.append(pre_ver[1]/(pre_ver[0] + pre_ver[1] + pre_ver[2]))
pre.append(pre_vir[2]/(pre_vir[0] + pre_vir[1] + pre_vir[2]))
rec_pre = pd.DataFrame(index = ['Sensitivity(Recall)', 'Precision'], columns = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
rec_pre['Iris-setosa'] = [rec[0], pre[0]]
rec_pre['Iris-versicolor'] = [rec[1], pre[1]]
rec_pre['Iris-virginica'] = [rec[2], pre[2]]
print(rec_pre)
