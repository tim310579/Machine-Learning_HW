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
#df = df.sample(frac=1).reset_index(drop = True)   #shuffle
df_set = pd.DataFrame()
df_ver = pd.DataFrame()
df_vir = pd.DataFrame()

for index, row in df.iterrows():
    if(row['class'].values == 'Iris-setosa'):   
        df_set = df_set.append(df[index:index+1])   #add to class1
    elif(row['class'].values == 'Iris-versicolor'):
        df_ver = df_ver.append(df[index:index+1])   #class2
    else:
        df_vir = df_vir.append(df[index:index+1])   #class3
df_set = df_set.sample(frac=1).reset_index(drop = True) #shuffle setosa
df_ver = df_ver.sample(frac=1).reset_index(drop = True) #shuffle ver..
df_vir = df_vir.sample(frac=1).reset_index(drop = True) #shuffle vir..
df_set.index = range(0, len(df_set))
df_ver.index = range(0, len(df_ver))
df_vir.index = range(0, len(df_vir))

count1 = int(len(df_set)*0.7)
count2 = int(len(df_ver)*0.7)
count3 = int(len(df_vir)*0.7)
df_train_set = df_set[0:count1]
df_train_ver = df_ver[0:count2]
df_train_vir = df_vir[0:count3]
df_test_set = df_set[count1:len(df_set)]
df_test_ver = df_ver[count1:len(df_ver)]
df_test_vir = df_vir[count1:len(df_vir)]
df_test = pd.concat([df_test_set, df_test_ver, df_test_vir], axis = 0) #testdata
set_mean = []
ver_mean = []
vir_mean = []
set_devi = []
ver_devi = []
vir_devi = []
i = 0
colum = df_set.shape[1]-1
for col in df_set.columns:
    set_mean.append(df_train_set[col].mean(0))    #avg and deviation of setosa
    set_devi.append(df_train_set[col].std(0))
    i = i+1
    if(i == colum):
        break
i = 0
for col in df_ver.columns:
    ver_mean.append(df_train_ver[col].mean(0))    #ver...
    ver_devi.append(df_train_ver[col].std(0))
    i = i+1
    if(i == colum):
        break
i = 0
for col in df_vir.columns:
    vir_mean.append(df_train_vir[col].mean(0))    #the same in vir..
    vir_devi.append(df_train_vir[col].std(0))
    i = i+1
    if(i == colum):
        break

#print(df_set)
'''with open("mean_devi.data", "w") as f:
    for it in set_mean:
        f.write("%s   "% it)
    f.write("\n")
    for it in set_devi:
        f.write("%s   "% it)
    f.write("\n")
    for it in ver_mean:
        f.write("%s   "% it)
    f.write("\n")
    for it in ver_devi:
        f.write("%s   "% it)
    f.write("\n")
    for it in vir_mean:
        f.write("%s   "% it)
    f.write("\n")
    for it in vir_devi:
        f.write("%s   "% it)
'''
#print(set_mean, set_devi)
#print(ver_mean, ver_devi)
#print(vir_mean, vir_devi)
pi = math.pi
sqrt_2pi = math.sqrt(2*pi)
df_test.index = range(0, len(df_test))
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
print("Confusion Matrix for holdout validation, Pre->Predict, Act->Actual--------------")

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
print('')
'''
K-fold
'''
dfk = df.copy()
dfk = dfk.sample(frac=1).reset_index(drop = True)
k1 = (len(dfk)/3)
k1 = int(k1)
k2 = k1*2
#print(k1)
df_k0 = dfk[0:k1]   #can choose 1 to be testdata
df_k1 = dfk[k1:k2]
df_k2 = dfk[k2:len(dfk)]
df_k0.index = range(0, len(df_k0))
df_k1.index = range(0, len(df_k1))
df_k2.index = range(0, len(df_k2))
df_k0_set = pd.DataFrame()
df_k0_ver = pd.DataFrame()
df_k0_vir = pd.DataFrame()
df_k1_set = pd.DataFrame()
df_k1_ver = pd.DataFrame()
df_k1_vir = pd.DataFrame()
df_k2_set = pd.DataFrame()
df_k2_ver = pd.DataFrame()
df_k2_vir = pd.DataFrame()
for index, col in df_k0.iterrows():
    if(col['class'].values == "Iris-setosa"):
        df_k0_set = df_k0_set.append(df_k0[index:index+1])  #choose data is setosa. in k0
    elif(col['class'].values == "Iris-versicolor"):
        df_k0_ver = df_k0_ver.append(df_k0[index:index+1])  #choose data is versi.. in k0
    else:
        df_k0_vir = df_k0_vir.append(df_k0[index:index+1])  #virgi
for index1, col1 in df_k1.iterrows():
    if(col1['class'].values == "Iris-setosa"):
        df_k1_set = df_k1_set.append(df_k1[index1:index1+1])
    elif(col1['class'].values == "Iris-versicolor"):
        df_k1_ver = df_k1_ver.append(df_k1[index1:index1+1])
    else:
        df_k1_vir = df_k1_vir.append(df_k1[index1:index1+1])

for index, col in df_k2.iterrows():
    if(col['class'].values == "Iris-setosa"):
        df_k2_set = df_k2_set.append(df_k2[index:index+1])
    elif(col['class'].values == "Iris-versicolor"):
        df_k2_ver = df_k2_ver.append(df_k2[index:index+1])
    else:
        df_k2_vir = df_k2_vir.append(df_k2[index:index+1])
df_k0_set.index = range(0, len(df_k0_set))  #reindex
df_k0_ver.index = range(0, len(df_k0_ver))
df_k0_vir.index = range(0, len(df_k0_vir))
df_k1_set.index = range(0, len(df_k1_set))
df_k1_ver.index = range(0, len(df_k1_ver))
df_k1_vir.index = range(0, len(df_k1_vir))
df_k2_set.index = range(0, len(df_k2_set))
df_k2_ver.index = range(0, len(df_k2_ver))
df_k2_vir.index = range(0, len(df_k2_vir))

#print(df_k0_set, df_k0_ver, df_k0_vir)
#print(df_k1_set, df_k1_ver, df_k1_vir)
#print(df_k2_set, df_k2_ver, df_k2_vir)

df_k01_set = pd.concat([df_k0_set, df_k1_set], axis = 0)    #merge k0,k1 set
df_k01_ver = pd.concat([df_k0_ver, df_k1_ver], axis = 0)    #....ver
df_k01_vir = pd.concat([df_k0_vir, df_k1_vir], axis = 0)    #....vir
#print(df_k01_set, df_k01_ver, df_k01_vir)
df_k12_set = pd.concat([df_k1_set, df_k2_set], axis = 0)    #merge k2,k1 set
df_k12_ver = pd.concat([df_k1_ver, df_k2_ver], axis = 0)    #....ver
df_k12_vir = pd.concat([df_k1_vir, df_k2_vir], axis = 0)    #....vir
df_k02_set = pd.concat([df_k0_set, df_k2_set], axis = 0)    #merge k0,k2 set
df_k02_ver = pd.concat([df_k0_ver, df_k2_ver], axis = 0)    #....ver
df_k02_vir = pd.concat([df_k0_vir, df_k2_vir], axis = 0)    #....vir
set_mean_devi01 = [] #mean, deviation for merge k0,k1
ver_mean_devi01 = []
vir_mean_devi01 = []
set_mean_devi12 = []
ver_mean_devi12 = []
vir_mean_devi12 = []
set_mean_devi02 = []
ver_mean_devi02 = []
vir_mean_devi02 = []
i = 0
colun = df_k01_set.shape[1]-1
for col in df_k01_set.columns:
    set_mean_devi01.append(df_k01_set[col].mean(0))   #avg
    set_mean_devi01.append(df_k01_set[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
i = 0
colun = df_k01_ver.shape[1]-1
for col in df_k01_ver.columns:
    ver_mean_devi01.append(df_k01_ver[col].mean(0))   #avg
    ver_mean_devi01.append(df_k01_ver[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
i = 0
colun = df_k01_vir.shape[1]-1
for col in df_k01_vir.columns:
    vir_mean_devi01.append(df_k01_vir[col].mean(0))   #avg
    vir_mean_devi01.append(df_k01_vir[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
#df_k01 mean, std for 3 class is ok
i = 0
colun = df_k12_set.shape[1]-1
for col in df_k12_set.columns:
    set_mean_devi12.append(df_k12_set[col].mean(0))   #avg
    set_mean_devi12.append(df_k12_set[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
i = 0
colun = df_k12_ver.shape[1]-1
for col in df_k12_ver.columns:
    ver_mean_devi12.append(df_k12_ver[col].mean(0))   #avg
    ver_mean_devi12.append(df_k12_ver[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
i = 0
colun = df_k12_vir.shape[1]-1
for col in df_k12_vir.columns:
    vir_mean_devi12.append(df_k12_vir[col].mean(0))   #avg
    vir_mean_devi12.append(df_k12_vir[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
#k12 ok
i = 0
colun = df_k02_set.shape[1]-1
for col in df_k02_set.columns:
    set_mean_devi02.append(df_k02_set[col].mean(0))   #avg
    set_mean_devi02.append(df_k02_set[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
i = 0
colun = df_k02_ver.shape[1]-1
for col in df_k02_ver.columns:
    ver_mean_devi02.append(df_k02_ver[col].mean(0))   #avg
    ver_mean_devi02.append(df_k02_ver[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
i = 0
colun = df_k02_vir.shape[1]-1
for col in df_k02_vir.columns:
    vir_mean_devi02.append(df_k02_vir[col].mean(0))   #avg
    vir_mean_devi02.append(df_k02_vir[col].std(0))   #deviation
    i = i + 1
    if(i == colun):
        break
pi = math.pi
sqrt_2pi = math.sqrt(2*pi)
pre_set0 = [0, 0, 0] #[0] mean predict setosa, and actual setosa  [1] means predict setosa, and actual versi..   [2] mean pre set, and actual virgi..
pre_ver0 = [0, 0, 0]
pre_vir0 = [0, 0, 0]
#take k0 for test
for j in range(0, len(df_k0)):
    pro1 = 1
    pro2 = 1
    pro3 = 1
    for i in range(0, 4):
        tmp1 = 1/(set_mean_devi12[2*i+1]*sqrt_2pi)*math.exp((-(df_k0.ix[j,i]-set_mean_devi12[2*i])**2)/(2*set_mean_devi12[2*i+1]**2))
        tmp2 = 1/(ver_mean_devi12[2*i+1]*sqrt_2pi)*math.exp((-(df_k0.ix[j,i]-ver_mean_devi12[2*i])**2)/(2*ver_mean_devi12[2*i+1]**2))
        tmp3 = 1/(vir_mean_devi12[2*i+1]*sqrt_2pi)*math.exp((-(df_k0.ix[j,i]-vir_mean_devi12[2*i])**2)/(2*vir_mean_devi12[2*i+1]**2))
        pro1 *= tmp1
        pro2 *= tmp2
        pro3 *= tmp3
    maxx = max(pro1, pro2, pro3)
    if(pro1 == maxx):
        if(df_k0.ix[j, 4] == "Iris-setosa"):
            pre_set0[0] += 1
        elif(df_k0.ix[j, 4] == "Iris-versicolor"):
            pre_set0[1] += 1
        else:
            pre_set0[2] += 1
        #print("Iris-setosa", df_test.ix[j, 4])
    elif(pro2 == maxx):
        if(df_k0.ix[j, 4] == "Iris-setosa"):
            pre_ver0[0] += 1
        elif(df_k0.ix[j, 4] == "Iris-versicolor"):
            pre_ver0[1] += 1
        else:
            pre_ver0[2] += 1
        #print("Iris-versicolor", df_test.ix[j, 4])
    else:
        if(df_k0.ix[j, 4] == "Iris-setosa"):
            pre_vir0[0] += 1
        elif(df_k0.ix[j, 4] == "Iris-versicolor"):
            pre_vir0[1] += 1
        else:
            pre_vir0[2] += 1
outcome_k0 = pd.DataFrame(index = ['Act Iris-setosa', 'Act versicolor', 'Act virginica'], columns = ['Pre Iris-setosa', 'Pre Iris-versicolor', 'Pre Iris-virginica'])
outcome_k0['Pre Iris-setosa'] = [pre_set0[0], pre_set0[1], pre_set0[2]]
outcome_k0['Pre Iris-versicolor'] = [pre_ver0[0], pre_ver0[1], pre_ver0[2]]
outcome_k0['Pre Iris-virginica'] = [pre_vir0[0], pre_vir0[1], pre_vir0[2]]
#print("Confusion Matrix for k0 be testdata, Pre->Predict, Act->Actual----------------")
#print(outcome_k0)
acc0 = (pre_set0[0] + pre_ver0[1] + pre_vir0[2])/len(df_k0)
#print("Accuracy: ", acc0)
rec0 = []
rec0.append(pre_set0[0]/(pre_set0[0] + pre_ver0[0] + pre_vir0[0]))
rec0.append(pre_ver0[1]/(pre_set0[1] + pre_ver0[1] + pre_vir0[1]))
rec0.append(pre_vir0[2]/(pre_set0[2] + pre_ver0[2] + pre_vir0[2]))
pre0 = []
pre0.append(pre_set0[0]/(pre_set0[0] + pre_set0[1] + pre_set0[2]))
pre0.append(pre_ver0[1]/(pre_ver0[0] + pre_ver0[1] + pre_ver0[2]))
pre0.append(pre_vir0[2]/(pre_vir0[0] + pre_vir0[1] + pre_vir0[2]))
rec_pre0 = pd.DataFrame(index = ['Sensitivity(Recall)', 'Precision'], columns = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
rec_pre0['Iris-setosa'] = [rec0[0], pre0[0]]
rec_pre0['Iris-versicolor'] = [rec0[1], pre0[1]]
rec_pre0['Iris-virginica'] = [rec0[2], pre0[2]]
#print(rec_pre0)

#test k1

pre_set1 = [0, 0, 0] #[0] mean predict setosa, and actual setosa  [1] means predict setosa, and actual versi..   [2] mean pre set, and actual virgi..
pre_ver1 = [0, 0, 0]
pre_vir1 = [0, 0, 0]
#take k1 for test
for j in range(0, len(df_k1)):
    pro1 = 1
    pro2 = 1
    pro3 = 1
    for i in range(0, 4):
        tmp1 = 1/(set_mean_devi02[2*i+1]*sqrt_2pi)*math.exp((-(df_k1.ix[j,i]-set_mean_devi02[2*i])**2)/(2*set_mean_devi02[2*i+1]**2))
        tmp2 = 1/(ver_mean_devi02[2*i+1]*sqrt_2pi)*math.exp((-(df_k1.ix[j,i]-ver_mean_devi02[2*i])**2)/(2*ver_mean_devi02[2*i+1]**2))
        tmp3 = 1/(vir_mean_devi02[2*i+1]*sqrt_2pi)*math.exp((-(df_k1.ix[j,i]-vir_mean_devi02[2*i])**2)/(2*vir_mean_devi02[2*i+1]**2))
        pro1 *= tmp1
        pro2 *= tmp2
        pro3 *= tmp3
    maxx = max(pro1, pro2, pro3)
    if(pro1 == maxx):
        if(df_k1.ix[j, 4] == "Iris-setosa"):
            pre_set1[0] += 1
        elif(df_k1.ix[j, 4] == "Iris-versicolor"):
            pre_set1[1] += 1
        else:
            pre_set1[2] += 1
        #print("Iris-setosa", df_test.ix[j, 4])
    elif(pro2 == maxx):
        if(df_k1.ix[j, 4] == "Iris-setosa"):
            pre_ver1[0] += 1
        elif(df_k1.ix[j, 4] == "Iris-versicolor"):
            pre_ver1[1] += 1
        else:
            pre_ver1[2] += 1
        #print("Iris-versicolor", df_test.ix[j, 4])
    else:
        if(df_k1.ix[j, 4] == "Iris-setosa"):
            pre_vir1[0] += 1
        elif(df_k1.ix[j, 4] == "Iris-versicolor"):
            pre_vir1[1] += 1
        else:
            pre_vir1[2] += 1
outcome_k1 = pd.DataFrame(index = ['Act Iris-setosa', 'Act versicolor', 'Act virginica'], columns = ['Pre Iris-setosa', 'Pre Iris-versicolor', 'Pre Iris-virginica'])
outcome_k1['Pre Iris-setosa'] = [pre_set1[0], pre_set1[1], pre_set1[2]]
outcome_k1['Pre Iris-versicolor'] = [pre_ver1[0], pre_ver1[1], pre_ver1[2]]
outcome_k1['Pre Iris-virginica'] = [pre_vir1[0], pre_vir1[1], pre_vir1[2]]
#print("Confusion Matrix for k1 be testdata, Pre->Predict, Act->Actual----------------")
#print(outcome_k1)
acc1 = (pre_set1[0] + pre_ver1[1] + pre_vir1[2])/len(df_k1)
#print("Accuracy: ", acc1)
rec1 = []
rec1.append(pre_set1[0]/(pre_set1[0] + pre_ver1[0] + pre_vir1[0]))
rec1.append(pre_ver1[1]/(pre_set1[1] + pre_ver1[1] + pre_vir1[1]))
rec1.append(pre_vir1[2]/(pre_set1[2] + pre_ver1[2] + pre_vir1[2]))
pre1 = []
pre1.append(pre_set1[0]/(pre_set1[0] + pre_set1[1] + pre_set1[2]))
pre1.append(pre_ver1[1]/(pre_ver1[0] + pre_ver1[1] + pre_ver1[2]))
pre1.append(pre_vir1[2]/(pre_vir1[0] + pre_vir1[1] + pre_vir1[2]))
rec_pre1 = pd.DataFrame(index = ['Sensitivity(Recall)', 'Precision'], columns = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
rec_pre1['Iris-setosa'] = [rec1[0], pre1[0]]
rec_pre1['Iris-versicolor'] = [rec1[1], pre1[1]]
rec_pre1['Iris-virginica'] = [rec1[2], pre1[2]]
#print(rec_pre1)

#k2 test
pre_set2 = [0, 0, 0] #[0] mean predict setosa, and actual setosa  [1] means predict setosa, and actual versi..   [2] mean pre set, and actual virgi..
pre_ver2 = [0, 0, 0]
pre_vir2 = [0, 0, 0]
#take k2 for test
for j in range(0, len(df_k0)):
    pro1 = 1
    pro2 = 1
    pro3 = 1
    for i in range(0, 4):
        tmp1 = 1/(set_mean_devi01[2*i+1]*sqrt_2pi)*math.exp((-(df_k2.ix[j,i]-set_mean_devi01[2*i])**2)/(2*set_mean_devi01[2*i+1]**2))
        tmp2 = 1/(ver_mean_devi01[2*i+1]*sqrt_2pi)*math.exp((-(df_k2.ix[j,i]-ver_mean_devi01[2*i])**2)/(2*ver_mean_devi01[2*i+1]**2))
        tmp3 = 1/(vir_mean_devi01[2*i+1]*sqrt_2pi)*math.exp((-(df_k2.ix[j,i]-vir_mean_devi01[2*i])**2)/(2*vir_mean_devi01[2*i+1]**2))
        pro1 *= tmp1
        pro2 *= tmp2
        pro3 *= tmp3
    maxx = max(pro1, pro2, pro3)
    if(pro1 == maxx):
        if(df_k2.ix[j, 4] == "Iris-setosa"):
            pre_set2[0] += 1
        elif(df_k2.ix[j, 4] == "Iris-versicolor"):
            pre_set2[1] += 1
        else:
            pre_set2[2] += 1
        #print("Iris-setosa", df_test.ix[j, 4])
    elif(pro2 == maxx):
        if(df_k2.ix[j, 4] == "Iris-setosa"):
            pre_ver2[0] += 1
        elif(df_k2.ix[j, 4] == "Iris-versicolor"):
            pre_ver2[1] += 1
        else:
            pre_ver2[2] += 1
        #print("Iris-versicolor", df_test.ix[j, 4])
    else:
        if(df_k2.ix[j, 4] == "Iris-setosa"):
            pre_vir2[0] += 1
        elif(df_k2.ix[j, 4] == "Iris-versicolor"):
            pre_vir2[1] += 1
        else:
            pre_vir2[2] += 1
outcome_k2 = pd.DataFrame(index = ['Act Iris-setosa', 'Act versicolor', 'Act virginica'], columns = ['Pre Iris-setosa', 'Pre Iris-versicolor', 'Pre Iris-virginica'])
outcome_k2['Pre Iris-setosa'] = [pre_set2[0], pre_set2[1], pre_set2[2]]
outcome_k2['Pre Iris-versicolor'] = [pre_ver2[0], pre_ver2[1], pre_ver2[2]]
outcome_k2['Pre Iris-virginica'] = [pre_vir2[0], pre_vir2[1], pre_vir2[2]]
#print("Confusion Matrix for k2 be testdata, Pre->Predict, Act->Actual----------------")
#print(outcome_k2)
acc2 = (pre_set2[0] + pre_ver2[1] + pre_vir2[2])/len(df_k2)
#print("Accuracy: ", acc2)
rec2 = []
rec2.append(pre_set2[0]/(pre_set2[0] + pre_ver2[0] + pre_vir2[0]))
rec2.append(pre_ver2[1]/(pre_set2[1] + pre_ver2[1] + pre_vir2[1]))
rec2.append(pre_vir2[2]/(pre_set2[2] + pre_ver2[2] + pre_vir2[2]))
pre2 = []
pre2.append(pre_set2[0]/(pre_set2[0] + pre_set2[1] + pre_set2[2]))
pre2.append(pre_ver2[1]/(pre_ver2[0] + pre_ver2[1] + pre_ver2[2]))
pre2.append(pre_vir2[2]/(pre_vir2[0] + pre_vir2[1] + pre_vir2[2]))
rec_pre2 = pd.DataFrame(index = ['Sensitivity(Recall)', 'Precision'], columns = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
rec_pre2['Iris-setosa'] = [rec2[0], pre2[0]]
rec_pre2['Iris-versicolor'] = [rec2[1], pre2[1]]
rec_pre2['Iris-virginica'] = [rec2[2], pre2[2]]
#print(rec_pre2)
#print('')
print("Average Confusion matrix with K-fold--------------------------------------------")
outcome_avg = pd.DataFrame(index = ['Act Iris-setosa', 'Act versicolor', 'Act virginica'], columns = ['Pre Iris-setosa', 'Pre Iris-versicolor', 'Pre Iris-virginica'])
outcome_avg = (outcome_k0+outcome_k1+outcome_k2)/3
print(outcome_avg)
rec_pre_avg = pd.DataFrame(index = ['Sensitivity(Recall)', 'Precision'], columns = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
rec_pre_avg = (rec_pre0+rec_pre1+rec_pre2)/3
acc_avg = (acc0 + acc1 + acc2)/3
print("Average Accuracy:", acc_avg)
print(rec_pre_avg)
