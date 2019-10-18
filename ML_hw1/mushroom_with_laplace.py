import numpy as np
import csv
import pandas as pd
import re
import math
c = ['eaten','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-numbe','ring-type','spore-print-color','population','habitat']
#df = pd.read_csv('agaricus-lepiota.data')
#count = df.groupby('p').size()
#sr = pd.Series(df, index = c)
df = pd.read_csv('agaricus-lepiota.data', header = None)
df.columns = [c]
for col in df.columns:
    delete = df[df[col] == "?"].index
    df.drop(delete, inplace = True)
#df = df.sample(frac=1).reset_index(drop = True)   #shuffle
df_e = pd.DataFrame()
df_e = df.copy()
df_p = df.copy()

for col in df_e.columns:
    delete = df_e[df_e[col] == "p"].index
    df_e.drop(delete, inplace = True)
    break
for col in df_p.columns:
    delete = df_p[df_p[col] == "e"].index
    df_p.drop(delete, inplace = True)
    break
df_e = df_e.sample(frac=1).reset_index(drop = True)   #shuffle
df_p = df_p.sample(frac=1).reset_index(drop = True)   #shuffle
count_e = len(df_e)*0.7
count_e = int(count_e)
count_p = len(df_p)*0.7
count_p = int(count_p)
df_teste = pd.DataFrame()
df_testp = pd.DataFrame()
df_teste = df_e[count_e:len(df_e)]  #test data for 'e'
df_testp = df_p[count_p:len(df_p)]  #....'p'
df_test = pd.concat([df_teste, df_testp], axis = 0)
df_test.index = range(0, len(df_test))
df_e = df_e[0: count_e] #7/10 for train e
df_p = df_p[0: count_p] #'p'
epep = []
epep.append(count_e/(count_e + count_p))
epep.append(count_p/(count_e + count_p))
tmp_test0 = pd.DataFrame()
#print(len(df_e), len(df_p))
for col in df_e.columns:
    tmp_test0 = tmp_test0.append(df_e[col].value_counts())  #numbers in cond of etible
tmp_test = pd.DataFrame()
for col in df_p.columns:
    tmp_test = tmp_test.append(df_p[col].value_counts())    #nums in condition of poison    
tmp_test0 = tmp_test0.fillna(0)
tmp_test = tmp_test.fillna(0)
tmp_test0 = (tmp_test0 + 3)/ (len(df_e) + 3*22)
tmp_test = (tmp_test + 3)/ (len(df_p) + 3*22)

tmp_test.index = range(0, len(tmp_test))
tmp_test0.index = range(0, len(tmp_test0))
tmp_test = tmp_test.fillna(0)  #poison
tmp_test0 = tmp_test0.fillna(0) #etible
cnt = 0
cnt2 = 0
i = 0
TP = 0
TN = 0
FP = 0
FN = 0
for y in range (0, len(df_test)):
    ft1 = 0
    ft2 = 0
    for ik in range(1, 23):
        #print(df_test.iat[y, ik])
        ft1 += math.log(tmp_test.at[ik, df_test.iat[y, ik]])
        ft2 += math.log(tmp_test0.at[ik, df_test.iat[y, ik]])
        #ft1 = ft1*tmp_test.at[ik, df_test.iat[y, ik]]
        #ft2 = ft2*tmp_test0.at[ik, df_test.iat[y, ik]]
    #$if(df_test.iat[y,0] == 'e'):
    #ft2 = ft2/epep[0]   #eat
    ft2 += math.log(epep[0])
    #ft1 = ft1/epep[1]    #poisin
    ft1 += math.log(epep[1])
    if(ft1 <= ft2):
        predict = 'e'
        if(predict == df_test.iat[i, 0]):
            TP = TP + 1
        else:
            FP = FP + 1
    else:
        predict = 'p'
        if(predict == df_test.iat[i, 0]):
            TN = TN + 1
        else:
            FN = FN + 1
    i = i + 1
#print(TP, FN, FP, TN)
outcome = pd.DataFrame(index = ['Actual Positive(etible)', 'Actual Negative(poison)'], columns = ['Predict Positive(etible)', 'Predict negative(poison)'])
outcome['Predict Positive(etible)'] = [TP, FP]
outcome['Predict negative(poison)'] = [FN, TN]
print("Confusion matrix with Holdout validation-------------------------------")
print(outcome)
acc = (TP+TN) / (TP+TN+FP+FN)
rec = TP/(TP+FN)
pre = TP/(TP+FP)
print("Accuracy:", acc)
print("Sensitivity(Recall):", rec)
print("Precision:", pre)
print('')
'''
K-fold
'''
dfk = df.copy()
dfk = dfk.sample(frac=1).reset_index(drop = True)
k1 = len(dfk)/3
k1 = int(k1)
k2 = k1*2
#print(k1)
df_k0 = dfk[0:k1]   #can choose 1 to be testdata
df_k1 = dfk[k1:k2]
df_k2 = dfk[k2:len(dfk)]
#print(df_k0.head())
df_k0e = df_k0.copy()
df_k0p = df_k0.copy()
df_k1e = df_k1.copy()
df_k1p = df_k1.copy()
df_k2e = df_k2.copy()
df_k2p = df_k2.copy()

for col in df_k0e.columns:
    delete = df_k0e[df_k0e[col] == "p"].index
    df_k0e.drop(delete, inplace = True)
    break
for col in df_k0p.columns:
    delete = df_k0p[df_k0p[col] == "e"].index
    df_k0p.drop(delete, inplace = True)
    break
for col in df_k1e.columns:
    delete = df_k1e[df_k1e[col] == "p"].index
    df_k1e.drop(delete, inplace = True)
    break
for col in df_k1p.columns:
    delete = df_k1p[df_k1p[col] == "e"].index
    df_k1p.drop(delete, inplace = True)
    break
for col in df_k2e.columns:
    delete = df_k2e[df_k2e[col] == "p"].index
    df_k2e.drop(delete, inplace = True)
    break
for col in df_k2p.columns:
    delete = df_k2p[df_k2p[col] == "e"].index
    df_k2p.drop(delete, inplace = True)
    break
ep = []
ep.append((len(df_k0e)+len(df_k1e))/(len(df_k0)+len(df_k1)))#let k2 be testdata
ep.append((len(df_k1e)+len(df_k2e))/(len(df_k1)+len(df_k2)))#k0
ep.append((len(df_k2e)+len(df_k0e))/(len(df_k0)+len(df_k2)))#k1
ep.append((len(df_k0p)+len(df_k1p))/(len(df_k0)+len(df_k1)))#let k2 be testdata
ep.append((len(df_k1p)+len(df_k2p))/(len(df_k1)+len(df_k2)))#k0
ep.append((len(df_k2p)+len(df_k0p))/(len(df_k0)+len(df_k2)))#k1

#print(ep)
df_k01e = pd.concat([df_k0e, df_k1e], axis = 0) #concat k0, k1 e
df_k01p = pd.concat([df_k0p, df_k1p], axis = 0) #concat k0, k1 p
df_k12e = pd.concat([df_k1e, df_k2e], axis = 0) #concat k2, k1 e
df_k12p = pd.concat([df_k1p, df_k2p], axis = 0) #concat k2, k1 p
df_k02e = pd.concat([df_k0e, df_k2e], axis = 0) #concat k0, k2 e
df_k02p = pd.concat([df_k0p, df_k2p], axis = 0) #concat k0, k2 p
pro_01e = pd.DataFrame()
pro_01p = pd.DataFrame()
pro_12e = pd.DataFrame()
pro_12p = pd.DataFrame()
pro_02e = pd.DataFrame()
pro_02p = pd.DataFrame()
#print(df_k01e)
for col in df_k01e.columns:
    pro_01e = pro_01e.append(df_k01e[col].value_counts())
for col in df_k01p.columns:
    pro_01p = pro_01p.append(df_k01p[col].value_counts())
for col in df_k12e.columns:
    pro_12e = pro_12e.append(df_k12e[col].value_counts())
for col in df_k12p.columns:
    pro_12p = pro_12p.append(df_k12p[col].value_counts())
for col in df_k02e.columns:
    pro_02e = pro_02e.append(df_k02e[col].value_counts())
for col in df_k01p.columns:
    pro_02p = pro_02p.append(df_k02p[col].value_counts())
pro_01e = pro_01e.fillna(0)
pro_01p = pro_01p.fillna(0)
pro_12e = pro_12e.fillna(0)
pro_12p = pro_12p.fillna(0)
pro_02e = pro_02e.fillna(0)
pro_02p = pro_02p.fillna(0)
pro_01e = (pro_01e + 3)/(len(df_k01e) + 3*22)   #laplace smoothing
pro_01p = (pro_01p + 3)/(len(df_k01p) + 3*22)
pro_12e = (pro_12e + 3)/(len(df_k12e) + 3*22)
pro_12p = (pro_12p + 3)/(len(df_k12p) + 3*22)
pro_02e = (pro_02e + 3)/(len(df_k02e) + 3*22)
pro_02p = (pro_02p + 3)/(len(df_k02p) + 3*22)
#print(pro_01e, pro_01p)
'''
print(pro_01e.head())
print(pro_01p.head())
print(pro_12e.head())
print(pro_12p.head())
print(pro_02e.head())
print(pro_02p.head())'''
tp = 0
tn = 0
fp = 0
fn = 0
pro_12e.index = range(0, len(pro_12e))
pro_12p.index = range(0, len(pro_12p))
pro_01e.index = range(0, len(pro_01e))
pro_01p.index = range(0, len(pro_01p))
pro_02e.index = range(0, len(pro_02e))
pro_02p.index = range(0, len(pro_02p))
i = 0
for y in range(0, len(df_k0)):      #let k0 be testdata first
    f1 = 0
    f2 = 0
    for k in range(1, 23):
        f1 = f1 + math.log(pro_12e.at[k, df_k0.iat[y, k]])
        f2 = f2 + math.log(pro_12p.at[k, df_k0.iat[y, k]])
    f1 = f1 + math.log(ep[1]) #'e'
    f2 = f2 + math.log(ep[4]) #'p'
    if(f1 >= f2):
        predict = 'e'
        if(predict == df_k0.iat[i, 0]):
            tp += 1
        else:
            fp += 1
    else:
        predict = 'p'
        if(predict == df_k0.iat[i, 0]):
            tn += 1
        else:
            fn += 1
    i += 1
outcome_k0 = pd.DataFrame(index = ['Actual Positive(etible)', 'Actual Negative(poison)'], columns = ['Predict Positive(etible)', 'Predict negative(poison)'])
outcome_k0['Predict Positive(etible)'] = [tp, fp]
outcome_k0['Predict negative(poison)'] = [fn, tn]
#print('')
#print("Confusion matrix with k0 be testdata-----------------------------------")
#print(outcome_k0)
acc_k0 = (tp+tn) / (tp+tn+fp+fn)
rec_k0 = tp/(tp+fn)
pre_k0 = tp/(tp+fp)
#print("Accuracy:", acc_k0)
#print("Sensitivity(Recall):", rec_k0)
#print("Precision:", pre_k0)
tp1 = 0
tn1 = 0
fp1 = 0
fn1 = 0
i = 0
for y in range(0, len(df_k1)):      #let k1 be testdata
    f1 = 0
    f2 = 0
    for k in range(1, 23):
        f1 = f1 + math.log(pro_02e.at[k, df_k1.iat[y, k]])
        f2 = f2 + math.log(pro_02p.at[k, df_k1.iat[y, k]])
    f1 = f1 + math.log(ep[2]) #'e'
    f2 = f2 + math.log(ep[5]) #'p'
    if(f1 >= f2):
        predict = 'e'
        if(predict == df_k1.iat[i, 0]):
            tp1 += 1
        else:
            fp1 += 1
    else:
        predict = 'p'
        if(predict == df_k1.iat[i, 0]):
            tn1 += 1
        else:
            fn1 += 1
    i += 1
outcome_k1 = pd.DataFrame(index = ['Actual Positive(etible)', 'Actual Negative(poison)'], columns = ['Predict Positive(etible)', 'Predict negative(poison)'])
outcome_k1['Predict Positive(etible)'] = [tp1, fp1]
outcome_k1['Predict negative(poison)'] = [fn1, tn1]
#print('')
#print("Confusion matrix with k1 be testdata-----------------------------------")
#print(outcome_k1)
acc_k1 = (tp1+tn1) / (tp1+tn1+fp1+fn1)
rec_k1 = tp1/(tp1+fn1)
pre_k1 = tp1/(tp1+fp1)
#print("Accuracy:", acc_k1)
#print("Sensitivity(Recall):", rec_k1)
#print("Precision:", pre_k1)
tp2 = 0
tn2 = 0
fp2 = 0
fn2 = 0
i = 0
for y in range(0, len(df_k2)):      #let k2 be testdata
    f1 = 0
    f2 = 0
    for k in range(1, 23):
        f1 = f1 + math.log(pro_01e.at[k, df_k2.iat[y, k]])
        f2 = f2 + math.log(pro_01p.at[k, df_k2.iat[y, k]])
    f1 = f1 + math.log(ep[0]) #'e'
    f2 = f2 + math.log(ep[3]) #'p'
    if(f1 >= f2):
        predict = 'e'
        if(predict == df_k2.iat[i, 0]):
            tp2 += 1
        else:
            fp2 += 1
    else:
        predict = 'p'
        if(predict == df_k2.iat[i, 0]):
            tn2 += 1
        else:
            fn2 += 1
    i += 1
outcome_k2 = pd.DataFrame(index = ['Actual Positive(etible)', 'Actual Negative(poison)'], columns = ['Predict Positive(etible)', 'Predict negative(poison)'])
outcome_k2['Predict Positive(etible)'] = [tp2, fp2]
outcome_k2['Predict negative(poison)'] = [fn2, tn2]
#print('')
#print("Confusion matrix with k2 be testdata-----------------------------------")
#print(outcome_k2)
acc_k2 = (tp2+tn2) / (tp2+tn2+fp2+fn2)
rec_k2 = tp2/(tp2+fn2)
pre_k2 = tp2/(tp2+fp2)
#print("Accuracy:", acc_k2)
#print("Sensitivity(Recall):", rec_k2)
#print("Precision:", pre_k2)
print('')
print("Average Confusion matrix with K-fold------------------------")
outcome_avg = pd.DataFrame(index = ['Actual Positive(etible)', 'Actual Negative(poison)'], columns = ['Predict Positive(etible)', 'Predict negative(poison)'])
outcome_avg['Predict Positive(etible)'] = [(tp+tp1+tp2)/3, (fp+fp1+fp2)/3]
outcome_avg['Predict negative(poison)'] = [(fn+fn1+fn2)/3, (tn+tn1+tn2)/3]
print(outcome_avg)
av_acc = np.mean([acc_k0, acc_k1, acc_k2])
av_rec = np.mean([rec_k0, rec_k1, rec_k2])
av_pre = np.mean([pre_k0, pre_k1, pre_k2])
print("Average Accuracy:", av_acc)
print("Average Sensitivity(Recall):", av_rec)
print("Average Precision:", av_pre)
