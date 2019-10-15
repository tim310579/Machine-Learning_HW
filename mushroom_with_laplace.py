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
df = df.sample(frac=1).reset_index(drop = True)   #shuffle
count = len(df)
tmp = count*0.7
tmp2 = int(tmp)
df_test = df[tmp2:count]
df = df[0:tmp2]

df_e = pd.DataFrame()
df_e = df.copy()
df_p = df.copy()
#print(df[df['eaten']=='p'])
#df.drop(dell, inplace = True)
for col in df_e.columns:
    delete = df_e[df_e[col] == "p"].index
    #print(delete)
    df_e.drop(delete, inplace = True)
    break
for col in df_p.columns:
    delete = df_p[df_p[col] == "e"].index
    df_p.drop(delete, inplace = True)
    break
ppp = pd.DataFrame()
epep = []
for col in df.columns:
    ppp = ppp.append(df[col].value_counts(normalize = True))
    break
epep.append((ppp.iat[0,0]))
epep.append((ppp.iat[0,1]))
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

tmp_test.index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
tmp_test0.index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
tmp_test = tmp_test.fillna(0)  #poison
tmp_test0 = tmp_test0.fillna(0) #etible
cnt = 0
cnt2 = 0
i = 0
TP = 0
TN = 0
FP = 0
FN = 0
for y in range (0, count-tmp2):
    ft1 = 1
    ft2 = 1
    for ik in range(1, 23):
        #print(df_test.iat[y, ik])
        ft1 = ft1*tmp_test.at[ik, df_test.iat[y, ik]]
        ft2 = ft2*tmp_test0.at[ik, df_test.iat[y, ik]]
    #$if(df_test.iat[y,0] == 'e'):
    ft2 = ft2/epep[0]   #eat
    
    ft1 = ft1/epep[1]    #poisin
      
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
print(outcome)
acc = (TP+TN) / (TP+TN+FP+FN)
rec = TP/(TP+FN)
pre = TP/(TP+FP)
print("Accuracy:", acc)
print("Sensitivity(Recall):", rec)
print("Precision:", pre)
