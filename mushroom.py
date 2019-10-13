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
#print(df.tail())
#ct = df.groupby(["eaten"]).size()
val_fre = pd.DataFrame()
#mat = np.zeros((23, 1), dtype = np.int)    #probility matrix
arr = []
for i in range (0,24):
    arr.append([])
    arr[i].append(0)
#print(arr[14][0])
cha = []
i = 0
type_num = []   #means number of types in every feature
type_num.append(0)
df = df.sample(frac=1).reset_index(drop = True)   #shuffle
with open('output.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df.to_csv('output.data')

count = len(df)
tmp = count*0.7
tmp2 = int(tmp)
df_test = df[tmp2:count]
df = df[0:tmp2]

#print(len(df))
'''with open('output2.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df.to_csv('output2.data')
with open('outputtest.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df_test.to_csv('outputtest.data')
'''
fiter = df[df['eaten'] == 'e']
print(df[0:1])
#df.loc[df["eaten"]=='e'].head()
print(df.head())
df_e = pd.DataFrame()
for inde in range(0, tmp2):
    if(df.ix[inde,0] == 'e'):   
        df_e.append(df.ix[inde])
print(df.head())
print(df_e.head())
#for u in range(0, tmp2):
    #if(fiter[u] == True):
     #   print('1')
        #print(df.irow(u))
#print(fg)
for col in df.columns:
    i = i + 1
    val_fre = pd.DataFrame()
    #val_fre = val_fre.append(df[col].value_counts())
    val_fre = val_fre.append(df[col].value_counts(normalize = True))
    #print(val_fre)
    #cha.append([])
    cha.append(val_fre.columns.values.tolist()) 
    #print(val_fre.columns.values.tolist())
    j = 0
    #print(val_fre)
    jk = val_fre.shape[1]
    type_num.append(jk)
    for j in range(0, jk):
            #print(val_fre.values)
         arr[i].append(val_fre.ix[[0]].values[0][j])    #probility
    #val_fre.drop(val_fre.index[:1])
ct = 1
#print(arr)
#print(cha)
for cl in df.columns:
    for nums in range(1, type_num[ct]+1):
        #print(nums)
        df[cl].replace(cha[ct-1][nums-1], nums, inplace = True)     #transform
    ct = ct +1                                                  #to int struct
    #break
ct = 1
for cl in df_test.columns:
    for nums in range(1, type_num[ct]+1):
        #print(nums)
        df_test[cl].replace(cha[ct-1][nums-1], nums, inplace = True) #transform
    ct = ct +1                                                  #to int struct

#print(df.tail())
#print(df_test.tail())

#print(type_num)
#df.info()
#df_test.info()
with open('outputtest.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df_test.to_csv('outputtest.data')

ft = 1
for y in range (0, count-tmp2):
    
    for ik in range(1, 23):
        print(df_test.iat[y, ik])
        ft = ft*arr[ik+1][df_test.iat[y, ik]]
    break
                #ft = ft*arr[ik][df.iat[y, ik]]
    
#print(arr)
print(ft)
'''for k in range(1, 22):
    for l in range(0,3):
        print(arr[k][l])
with open('output.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df.to_csv('output.data')'''
#print(count)
