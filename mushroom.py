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
arr_e = []
arr_p = []
for i in range (0,24):
    arr_e.append([])
    arr_p.append([])
    arr_e[i].append(0)
    arr_p[i].append(0)
#print(arr[14][0])
cha = []
cha2 = []
type_num_e = []   #means number of types in every feature
type_num_p = []
type_num_e.append(0)
type_num_p.append(0)
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

print(len(df), len(df_e), len(df_p))
i = 0
for col in df_e.columns:
    i = i + 1
    val_fre = pd.DataFrame()
    #val_fre = val_fre.append(df[col].value_counts())
    val_fre = val_fre.append(df_e[col].value_counts(normalize = True))
    #print(val_fre)
    #cha.append([])
    cha.append(val_fre.columns.values.tolist()) 
    #print(val_fre.columns.values.tolist())
    j = 0
    #print(val_fre)
    jk = val_fre.shape[1]
    type_num_e.append(jk)
    for j in range(0, jk):
            #print(val_fre.values)
         arr_e[i].append(val_fre.ix[[0]].values[0][j])    #probility
    #val_fre.drop(val_fre.index[:1])
i = 0
for col in df_p.columns:
    i = i + 1
    val_fre = pd.DataFrame()
    #val_fre = val_fre.append(df[col].value_counts())
    val_fre = val_fre.append(df_p[col].value_counts(normalize = True))
    cha2.append(val_fre.columns.values.tolist()) 
    #print(val_fre.columns.values.tolist())
    j = 0
    #print(val_fre)
    jk = val_fre.shape[1]
    type_num_p.append(jk)
    for j in range(0, jk):
            #print(val_fre.values)
         arr_p[i].append(val_fre.ix[[0]].values[0][j])    #probility in condi p
#print(type_num_p)
i = 0
cha_test = []
for col in df_test.columns:
    i = i + 1
    val_fre = pd.DataFrame()
    val_fre = val_fre.append(df_test[col].value_counts(normalize = True))
    cha_test.append(val_fre.columns.values.tolist())
print(cha_test) 
ct = 1
'''
print(arr_e)
print(arr_p)
print(cha)
print(cha2)
print(type_num_e)
print(type_num_p)'''
type_num_test = [0, 2, 6, 4, 8, 2, 7, 2, 2, 2, 9, 2, 4, 4, 4, 7, 7, 1, 2, 3, 4, 6, 6, 6]
#type_num_test= [0, 2, 5, 4, 8, 2, 5, 2, 2, 2, 9, 2, 4, 4, 4, 6, 6, 1, 2, 3, 4, 5, 5, 6]
for cl in df_e.columns:
    for nums in range(1, type_num_test[ct]+1):
        #print(nums)
        df_e[cl].replace(cha_test[ct-1][nums-1], nums, inplace = True)     #transform
    ct = ct +1                                                  #to int struct
    #break
ct = 1
for cl in df_p.columns:
    for nums in range(1, type_num_test[ct]+1):
        #print(nums)
        df_p[cl].replace(cha_test[ct-1][nums-1], nums, inplace = True) #transform
    ct = ct +1                                          #to int struct
'''
type_num_test = []
#cha_test = []
for j in range(0, 24):
    if type_num_e[j] > type_num_p[j]:
        type_num_test.append(type_num_e[j])
    else:
        type_num_test.append(type_num_p[j])
type_num_test[1] = 2
print(type_num_test)
print(cha)
print(cha2)
print(cha_test)'''

ct = 1
for cl in df_test.columns:
    for nums in range(1, type_num_test[ct]+1):
        #print(nums)
        df_test[cl].replace(cha_test[ct-1][nums-1], nums, inplace = True) #transform
    ct = ct +1 
print(df_e.tail())
print(df_p.tail())
print(df_test.tail())

with open('outputtest.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df_test.to_csv('outputtest.data')
with open('output.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df_e.to_csv('output.data')
with open('output2.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df_p.to_csv('output2.data')

ft = 1
for y in range (0, count-tmp2):
   for ik in range(1, 23):
        #print(df_test.iat[y, ik])
        ft = ft*arr_e[ik+1][df_test.iat[y, ik]]

                #ft = ft*arr[ik][df.iat[y, ik]]
    
#print(arr)
#print(ft)
'''for k in range(1, 22):
    for l in range(0,3):
        print(arr[k][l])
with open('output.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df.to_csv('output.data')'''
#print(count)
