import numpy as np
import csv
import pandas as pd
import re
import math

def cal_entropy(col, category):
    if col.dtypes == int:
        #merge = pd.DataFrame()
        #merge = pd.concat([col, category], axis = 1)
        #merge.columns = ['col', 'Category']
        #return merge
        tmp1 = pd.DataFrame()
        tmp2 = pd.DataFrame()
        tmp3 = pd.DataFrame()
        tmp1 = col.copy()
        tmp2 = col.copy()
        tmp3 = col.copy()
        
        fir = col.quantile(.25)
        mid = col.median(axis = 0)
        thi = col.quantile(.75)
        tmp1[tmp1 < fir] = 0    #use 1/4, 2/4, 3/4 to let it be discrete
        tmp1[tmp1 > fir] = 1
        tmp2[tmp2 < mid] = 0
        tmp2[tmp2 > mid] = 1
        tmp3[tmp3 < thi] = 0
        tmp3[tmp3 > thi] = 1
        merge1 = pd.concat([tmp1, category], axis = 1)
        merge2 = pd.concat([tmp2, category], axis = 1)
        merge3 = pd.concat([tmp3, category], axis = 1)
        
        merge1.columns = ['col', 'cate']
        merge2.columns = ['col', 'cate']
        merge3.columns = ['col', 'cate']
        a = merge1.median()
        #return merge1
        arr1 = pd.DataFrame()
        arr1 = arr1.append(tmp1.value_counts(normalize = True))
        arr2 = pd.DataFrame()
        arr2 = arr2.append(tmp2.value_counts(normalize = True))
        arr3 = pd.DataFrame()
        arr3 = arr3.append(tmp3.value_counts(normalize = True))
    
        H1 = -arr1.ix[0, 0]*math.log(arr1.ix[0, 0], 2) - arr1.ix[0, 1]*math.log(arr1.ix[0, 1], 2) # H(T, D)
        H2 = -arr2.ix[0, 0]*math.log(arr2.ix[0, 0], 2) - arr2.ix[0, 1]*math.log(arr2.ix[0, 1], 2)
        H3 = -arr3.ix[0, 0]*math.log(arr3.ix[0, 0], 2) - arr3.ix[0, 1]*math.log(arr3.ix[0, 1], 2)
        #return H1, H2, H3
        part1 = pd.DataFrame()
        part2 = pd.DataFrame()
        part3 = pd.DataFrame() #probability of occurence

        part1 = part1.append(merge1['col'].value_counts(normalize = True))
        part2 = part2.append(merge2['col'].value_counts(normalize = True))
        part3 = part3.append(merge3['col'].value_counts(normalize = True)) 
        #return part1, part2, part3
        merge1_2 = merge1.copy()
        merge2_2 = merge2.copy()
        merge3_2 = merge3.copy()
        delete1 = merge1[merge1['col'] == 1].index
        delete1_2 = merge1[merge1['col'] == 0].index
        delete2 = merge2[merge2['col'] == 1].index
        delete2_2 = merge2[merge2['col'] == 0].index
        delete3 = merge3[merge3['col'] == 1].index
        delete3_2 = merge3[merge3['col'] == 0].index

        merge1.drop(delete1, inplace = True)
        merge1_2.drop(delete1_2, inplace = True)
        merge2.drop(delete2, inplace = True)
        merge2_2.drop(delete2_2, inplace = True)
        merge3.drop(delete3, inplace = True)
        merge3_2.drop(delete3_2, inplace = True)
        #return merge1, merge1_2
        arr1_1 = pd.DataFrame()
        arr1_1 = arr1_1.append(merge1['cate'].value_counts(normalize = True))
        arr1_2 = pd.DataFrame()
        arr1_2 = arr1_2.append(merge1_2['cate'].value_counts(normalize = True))
        arr2_1 = pd.DataFrame()
        arr2_1 = arr2_1.append(merge2['cate'].value_counts(normalize = True))
        arr2_2 = pd.DataFrame()
        arr2_2 = arr2_2.append(merge2_2['cate'].value_counts(normalize = True))
        arr3_1 = pd.DataFrame()
        arr3_1 = arr3_1.append(merge3['cate'].value_counts(normalize = True))
        arr3_2 = pd.DataFrame()
        arr3_2 = arr3_2.append(merge3_2['cate'].value_counts(normalize = True))
        
        R1 = (-arr1_1.ix[0, 0]*math.log(arr1_1.ix[0, 0], 2) - arr1_1.ix[0, 1]*math.log(arr1_1.ix[0, 1], 2))*part1.ix[0, 0] + (-arr1_2.ix[0, 0]*math.log(arr1_2.ix[0, 0], 2) - arr1_2.ix[0, 1]*math.log(arr1_2.ix[0, 1], 2))*part1.ix[0, 1]
        R2 = (-arr2_1.ix[0, 0]*math.log(arr2_1.ix[0, 0], 2) - arr2_1.ix[0, 1]*math.log(arr2_1.ix[0, 1], 2))*part2.ix[0, 0] + (-arr2_2.ix[0, 0]*math.log(arr2_2.ix[0, 0], 2) - arr2_2.ix[0, 1]*math.log(arr2_2.ix[0, 1], 2))*part2.ix[0, 1]
        R3 = (-arr3_1.ix[0, 0]*math.log(arr3_1.ix[0, 0], 2) - arr3_1.ix[0, 1]*math.log(arr3_1.ix[0, 1], 2))*part3.ix[0, 0] + (-arr3_2.ix[0, 0]*math.log(arr3_2.ix[0, 0], 2) - arr3_2.ix[0, 1]*math.log(arr3_2.ix[0, 1], 2))*part3.ix[0, 1]

        G1 = H1 - R1
        G2 = H2 - R2
        G3 = H3 - R3
        return G1, G2, G3
        #return arr
    else:
        print('obb')
'''
    tmp = pd.DataFrame()
    tmp = tmp.append(col.value_counts(normalize = True))
    return tmp
'''
dfx = pd.read_csv('X_train.csv')
dfy = pd.read_csv('y_train.csv')
#print(len(dfx), len(dfy))
c = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
dfa = pd.DataFrame()

#print(dfx['workclass']=="?")


for col in c:
    dfx[col].replace([" ?"], [dfx[col].mode()], inplace = True)     #replace the missing vlaue with mode
dfy = dfy.drop(columns = ['Id'])    
df_train = pd.concat([dfx, dfy], axis = 1)  #merge x and y
#print(df_train.tail())
#print(dfy)

print(cal_entropy(df_train['age'], df_train['Category']))

