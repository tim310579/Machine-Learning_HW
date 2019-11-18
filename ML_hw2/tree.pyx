import warnings
import numpy as np
import csv
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import cython

#warnings.filterwarnings('ignore')

cdef float H, R1, R2, R3, G1, G2, G3, maxx, fir, mid, thi, H_part
cpdef cal_gain(col, category):
    #base_H = 0
    df_H = pd.DataFrame()
    df_H = df_H.append(category.value_counts(normalize = True))
    H = 0
    H += -df_H.iat[0, 0]*math.log(df_H.iat[0, 0], 2) - df_H.iat[0, 1]*math.log(df_H.iat[0, 1], 2)
    #return H
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
        tmp1[tmp1 <= fir] = 0    #use 1/4, 2/4, 3/4 to let it be discrete
        tmp1[tmp1 > fir] = 1
        tmp2[tmp2 <= mid] = 0
        tmp2[tmp2 > mid] = 1
        tmp3[tmp3 <= thi] = 0
        tmp3[tmp3 > thi] = 1
        merge1 = pd.concat([tmp1, category], axis = 1)
        merge2 = pd.concat([tmp2, category], axis = 1)
        merge3 = pd.concat([tmp3, category], axis = 1)
        
        merge1.columns = ['col', 'cate']
        merge2.columns = ['col', 'cate']
        merge3.columns = ['col', 'cate']

        part1 = pd.DataFrame()
        part2 = pd.DataFrame()
        part3 = pd.DataFrame() #probability of occurence

        part1 = part1.append(merge1['col'].value_counts(normalize = True))
        part2 = part2.append(merge2['col'].value_counts(normalize = True))
        part3 = part3.append(merge3['col'].value_counts(normalize = True)) 
        #print(part1, part2, part3)
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
        '''
        print(merge1)
        print(merge1_2)
        print(arr1_1, 'lalalala')
        print(arr1_2)
        print(arr2_1)
        print(arr2_2)
        print(arr3_1)
        print(arr3_2)'''
        
        arr1_1 = arr1_1.fillna(1)
        arr1_2 = arr1_2.fillna(1)
        arr2_1 = arr2_1.fillna(1)
        arr2_2 = arr2_2.fillna(1)
        arr3_1 = arr3_1.fillna(1)
        arr3_2 = arr3_2.fillna(1)
        '''
        print(part1)
        print(part2)
        print(part3)
        print(col.tail())
        
        print(arr1_1, 'lalalala')
        print(arr1_2)
        print(arr2_1)
        print(arr2_2)
        print(arr3_1)
        print(arr3_2)
        '''
        #return
        if(arr1_1.empty):
            R1 = 0
        else:
            if(arr1_1.shape[1] == 1):
                R1 = 0
            else:
                R1 = (-arr1_1.iat[0, 0]*math.log(arr1_1.iat[0, 0], 2) - arr1_1.iat[0, 1]*math.log(arr1_1.iat[0, 1], 2))*part1.iat[0, 0]
        if(arr1_2.empty):
            R1 += 0
        else:
            if(arr1_2.shape[1] == 1):
                R1 += 0
            else:
                R1 += (-arr1_2.iat[0, 0]*math.log(arr1_2.iat[0, 0], 2) - arr1_2.iat[0, 1]*math.log(arr1_2.iat[0, 1], 2))*part1.iat[0, 1]
        if(arr2_1.empty):
            R2 = 0
        else:
            if(arr2_1.shape[1] == 1):
                R2 = 0
            else:
                R2 = (-arr2_1.iat[0, 0]*math.log(arr2_1.iat[0, 0], 2) - arr2_1.iat[0, 1]*math.log(arr2_1.iat[0, 1], 2))*part2.iat[0, 0]
        if(arr2_2.empty):
            R2 += 0
        else:
            if(arr2_2.shape[1] == 1):
                R2 += 0
            else:
                R2 += (-arr2_2.iat[0, 0]*math.log(arr2_2.iat[0, 0], 2) - arr2_2.iat[0, 1]*math.log(arr2_2.iat[0, 1], 2))*part2.iat[0, 1]
        if(arr3_1.empty):
            R3 = 0
        else:
            if(arr3_1.shape[1] == 1):
                R3 = 0
            else:
                R3 = (-arr3_1.iat[0, 0]*math.log(arr3_1.iat[0, 0], 2) - arr3_1.iat[0, 1]*math.log(arr3_1.iat[0, 1], 2))*part3.iat[0, 0]
        if(arr3_2.empty):
            R3 += 0
        else:
            if(arr3_2.shape[1] == 1):
                R3 += 0
            else:
                R3 += (-arr3_2.iat[0, 0]*math.log(arr3_2.iat[0, 0], 2) - arr3_2.iat[0, 1]*math.log(arr3_2.iat[0, 1], 2))*part3.iat[0, 1]

        G1 = H - R1
        G2 = H - R2
        G3 = H - R3
        #return R1, R2, R3, G1, G2, G3
        maxx = max(G1, G2, G3)
        if(maxx == G1):
            return fir, G1
        elif(maxx == G2):
            return mid, G2
        else:
            return thi, G3
        #return arr
    else:
        tmp = pd.DataFrame()
        tmp = tmp.append(col.value_counts(normalize = True))
        merge = pd.concat([col, category], axis = 1)
        merge.columns = ['col', 'cate']
        R = []
        H_part = 0
        df_tmp_entropy = pd.DataFrame()
        df_h = pd.DataFrame()
        #print(merge)
        #merge.to_csv('see.data')
        for col in tmp.columns:
            df_h = df_h.drop(df_h.index, inplace = True)
            df_h = df_tmp_entropy.drop(df_tmp_entropy.index, inplace = True)
            choose = 0
            choose = merge[merge['col'] == col].index
            #print(merge)
            df_h = merge.loc[choose, 'cate']
            #print(df_h)
            df_tmp_entropy = df_tmp_entropy.append(df_h.value_counts(normalize = True))
            #print(df_tmp_entropy)
            
            df_tmp_entropy = df_tmp_entropy.fillna(1)
            #print(df_tmp_entropy, 'lalalala')
            if(df_tmp_entropy.shape[1] > 1):
                H_part = -df_tmp_entropy.iat[0, 0]*math.log(df_tmp_entropy.iat[0, 0], 2) - df_tmp_entropy.iat[0, 1]*math.log(df_tmp_entropy.iat[0, 1], 2)
            else:
                H_part = 0
            R.append(H_part)
           # H_part *= tmp.ix[0, i]
        
        for i in range(tmp.shape[1]):
            R[i] *= tmp.iat[0, i]
        ret_R = 0
        for i in range(len(R)):
            ret_R += R[i]
        G = H - ret_R
        string = 'string' 
        return string, G
       
'''
create tree
'''
cdef float compare_num = 0
cdef int le
cdef list label
cpdef create_tree(data, label, target):
    ret_all_p_n = pd.DataFrame()
    ret_all_p_n = ret_all_p_n.append(data['Category'].value_counts())
    #print(label)
    #data = data.reset_index()
    #print(ret_all_p_n)
    #print(len(data))
    for col in range(ret_all_p_n.shape[1]):
        if(ret_all_p_n.iat[0, col] == len(data)):
            #if(ret_all_p_n.ix[0, 0] == len(data)):
            #print("AAAAA")
            data = data.reset_index()
            le = data.shape[1]
            return (data.iat[0, le-1])
        '''
        if(cate.ix[0, 0] == 0):
            return 0
        else:
            return 1
        '''
    else:
        tmp = []
        Gain = []
        for col in label:
            G = cal_gain(data[col], data['Category'])
            tmp.append(G[1])
            Gain.append(G)
        i = tmp.index(max(tmp))
        data_copy = pd.DataFrame()
        data_copy = data_copy.drop(data_copy.index, inplace = True)
        data_copy = data.copy()
        #print(i, Gain)
        if(Gain[i][0] != 'string'):
            target = label[i]
            compare_num = Gain[i][0]
            data_copy[data_copy[target] <= compare_num] = 0
            data_copy[data_copy[target] > compare_num] = 1
        elif(Gain[i][0] == 'string'):
            target = label[i]
            #del(label[i])
        label_target = target
        if(Gain[i][0] != 'string'):
            target = target + '==' + str(Gain[i][0])
        #label_target = target
        mytree = {target:{}}
        #print(target)
        sub_tree = pd.DataFrame()
        sub_tree = sub_tree.append(data_copy[label_target].value_counts())
        for col in sub_tree.columns:
            choose = data_copy[data_copy[label_target] == col].index
            #print(choose)
            branch = pd.DataFrame()
            #branch = data_copy[data_copy[target[0]].isin([col])]
            branch = data.loc[choose]
            #print(branch)
            mytree[target][col] = create_tree(branch, label, label_target)
            #return branch
        return mytree
'''
test
'''
cdef dict tree, mytree, second, valueOFfeat
cdef str root, key
cdef int p
cpdef classify(tree, featlabel, testdata):
    testdata.index = ['0']
    root = list(tree.keys())[0]
    #print(type(root))
    root_feat = root.split('==')
    second = tree[root]
    p = featlabel.index(root_feat[0])+1
    #print(p, root_feat, testdata)
    key = testdata.iat[0, p]
    #print(testdata.ix[0, 'Id'], key)
    #valueOFfeat = second[key]
    if(len(root_feat) == 2):
        compare_num = float(root_feat[1])
        if key <= compare_num:
            valueOFfeat = second[0]
        else:
            valueOFfeat = second[1]
        if isinstance(valueOFfeat, dict):
            classlabel = classify(valueOFfeat, featlabel, testdata)
        else:
            classlabel = valueOFfeat
    else:
        try:
            valueOFfeat = second[key]
        except KeyError:
            #print('cannot predict this one since the node has no feature of it, so replace it with zero')
            return 0
        if isinstance(valueOFfeat, dict):
            classlabel = classify(valueOFfeat, featlabel, testdata)
        else:
            classlabel = valueOFfeat
    return classlabel
cdef int predict
cdef int Id
cdef list c, cc
dfx = pd.read_csv('X_train.csv')
dfy = pd.read_csv('y_train.csv')
c = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cc = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
for col in c:
    dfx[col].replace([" ?"], [dfx[col].mode()], inplace = True)     #replace the missing vlaue with mode
dfy = dfy.drop(columns = ['Id'])    


cpdef do_all():
#print(len(dfx), len(dfy))
    
#print(dfx['workclass']=="?")
    dfx_test = pd.read_csv('X_test.csv')
    df_train = pd.concat([dfx, dfy], axis = 1)  #merge x and y
    df_train = df_train.sample(frac=1).reset_index(drop = True) #shuffle
    #df_train = df_train[0:6000]

    choose1 = df_train[df_train['Category'] == 1].index
    df_train1 = df_train.loc[choose1]
    #print(df_train1)
    choose0 = df_train[df_train['Category'] == 0].index
    df_train0 = df_train.loc[choose0]
    split1 = len(df_train1)
    split0 = len(df_train0)
    df_test1 = df_train1[int(split1*0.7): split1]
    df_test0 = df_train0[int(split0*0.7): split0]
    df_train1 = df_train1[0: int(split1*0.7)]
    df_train0 = df_train0[0: int(split0*0.7)]
    df_test = pd.concat([df_test1, df_test0], axis = 0)
    df_train10 = pd.concat([df_train1, df_train0], axis = 0)
    df_test = df_test.reset_index()
    df_train10 = df_train10.reset_index()
    #df_check = df_test['Category']
    df_test = df_test.drop('index', axis = 1)
    
    
    #print(df_test)
#print(df_train.tail())
#print(dfy)
    
    target = []
#for col in cc:
 #   print(cal_gain(df_train[col], df_train['Category']))
    mytree = (create_tree(df_train10, cc, target))
    matrix = pd.DataFrame(index = ['Actual > 50k(1)', 'Actual <= 50k(0)'], columns = ['Predict > 50k(1)', 'Predict <= 50k(0)'])
    #outcome = pd.DataFrame(columns = ['Id', 'Category'])
    #print(outcome.dtypes)
    tp = float(0)
    tn = float(0)
    fp = float(0)
    fn = float(0)
    for i in range(0, len(df_test)):
        predict = classify(mytree, cc, df_test[i:i+1])
        #print(predict, df_test.iat[i, df_test.shape[1]-1])
        if(predict == df_test.iat[i, df_test.shape[1]-1]):
            if(predict == 1):   tp += 1
            else:   tn += 1
        else:
            if(predict == 1):   fp += 1
            else:   fn += 1
    #Id = df_test.iat[i, 0]
    #outcome.loc[i, 'Id'] = Id
    #outcome.loc[i, 'Category'] = predict
    #outcome.columns = ['Id', 'Category']
    #print(outcome.dtypes)
    #outcome['Id'] = outcome['Id'].astype('int')
    #outcome['Category'] = outcome['Category'].astype('int')
    #print(outcome.dtypes)
    #outcome.to_csv('submission.csv', index 
    
    acc = (tp+tn) / (tp+tn+fp+fn)
    rec = tp/(tp+fn)
    pre = tp/(tp+fp)
    matrix['Predict > 50k(1)'] = [tp, fp]
    matrix['Predict <= 50k(0)'] = [fn, tn]
    print('Confusion Matrix--------------------------')
    print(matrix)
    print 'Accuracy:', acc
    print 'Sensitivity(Recall):', rec
    print 'precision:', pre

