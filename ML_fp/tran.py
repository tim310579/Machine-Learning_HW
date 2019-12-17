import warnings
import numpy as np
import csv
import pandas as pd
import re
import math
import random
import codecs

max_depth = 7
def cal_gain(col, category):
    df_H = pd.DataFrame()
    df_H = df_H.append(category.value_counts(normalize = True))
    H = 0
    H += -df_H.iat[0, 0]*math.log(df_H.iat[0, 0], 2) - df_H.iat[0, 1]*math.log(df_H.iat[0, 1], 2)

    if col.dtypes == np.int64:
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
        
        arr1_1 = arr1_1.fillna(1)
        arr1_2 = arr1_2.fillna(1)
        arr2_1 = arr2_1.fillna(1)
        arr2_2 = arr2_2.fillna(1)
        arr3_1 = arr3_1.fillna(1)
        arr3_2 = arr3_2.fillna(1)
    
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


def create_tree(data, label, target, height):
    height += 1
    #print(height)
    if height > max_depth:
        rett = data['transportation'].mode()[0]
        return rett
    ret_all_p_n = pd.DataFrame()
    ret_all_p_n = ret_all_p_n.append(data['transportation'].value_counts())
    for col in range(ret_all_p_n.shape[1]):
        if ret_all_p_n.iat[0, col] == len(data):
            data = data.reset_index()
            le = data.shape[1]
            return (data.iat[0, le-1])
    else:
        tmp = []
        Gain = []
        for col in label:
            G = cal_gain(data[col], data['transportation'])
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
        if(Gain[i][1] == 0):
            #print("8787")
            rett = data['transportation'].mode()[0]  #deal with Gain == 0, but category has different value
            return rett
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
            mytree[target][col] = create_tree(branch, label, label_target, height)
            #return branch
        return mytree

def classify(tree, featlabel, testdata):
    testdata.index = ['0']
    root = list(tree.keys())[0]
    root_feat = root.split('==')
    second = tree[root]
    p = featlabel.index(root_feat[0])
    key = testdata.iat[0, p]
    if(len(root_feat) == 2):
        #print(key)
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
            classlabel = 'train bus or ship'
            #print(root, key)
            #print('cannot predict this one since the node has no feature of it, so replace it with zero')
            return classlabel
        if isinstance(valueOFfeat, dict):
            classlabel = classify(valueOFfeat, featlabel, testdata)
            #print(valueOFfeat)
        else:
            classlabel = valueOFfeat
    
    return classlabel

def predict_res(df_train, feature, df_test):
    tree_index = 0
    tree = pd.DataFrame()
    for i in range(len(df_train)):
    	tree_index = random.randint(0, len(df_train))
    	tree = tree.append(df_train[tree_index: tree_index+1])
    tree = tree.reset_index(drop = True)
    f1 = list(feature)
    for i in range(3):
        delete = random.randint(0, len(f1)-1)
        tree = tree.drop(columns = [f1[delete]])
        del f1[delete]
    target = []
    height = 0
    mytree = (create_tree(tree, f1, target, height))
    print(mytree)
    df_test1 = df_test.copy()
    ret = list(set(feature).difference(set(f1)))
    df_test1 = df_test1.drop(columns = ret)
    predict = []
    cat = df_test.shape[1]-1
    for i in range(len(df_test1)):
        pre = classify(mytree, feature, df_test[i:i+1])
        if(pre == 1):
            predict.append(1)
        elif(pre == 2):
            predict.append(2)
        else:
            predict.append(3)

    return predict

df = pd.read_csv('final.csv', engine = 'python')
df.replace(np.nan, 23, inplace=True)
df['transportation'].replace(['train bus or ship'], 1, inplace = True)
df['transportation'].replace(['HSR or airplane'], 2, inplace = True)
df['transportation'].replace(['drive or ride by yourself'], 3, inplace = True)

df['age'] = df['age'].astype('int64')



feature = ['age', 'job', 'back home frequency', 'location of work (school)', 'hometown location', 'distance between the above two', 'interpersonal relationship', 'family relationship', 'gender', 'financial situation(income)', 'have boy/girlfriend/husband/wife']
f2 = ['job', 'back home frequency', 'location of work (school)', 'hometown location', 'distance between the above two', 'interpersonal relationship', 'family relationship', 'gender', 'financial situation(income)', 'have boy/girlfriend/husband/wife']
for col in f2:
    df[col].replace(["?"],  [df[col].mode()], inplace = True)   #deal with missing

df = df.sample(frac=1).reset_index(drop = True) #shuffle
#df = df[0:100]
choose1 = df[df['transportation'] == 1].index
choose2 = df[df['transportation'] == 2].index
choose3 = df[df['transportation'] == 3].index

df_train1 = df.loc[choose1]
df_train2 = df.loc[choose2]
df_train3 = df.loc[choose3]

split1 = len(df_train1)
split2 = len(df_train2)
split3 = len(df_train3)
df_test1 = df_train1[int(split1*0.7): split1]
df_test2 = df_train2[int(split2*0.7): split2]
df_test3 = df_train3[int(split3*0.7): split3]
df_train1 = df_train1[0: int(split1*0.7)]
df_train2 = df_train2[0: int(split2*0.7)]
df_train3 = df_train3[0: int(split3*0.7)]

df_train = pd.concat([df_train1, df_train2, df_train3], axis = 0)
df_test = pd.concat([df_test1, df_test2, df_test3], axis = 0)
df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

predict_all = []
tree_num = 1
for i in range(tree_num):
    predict = predict_res(df_train, feature, df_test)
    predict_all.append(predict)

#print(predict_all)
new_pd = pd.DataFrame(predict_all)
#print(new_pd)
#print(df.dtypes)
final = []
for i in range(len(df_test)):
    final.append(new_pd[i].mode()[0])
mat = np.zeros([3, 3])
cat = df_test.shape[1]-1
for i in range(len(df_test)):
    if(final[i] == 1):
        if(df_test.iat[i, cat] == 1):
            mat[0][0] += 1  #bingo
        elif(df_test.iat[i, cat] == 2):
            mat[1][0] += 1  #wrong
        else:
            mat[2][0] += 1  #wrong
    elif(final[i] == 2):
        if(df_test.iat[i, cat] == 1):
            mat[0][1] += 1  #wrong
        elif(df_test.iat[i, cat] == 2):
            mat[1][1] += 1  #bingo
        else:
            mat[2][1] += 1  #wrong
    else:
        if(df_test.iat[i, cat] == 1):
            mat[0][2] += 1  #wrong
        elif(df_test.iat[i, cat] == 2):
            mat[1][2] += 1  #wrong
        else:
            mat[2][2] += 1  #bingo
                 
data_frame = pd.DataFrame(mat, index = ['Actual train, bus, ship', 'Actual HSR, airplane', 'Actual drive, ride'], columns = ['Predict train, bus, ship', 'HSR, airplane', 'drive, ride'])
print(data_frame)
acc = (mat[0][0] + mat[1][1] + mat[2][2])/len(df_test)
rec = []
pre = []
for i in range(3):
    rec.append(mat[i][i]/(mat[i][0] + mat[i][1] + mat[i][2]))
    pre.append(mat[i][i]/(mat[0][i] + mat[1][i] + mat[2][i]))
rec_pre = np.array([rec, pre])
#print(rec_pre)
rp = pd.DataFrame(rec_pre, index = ['Sensitivity(Recall)', 'Precision'], columns = ['train, bus, ship', 'HSR, airplane', 'drive, ride'])
print('Accuracy: ', acc)
print(rp)
