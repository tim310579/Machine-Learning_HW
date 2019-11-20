import warnings
import numpy as np
import csv
import pandas as pd
import re
import math
import random

max_depth = 10
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


def create_tree(data, label, target):
    #print(data)
    ret_all_p_n = pd.DataFrame()
    ret_all_p_n = ret_all_p_n.append(data['Category'].value_counts())
    for col in range(ret_all_p_n.shape[1]):
        if ret_all_p_n.iat[0, col] == len(data):
            data = data.reset_index()
            le = data.shape[1]
            return (data.iat[0, le-1])
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
        if(Gain[i][1] == 0):
            #print("8787")
            rett = data['Category'].mode()  #deal with Gain == 0, but category has different value
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
            mytree[target][col] = create_tree(branch, label, label_target)
            #return branch
        return mytree

def classify(tree, featlabel, testdata):
    testdata.index = ['0']
    root = list(tree.keys())[0]
    #print(type(root))
    root_feat = root.split('==')
    second = tree[root]
    p = featlabel.index(root_feat[0])+1
    #print(p)
    #feat = featlabel[p]
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

dfx = pd.read_csv('X_train.csv')
dfy = pd.read_csv('y_train.csv')
c = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cc = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
cn = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for col in c:
    dfx[col].replace([" ?"], [dfx[col].mode()], inplace = True)     #replace the missing vlaue with mode
    
dfy = dfy.drop(columns = ['Id'])
    
def hold_out():
    df_train = pd.concat([dfx, dfy], axis = 1)  #merge x and y
    df_train = df_train.sample(frac=1).reset_index(drop = True) #shuffle
    df_train = df_train[0:300]
    #print(df_train)
    split = len(df_train)
    df_test = df_train[int(split*0.7): split]
    df_test = df_test.reset_index(drop = True)
    df_train = df_train[0: int(split*0.7)]
    tree1_index = 0
    tree1 = pd.DataFrame()
    for i in range(len(df_train)):
    	tree1_index = random.randint(0, len(df_train))
    	tree1 = tree1.append(df_train[tree1_index: tree1_index+1])
    tree1 = tree1.reset_index(drop = True)
    cf1 = list(cc)
    for i in range(5):
        delete = random.randint(0, len(cf1)-1)
        tree1 = tree1.drop(columns = [cf1[delete]])
        del cf1[delete]
    target = []
    mytree1 = create_tree(tree1, cf1, target)
    predict1 = []
    df_test1 = df_test.copy()
    ret = list(set(cc).difference(set(cf1)))
    df_test1 = df_test1.drop(columns = ret)
    
    for i in range(0, len(df_test1)):
        predict1.append(classify(mytree1, cf1, df_test1[i:i+1]))
    #second tree    
    tree2_index = 0
    tree2 = pd.DataFrame()
    for i in range(len(df_train)):
        tree2_index = random.randint(0, len(df_train))
        tree2 = tree2.append(df_train[tree2_index: tree2_index+1])
    tree2 = tree2.reset_index(drop = True)
    cf2 = list(cc)
    for i in range(5):
        delete = random.randint(0, len(cf2)-1)
        tree2 = tree2.drop(columns = [cf2[delete]])
        del cf2[delete]
    target2 = []
    mytree2 = create_tree(tree2, cf2, target2)
    predict2 = []
    df_test2 = df_test.copy()
    ret = list(set(cc).difference(set(cf2)))
    df_test2 = df_test2.drop(columns = ret)

    for i in range(0, len(df_test2)):
        predict2.append(classify(mytree2, cf2, df_test2[i:i+1]))

    tree3_index = 0
    tree3 = pd.DataFrame()
    for i in range(len(df_train)):
        tree3_index = random.randint(0, len(df_train))
        tree3 = tree3.append(df_train[tree3_index: tree3_index+1])
    tree3 = tree3.reset_index(drop = True)
    cf3 = list(cc)
    for i in range(5):
        delete = random.randint(0, len(cf3)-1)
        tree3 = tree3.drop(columns = [cf3[delete]])
        del cf3[delete]
    target3 = []
    mytree3 = create_tree(tree3, cf3, target3)
    predict3 = []
    df_test3 = df_test.copy()
    ret = list(set(cc).difference(set(cf3)))
    df_test3 = df_test3.drop(columns = ret)

    for i in range(0, len(df_test3)):
        predict3.append(classify(mytree3, cf3, df_test3[i:i+1]))
    predict_final = []
    for i in range(0, len(predict3)):
        predict_final.append(predict1[i] + predict2[i] + predict3[i])
    #predict_final = np.array(predict1) + np.array(predict2) + np.array(predict3)
    #print(predict_final)
    tp = float(0)
    tn = float(0)
    fp = float(0)
    fn = float(0)
    cat = df_test.shape[1]-1
    ser = pd.Series(predict_final)
    #ser[ser > 1]
    
    for i in range(0, len(ser)):
        if(ser[i] == 2 | ser[i] == 3):     #means two voted 1
            if(df_test.iat[i, cat] == 1):  tp += 1
            else:   fp += 1
        else:
            if(df_test.iat[i, cat] == 1):  fn += 1 #guess0, actual 1
            else:   tn += 1
    matrix = pd.DataFrame(index = ['Actual > 50k(1)', 'Actual <= 50k(0)'], columns = ['Predict > 50k(1)', 'Predict <= 50k(0)'])
    matrix['Predict > 50k(1)'] = [tp, fp]
    matrix['Predict <= 50k(0)'] = [fn, tn]
    print('Confusion Matrix Holdout--------------------------------------')
    print( matrix)
    acc = (tp+tn) / (tp+tn+fp+fn)
    rec = tp/(tp+fn)
    pre = tp/(tp+fp)
    print ('Accuracy:', acc)
    print ('Sensitivity(Recall):', rec)
    print ('precision:', pre)

def for_k123(df_train, df_test):
    tree1_index = 0
    tree1 = pd.DataFrame()
    for i in range(len(df_train)):
    	tree1_index = random.randint(0, len(df_train))
    	tree1 = tree1.append(df_train[tree1_index: tree1_index+1])
    tree1 = tree1.reset_index(drop = True)
    cf1 = list(cc)
    for i in range(5):
        delete = random.randint(0, len(cf1)-1)
        tree1 = tree1.drop(columns = [cf1[delete]])
        del cf1[delete]
    target = []
    mytree1 = create_tree(tree1, cf1, target)
    predict1 = []
    df_test1 = df_test.copy()
    ret = list(set(cc).difference(set(cf1)))
    df_test1 = df_test1.drop(columns = ret)
    
    for i in range(0, len(df_test1)):
        predict1.append(classify(mytree1, cf1, df_test1[i:i+1]))
    #second tree    
    tree2_index = 0
    tree2 = pd.DataFrame()
    for i in range(len(df_train)):
        tree2_index = random.randint(0, len(df_train))
        tree2 = tree2.append(df_train[tree2_index: tree2_index+1])
    tree2 = tree2.reset_index(drop = True)
    cf2 = list(cc)
    for i in range(5):
        delete = random.randint(0, len(cf2)-1)
        tree2 = tree2.drop(columns = [cf2[delete]])
        del cf2[delete]
    target2 = []
    mytree2 = create_tree(tree2, cf2, target2)
    predict2 = []
    df_test2 = df_test.copy()
    ret = list(set(cc).difference(set(cf2)))
    df_test2 = df_test2.drop(columns = ret)

    for i in range(0, len(df_test2)):
        predict2.append(classify(mytree2, cf2, df_test2[i:i+1]))

    tree3_index = 0
    tree3 = pd.DataFrame()
    for i in range(len(df_train)):
        tree3_index = random.randint(0, len(df_train))
        tree3 = tree3.append(df_train[tree3_index: tree3_index+1])
    tree3 = tree3.reset_index(drop = True)
    cf3 = list(cc)
    for i in range(5):
        delete = random.randint(0, len(cf3)-1)
        tree3 = tree3.drop(columns = [cf3[delete]])
        del cf3[delete]
    target3 = []
    mytree3 = create_tree(tree3, cf3, target3)
    predict3 = []
    df_test3 = df_test.copy()
    ret = list(set(cc).difference(set(cf3)))
    df_test3 = df_test3.drop(columns = ret)

    for i in range(0, len(df_test3)):
        predict3.append(classify(mytree3, cf3, df_test3[i:i+1]))
    predict_final = []
    for i in range(0, len(predict3)):
        predict_final.append(predict1[i] + predict2[i] + predict3[i])
    #predict_final = np.array(predict1) + np.array(predict2) + np.array(predict3)
    
    return predict_final
def K_fold():
    df_traink = pd.concat([dfx, dfy], axis = 1)  #merge x and y
    df_traink = df_traink.sample(frac=1).reset_index(drop = True) #shuffle
    df_traink = df_traink[0:300]
    split = len(df_traink)
    df_traink1 = df_traink[0:int(split/3)]
    df_traink2 = df_traink[int(split/3):int((split*2)/3)]
    df_traink3 = df_traink[int((split*2)/3):split]
    df_traink2 = df_traink2.reset_index(drop = True)
    df_traink3 = df_traink3.reset_index(drop = True)
    traink12 = pd.concat([df_traink1, df_traink2], axis = 0)
    traink23 = pd.concat([df_traink2, df_traink3], axis = 0)
    traink13 = pd.concat([df_traink1, df_traink3], axis = 0)
    traink12 = traink12.reset_index(drop = True)
    traink23 = traink23.reset_index(drop = True)
    traink13 = traink13.reset_index(drop = True)
    
    cat = df_traink3.shape[1]-1
    predictk3 = for_k123(traink12, df_traink3)    #k3 for test first
    tp = float(0)
    tn = float(0)
    fp = float(0)
    fn = float(0)
    #print(predictk3)
    ser = pd.Series(predictk3)
    for i in range(0, len(ser)):
        if(ser[i] == 2 | ser[i] == 3):  #means two or above vote for yes(1)
            if(df_traink3.iat[i, cat] == 1):  tp += 1
            else:   fp += 1
        else:
            if(df_traink3.iat[i, cat] == 1):  fn += 1 #guess0, actual 1
            else:   tn += 1
    matrix1 = pd.DataFrame(index = ['Actual > 50k(1)', 'Actual <= 50k(0)'], columns = ['Predict > 50k(1)', 'Predict <= 50k(0)'])
    matrix1['Predict > 50k(1)'] = [tp, fp]
    matrix1['Predict <= 50k(0)'] = [fn, tn]
    print ('Confusion Matrix---------------------------------------------')
    print (matrix1)
    
    cat = df_traink1.shape[1]-1
    predictk1 = for_k123(traink23, df_traink1)    #k3 for test first
    tp2 = float(0)
    tn2 = float(0)
    fp2 = float(0)
    fn2 = float(0)
    #print(predictk3)
    ser1 = pd.Series(predictk1)
    for i in range(0, len(ser1)):
        if(ser1[i] == 2 | ser[i] == 3):  #means two or above vote for yes(1)
            if(df_traink1.iat[i, cat] == 1):  tp2 += 1
            else:   fp2 += 1
        else:
            if(df_traink1.iat[i, cat] == 1):  fn2 += 1 #guess0, actual 1
            else:   tn2 += 1
    matrix2 = pd.DataFrame(index = ['Actual > 50k(1)', 'Actual <= 50k(0)'], columns = ['Predict > 50k(1)', 'Predict <= 50k(0)'])
    matrix2['Predict > 50k(1)'] = [tp2, fp2]
    matrix2['Predict <= 50k(0)'] = [fn2, tn2]
    print ('Confusion Matrix---------------------------------------------')
    print (matrix2)

    cat = df_traink2.shape[1]-1
    predictk2 = for_k123(traink13, df_traink2)    #k3 for test first
    tp3 = float(0)
    tn3 = float(0)
    fp3 = float(0)
    fn3 = float(0)
    #print(predictk3)
    ser2 = pd.Series(predictk2)
    for i in range(0, len(ser2)):
        if(ser2[i] == 2 | ser2[i] == 3):  #means two or above vote for yes(1)
            if(df_traink2.iat[i, cat] == 1):  tp3 += 1
            else:   fp3 += 1
        else:
            if(df_traink2.iat[i, cat] == 1):  fn3 += 1 #guess0, actual 1
            else:   tn3 += 1
    matrix3 = pd.DataFrame(index = ['Actual > 50k(1)', 'Actual <= 50k(0)'], columns = ['Predict > 50k(1)', 'Predict <= 50k(0)'])
    matrix3['Predict > 50k(1)'] = [tp3, fp3]
    matrix3['Predict <= 50k(0)'] = [fn3, tn3]
    print ('Confusion Matrix---------------------------------------------')
    print (matrix3)

    matrix_avg = pd.DataFrame(index = ['Actual > 50k(1)', 'Actual <= 50k(0)'], columns = ['Predict > 50k(1)', 'Predict <= 50k(0)'])
    matrix_avg['Predict > 50k(1)'] = [np.mean([tp, tp2, tp3]), np.mean([fp, fp2, fp3])]
    matrix_avg['Predict <= 50k(0)'] = [np.mean([fn, fn2, fn3]), np.mean([tn, tn2, tn3])]
    print(matrix_avg)
    acc = (tp+tp2+tp3+tn+tn2+tn3)/(tp+tp2+tp3+fp+fp2+fp3+fn+fn2+fn3+tn+tn2+tn3)
    rec = (tp+tp2+tp3)/(tp+tp2+tp3+fn+fn2+fn3)
    pre = (tp+tp2+tp3)/(tp+tp2+tp3+fp+fp2+fp3)
    print ('Average Accuracy:', acc)
    print ('Average Sensitivity(Recall):', rec)
    print ('Average Precision:', pre)
hold_out()
K_fold()
