from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing, linear_model
import pandas as pd
import numpy as np

fea = ['Choose hometown or travel', 'job', 'back home frequency', 'location of work (school)', 'hometown location', 'distance between the above two', 'interpersonal relationship', 'family relationship', 'gender', 'financial situation(income)', 'have boy/girlfriend/husband/wife']

train = pd.read_csv('tran.csv')
train = train.sample(frac=1).reset_index(drop = True) #shuffle
train.replace(np.nan, 23, inplace=True)
for col in fea:
    train[col].replace(["?"],  [train[col].mode()], inplace = True)   #deal with missing
delete =  train[train['Choose hometown or travel'] == 'travel'].index
train.drop(delete, inplace = True)
#train = train.drop('Choose hometown or travel', axis = 1)
fea = ['Choose hometown or travel','age', 'job', 'back home frequency', 'location of work (school)', 'hometown location', 'distance between the above two', 'interpersonal relationship', 'family relationship', 'gender', 'financial situation(income)', 'have boy/girlfriend/husband/wife']

le = preprocessing.LabelEncoder()
for col in fea:   #transform to numeric data
    encode_col = le.fit_transform(train[col])
    train[col] = encode_col
#print(train)

split = len(train)
test = train[int(split*0.7): split]
train = train[0: int(split*0.7)]
target = train['transportation']
train = train.drop(columns = 'transportation')

test_target = test['transportation']
test = test.drop(columns = 'transportation')


target.replace(['train bus or ship'], 1, inplace = True)
target.replace(['HSR or airplane'], 2, inplace = True)
target.replace(['drive or ride by yourself'], 3, inplace = True)
test_target.replace(['train bus or ship'], 1, inplace = True)
test_target.replace(['HSR or airplane'], 2, inplace = True)
test_target.replace(['drive or ride by yourself'], 3, inplace = True)


model = DecisionTreeClassifier(max_depth = 8)
#model = tree.DecisionTreeClassifier()
model = model.fit(train, target)

result = model.predict(test)
#print(result)
#print(test_target)
test_target = test_target.reset_index(drop = True)
mat = np.zeros([3, 3])
for i in range(len(result)):
    if(result[i] == 1):
        if(test_target[i] == 1):
            mat[0][0] += 1
        elif(test_target[i] == 2):
            mat[1][0] += 1
        else:
            mat[2][0] += 1
    elif(result[i] == 2):
        if(test_target[i] == 1):
            mat[0][1] += 1
        elif(test_target[i] == 2):
            mat[1][1] += 1
        else:
            mat[2][1] += 1
    else:
        if(test_target[i] == 1):
            mat[0][2] += 1
        elif(test_target[i] == 2):
            mat[1][2] += 1
        else:
            mat[2][2] += 1

data_frame = pd.DataFrame(mat, index = ['Actual train, bus, ship', 'Actual HSR, airplane', 'Actual drive, ride'], columns = ['Predict train, bus, ship', 'HSR, airplane', 'drive, ride'])
print('One tree')
print(data_frame)
acc = (mat[0][0] + mat[1][1] + mat[2][2])/len(result)
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
print(' ')
print(' ')


model2 = RandomForestClassifier(max_depth = 8)
model2 = model2.fit(train, target)
result2 = model2.predict(test)
mat2 = np.zeros([3, 3])
for i in range(len(result2)):
    if(result2[i] == 1):
        if(test_target[i] == 1):
            mat2[0][0] += 1
        elif(test_target[i] == 2):
            mat2[1][0] += 1
        else:
            mat2[2][0] += 1
    elif(result2[i] == 2):
        if(test_target[i] == 1):
            mat2[0][1] += 1
        elif(test_target[i] == 2):
            mat2[1][1] += 1
        else:
            mat2[2][1] += 1
    else:
        if(test_target[i] == 1):
            mat2[0][2] += 1
        elif(test_target[i] == 2):
            mat2[1][2] += 1
        else:
            mat2[2][2] += 1
data_frame2 = pd.DataFrame(mat2, index = ['Actual train, bus, ship', 'Actual HSR, airplane', 'Actual drive, ride'], columns = ['Predict train, bus, ship', 'HSR, airplane', 'drive, ride'])
print('Forest with 100 trees')
print(data_frame2)
acc2 = (mat2[0][0] + mat2[1][1] + mat2[2][2])/len(result2)
rec2 = []
pre2 = []
for i in range(3):
    rec2.append(mat2[i][i]/(mat2[i][0] + mat2[i][1] + mat2[i][2]))
    pre2.append(mat2[i][i]/(mat2[0][i] + mat2[1][i] + mat2[2][i]))
rec_pre2 = np.array([rec2, pre2])
#print(rec_pre)
rp2 = pd.DataFrame(rec_pre2, index = ['Sensitivity(Recall)', 'Precision'], columns = ['train, bus, ship', 'HSR, airplane', 'drive, ride'])
print('Accuracy: ', acc2)
print(rp2)
print(model)
print(model2)
