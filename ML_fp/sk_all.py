from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model  import LogisticRegression
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np



def print_matrix(name, result, test_target):
    
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
    print(name,'----------------------------------------------------------------------')
    print(data_frame)
    acc = (mat[0][0] + mat[1][1] + mat[2][2])/len(result)
    rec = []
    pre = []
    for i in range(3):
        rec.append(mat[i][i]/(mat[i][0] + mat[i][1] + mat[i][2]))
        pre.append(mat[i][i]/(mat[0][i] + mat[1][i] + mat[2][i]))
    rec_pre = np.array([rec, pre])

    rp = pd.DataFrame(rec_pre, index = ['Sensitivity(Recall)', 'Precision'], columns = ['train, bus, ship', 'HSR, airplane', 'drive, ride'])
    print('Accuracy: ', acc)
    print(rp)
    print(' ')
    print(' ')


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
test_target = test_target.reset_index(drop = True)

model0 = DecisionTreeClassifier(max_depth = 8).fit(train, target)
model1 = RandomForestClassifier(max_depth = 8).fit(train, target)
model2 = GaussianNB().fit(train, target)
model3 = LogisticRegression().fit(train, target)
model4 = MLPClassifier().fit(train, target)

result = []
result.append(model0.predict(test))
result.append(model1.predict(test))
result.append(model2.predict(test))
result.append(model3.predict(test))
result.append(model4.predict(test))

name = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Neural Network']

for i in range(5):
    print_matrix(name[i], result[i], test_target)

