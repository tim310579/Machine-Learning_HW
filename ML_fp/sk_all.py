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
#train.drop(delete, inplace = True)
#train = train.drop('Choose hometown or travel', axis = 1)
fea = ['Choose hometown or travel','age', 'job', 'back home frequency', 'location of work (school)', 'hometown location', 'distance between the above two', 'interpersonal relationship', 'family relationship', 'gender', 'financial situation(income)', 'have boy/girlfriend/husband/wife']

le = preprocessing.LabelEncoder()
for col in fea:   #transform to numeric data
    encode_col = le.fit_transform(train[col])
    train[col] = encode_col
train['transportation'].replace(['train bus or ship'], [1], inplace = True)
train['transportation'].replace(['HSR or airplane'], [2], inplace = True)
train['transportation'].replace(['drive or ride by yourself'], [3], inplace = True)


choose1 = train[train['transportation'] == 1].index
choose2 = train[train['transportation'] == 2].index
choose3 = train[train['transportation'] == 3].index

df1 = train.loc[choose1]
df2 = train.loc[choose2]
df3 = train.loc[choose3]
df1 = df1.reset_index(drop = True)
df2 = df2.reset_index(drop = True)
df3 = df3.reset_index(drop = True)
len1 = len(df1)
len2 = len(df2)
len3 = len(df3)

df1_test = df1[int(len1*0.7): len1]
df2_test = df2[int(len2*0.7): len2]
df3_test = df3[int(len3*0.7): len3]
df1 = df1[0: int(len1*0.7)]
df2 = df2[0: int(len2*0.7)]
df3 = df3[0: int(len3*0.7)]

train = pd.concat([df1, df2, df3], axis = 0)
train = train.reset_index(drop = True)
test = pd.concat([df1_test, df2_test, df3_test], axis =0)
test = test.reset_index(drop = True)

target = train['transportation']
train = train.drop(columns = 'transportation')
test_target = test['transportation']
test = test.drop(columns = 'transportation')

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

combine = []
df_combine = pd.DataFrame(result)
for col in df_combine.columns:
    combine.append(df_combine[col].mode()[0])

name = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Neural Network']

for i in range(5):
    print_matrix(name[i], result[i], test_target)
print_matrix('Combine All', combine, test_target)
