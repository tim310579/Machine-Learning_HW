import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model  import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def draw(name, model, y_bin, test, cl):
    y_score = model.predict_proba(test)
    fpr, tpr, thresholds = metrics.roc_curve(y_bin.ravel(),y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(fpr, tpr, c = cl, lw = 2, alpha = 0.7, label = u'%s AUC=%.3f' % (name, auc))
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'ROC and AUC for 6 models', fontsize=17)

def print_matrix(name, model, test, test_target):
    predict = model.predict(test)
    mat = confusion_matrix(test_target, predict)
    acc = model.score(test, test_target)
    pre = precision_score(test_target, predict, average=None)
    rec = recall_score(test_target, predict, average=None)
    data_frame = pd.DataFrame(mat, index = ['Actual train, bus, ship', 'Actual HSR, airplane', 'Actual drive, ride'], columns = ['Predict train, bus, ship', 'HSR, airplane', 'drive, ride'])
    print(name,'----------------------------------------------------------------------')
    print(data_frame)
    print('')
    print('Accuracy: ', acc)
    df_rp = pd.DataFrame([rec, pre], index = ['Recall', 'Precision'], columns = ['train, bus, ship', 'HSR, airplane', 'drive, ride'])
    #print('Precision: ', pre)
    #print('Recall: ', rec)
    print(df_rp)
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
train['transportation'].replace(['train bus or ship'], [0], inplace = True)
train['transportation'].replace(['HSR or airplane'], [1], inplace = True)
train['transportation'].replace(['drive or ride by yourself'], [2], inplace = True)

for i in range(len(train)):
    if(train.iat[i, 1] < 18): train.iat[i, 1] = 0
    elif(train.iat[i, 1] >= 18 and train.iat[i, 1] < 22): train.iat[i, 1] = 1
    elif(train.iat[i, 1] >= 22 and train.iat[i, 1] < 26): train.iat[i, 1] = 2
    elif(train.iat[i, 1] >= 26 and train.iat[i, 1] < 30): train.iat[i, 1] = 3
    elif(train.iat[i, 1] >= 30 and train.iat[i, 1] < 35): train.iat[i, 1] = 4
    elif(train.iat[i, 1] >= 35 and train.iat[i, 1] < 40): train.iat[i, 1] = 5
    else: train.iat[i, 1] = 6
    
train_all = train.copy()
train_all = train_all.drop(columns = 'transportation')
test_all = train['transportation']
#test_all = label_binarize(test_all, classes=[0, 1, 2])
#train, test, target, test_target = model_selection.train_test_split(train_all, test_all, test_size=0.30, random_state=100)
#result_k0 = model_selection.cross_val_score(model0, train_all, test_all, cv = kfold)
#result_k1 = model_selection.cross_val_score(model1, train_all, test_all, cv = kfold)
#result_k2 = model_selection.cross_val_score(model2, train_all, test_all, cv = kfold)
#result_k3 = model_selection.cross_val_score(model3, train_all, test_all, cv = kfold)
#result_k4 = model_selection.cross_val_score(model4, train_all, test_all, cv = kfold)
#test_all = label_binarize(test_all,classes=[0,1,2])
'''n_classes = test_all.shape[1]

random_state = np.random.RandomState(0)
n_samples, n_features = train_all.shape
train_all = np.c_[train_all,random_state.randn(n_samples,200 * n_features)]
'''
train, test, target, test_target = train_test_split(train_all, test_all, test_size=0.30, random_state=0)
n_class = 3
y_bin = label_binarize(test_target, np.arange(n_class))
alpha = np.logspace(-2, 2, 20)
 
model = []
model.append(DecisionTreeClassifier(max_depth = 8).fit(train, target))
model.append(RandomForestClassifier(n_estimators=10, max_depth = 8, random_state=42).fit(train, target))
model.append(GaussianNB().fit(train, target))
model.append(SVC(gamma = 'auto', probability=True).fit(train, target))
model.append(LogisticRegression().fit(train, target))
model.append(MLPClassifier().fit(train, target))

name = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'SVM', 'Logistic Regression', 'Neural Network']
color = ['r', 'g', 'b', 'pink', 'k', 'orange']
for i in range(6):
    print_matrix(name[i], model[i], test, test_target)
    draw(name[i], model[i], y_bin, test, color[i])
#print(roc_auc_score(test_target, y_pred_proba))
plt.show()
