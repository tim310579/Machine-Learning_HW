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
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


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
#train = train.sample(frac=1).reset_index(drop = True) #shuffle
train.replace(np.nan, 23, inplace=True)
for col in fea:
    train[col].replace(["?"],  [train[col].mode()], inplace = True)   #deal with missing
delete =  train[train['Choose hometown or travel'] == 'travel'].index
#train.drop(delete, inplace = True)
#train = train.reset_index(drop = True)
#train = train.drop('Choose hometown or travel', axis = 1)
fea = ['Choose hometown or travel', 'job', 'back home frequency', 'location of work (school)', 'hometown location', 'distance between the above two', 'interpersonal relationship', 'family relationship', 'gender', 'financial situation(income)', 'have boy/girlfriend/husband/wife']
'''
for i in range(len(train)):
    if(train.iat[i, 1] < 18): train.iat[i, 1] = 0
    elif(train.iat[i, 1] >= 18 and train.iat[i, 1] < 22): train.iat[i, 1] = 1
    elif(train.iat[i, 1] >= 22 and train.iat[i, 1] < 26): train.iat[i, 1] = 2
    elif(train.iat[i, 1] >= 26 and train.iat[i, 1] < 30): train.iat[i, 1] = 3
    elif(train.iat[i, 1] >= 30 and train.iat[i, 1] < 35): train.iat[i, 1] = 4
    elif(train.iat[i, 1] >= 35 and train.iat[i, 1] < 40): train.iat[i, 1] = 5
    else: train.iat[i, 1] = 6
'''
le = preprocessing.LabelEncoder()
for col in fea:   #transform to numeric data
    encode_col = le.fit_transform(train[col])
    train[col] = encode_col
train['transportation'].replace(['train bus or ship'], [0], inplace = True)
train['transportation'].replace(['HSR or airplane'], [1], inplace = True)
train['transportation'].replace(['drive or ride by yourself'], [2], inplace = True)

#delete = ['location of work (school)', 'hometown location','interpersonal relationship', 'family relationship','have boy/girlfriend/husband/wife','gender']
#train = train.drop(columns = delete)
labels = ['train bus or ship','drive or ride by yourself','HSR or airplane']
separeted = [0, 0.2, 0.2]
size = train['transportation'].value_counts()
plt.pie(size, labels = labels,  autopct = "%1.1f%%", explode = separeted,  pctdistance = 0.6, shadow=True)
plt.title('Pie chart of transportation', {"fontsize" : 18})
plt.legend(loc = "best")
plt.savefig('Pie chart of transportation.png')
plt.close()

train_all = train.copy()
train_all = train_all.drop(columns = 'transportation')
test_all = train['transportation']

train, test, target, test_target = train_test_split(train_all, test_all, test_size=0.30, random_state=0)
n_class = 3
y_bin = label_binarize(test_target, np.arange(n_class))
alpha = np.logspace(-2, 2, 20)
 
model = []
model.append(DecisionTreeClassifier(max_depth = 8).fit(train, target))
model.append(RandomForestClassifier(n_estimators=10, max_depth = 8, random_state=42).fit(train, target))
model.append(GaussianNB().fit(train, target))
model.append(SVC(gamma = 'auto', probability=True).fit(train, target))
model.append(LogisticRegression(max_iter=500).fit(train, target))
#model.append(MLPClassifier(max_iter=600).fit(train, target))

name = ['Decision Tree', 'Random Forest', 'Naive Bayes','SVM', 'Logistic Regression']
color = ['r', 'g', 'b', 'pink', 'k']
for i in range(5):
    print_matrix(name[i], model[i], test, test_target)
    draw(name[i], model[i], y_bin, test, color[i])
#print(roc_auc_score(test_target, y_pred_proba))

results = []
kfold = KFold(n_splits=10, random_state = 42, shuffle = True)
print('K-fold for K = 10')
for i in range(5):
    results.append(model_selection.cross_val_score(model[i], train_all, test_all, cv = kfold))
    print(name[i], 'Accuracy : ',results[i].mean())

plt.savefig('ROC curve.png')
plt.close()

#print(model[4])
