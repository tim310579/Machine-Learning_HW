import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets

if __name__ == '__main__':
    np.random.seed(0)
    data = datasets.load_iris()
    iris_types = data[4].unique()
    n_class = iris_types.size
    x = data.iloc[:, :2]  #只取前面兩個特徵
    y = pd.Categorical(data[4]).codes  #將標籤轉換0,1,...
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, random_state = 0)
    y_one_hot = label_binarize(y_test, np.arange(n_class))  #裝換成類似二進位制的編碼
    alpha = np.logspace(-2, 2, 20)  #設定超引數範圍
    model = LogisticRegressionCV(Cs = alpha, cv = 3, penalty = 'l2')  #使用L2正則化
    model.fit(x_train, y_train)
   
    # 計算屬於各個類別的概率，返回值的shape = [n_samples, n_classes]
    y_score = model.predict_proba(x_test)
    # 1、呼叫函式計算micro型別的AUC
   
    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    
    #繪圖
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    #FPR就是橫座標,TPR就是縱座標
    plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'鳶尾花資料Logistic分類後的ROC和AUC', fontsize=17)
    plt.show()
