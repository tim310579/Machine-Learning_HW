import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

train = pd.read_csv('final.csv',engine='python')

fea = ['choose hometown or travel', 'job', 'back home frequency', 'location of work (school)', 'hometown location', 'distance between the above two', 'interpersonal relationship', 'family relationship', 'gender', 'financial situation(income)', 'have boy or girlfriend or husband or wife']

for col in fea:
  
    size = train[col].value_counts()
    separeted = np.zeros([len(size)])
    labels = []
    for ind in size.index:    
        labels.append(ind)
    print(labels)
    
    plt.pie(size, labels = labels, autopct = "%1.1f%%", explode = separeted,  pctdistance = 0.6, shadow=True)
    plt.title('Pie chart of %s'%col, {"fontsize" : 18})
    plt.legend(loc = "best")
    plt.savefig('picture/%s'%col)
    plt.close()
