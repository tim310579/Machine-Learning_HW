import numpy as np
import csv
import pandas as pd
import re
import math
c = ['eaten','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-numbe','ring-type','spore-print-color','population','habitat']
#df = pd.read_csv('agaricus-lepiota.data')
#count = df.groupby('p').size()
#sr = pd.Series(df, index = c)
df = pd.read_csv('agaricus-lepiota.data', header = None)
df.columns = [c]
for col in df.columns:
    delete = df[df[col] == "?"].index
    #df.drop(delete, inplace = True)
print(df.tail())
#ct = df.groupby(["eaten"]).size()
#val_fre = pd.DataFrame()
for col in df.columns:
    val_fre = pd.DataFrame()
    val_fre = val_fre.append(df[col].value_counts())
    #val_fre = val_fre.append(df[col].value_counts(normalize = True))
    print(val_fre)
with open('output.data', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    df.to_csv('output.data')
#print(count)
