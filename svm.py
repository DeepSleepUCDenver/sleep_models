import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.semi_supervised import label_propagation

data = pd.read_feather('./feature_stage_data.ftr')
X = data[data.columns[3:]]
y = data['stage']
X = X.values
X = preprocessing.normalize(X)
#X = preprocessing.scale(X)
y = y.values
#y = pd.get_dummies(y)
#y = y.loc[:,1]

clf = svm.SVC()
clf.fit(X, y)
clf.score(X, y)
yp = clf.predict(X)

accuracy_score(y, [3 for i in range(len(y))])



data = pd.read_feather('./feature_stage_data.ftr')
data = pd.concat([data.iloc[:,3:], data.iloc[:,1]], axis=1)
dfs = [x for _, x in data.groupby(data.stage)]
len(dfs)
smpl = pd.DataFrame(columns=data.columns)
for df in dfs:
    s = df.sample(n=100, random_state=74)
    smpl = pd.concat([smpl, s], axis=0)

smpl.stage 
smpl.shape

X = smpl[data.columns[:21]]
y = smpl['stage']
#X = pd.to_numeric(X)
X = X.values
#X = preprocessing.normalize(X)
X = preprocessing.scale(X)
y = pd.to_numeric(y)
y = y.values

clf = svm.SVC()
clf.fit(X, y)
clf.score(X, y)
yp = clf.predict(X)
yp == 3
for i in yp:
    if i != 3:
        print(i)

accuracy_score(y, yp)
accuracy_score(y, [3 for i in range(len(y))])



