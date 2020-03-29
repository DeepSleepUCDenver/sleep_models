import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
from sklearn.decomposition import PCA
import numpy as np

data = pd.read_feather('feature_stage_data.ftr')
sampling_rate = min([x.shape[0] for _, x in data.groupby(data.stage)]) #Smallest Number of rows for any label
dfs = [x.sample(sampling_rate-1) for _, x in data.groupby(data.stage)]
samples_dfs = []

    #samples_dfs.append(x.sample(2000))
data = pd.concat(dfs, axis=0)
X = data[data.columns[3:]]

y = data['stage']

X = preprocessing.normalize(X)

'''
PCA ATTEMPT
'''
pca = PCA(n_components=10)
#pca.fit(X)
principalComponents = pca.fit_transform(X)
#variance = principalComponents.explained_variance_ratio_
#print(va)
#var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
#print(var)
#print(np.cumsum(np.round(principalComponents.explained_variance_ratio_, decimals=3)*100))


y = y.values
principalDf = pd.DataFrame(data = principalComponents)
finalDf = pd.concat([principalDf.reset_index(drop=True), data['stage'].reset_index(drop=True)], axis = 1)



a ="meh"




'''
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'y', 'v']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['stage'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
'''

'''
PCA ATTEMPT
'''

X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size=0.33, random_state=42)

epoch_count = []
test_acc = []
train_acc = []

#Simple Case
for i in range(1,50):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    epoch_count.append(i)
    train_acc.append(neigh.score(X_train,y_train))
    test_acc.append(neigh.score(X_test,y_test))

matplotlib.pyplot.plot(epoch_count, test_acc, 'b',  linewidth=2.0, label="Testing")
matplotlib.pyplot.plot(epoch_count, train_acc, 'g',  linewidth=2.0, label="Training")
matplotlib.pyplot.ylabel("Accuracy")
matplotlib.pyplot.xlabel("Neighbors")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
