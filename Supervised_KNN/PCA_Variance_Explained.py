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
from sklearn.preprocessing import scale, normalize, StandardScaler

'''
Variance Expained with the curated features
'''
data = pd.read_feather('feature_stage_data_all.ftr')
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
pca = PCA(n_components=5)
pca.fit(X)
#principalComponents = pca.fit_transform(X)
#variance = principalComponents.explained_variance_ratio_
print(pca.explained_variance_ratio_)
#print(va)
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
print(var)
#print(np.cumsum(np.round(principalComponents.explained_variance_ratio_, decimals=3)*100))



'''
Variance Explained on Raw Data
'''


def load_data():
    x = np.load('../reshap_to_npy/all_raw_data_X.npy')
    #x =  preprocessing.StandardScaler(x)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    print(x.shape)
    pca = PCA(250)
    print("1")
    x = pca.fit_transform(x)
    #take your pick
    print("2")

    # for i in range(3):
    #     x[:,:,i] = scale(x[:,:,i])
    y = np.load('../reshap_to_npy/all_raw_data_y.npy')

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.20)
    print("3")
    return  x_tr, x_te, y_tr, y_te

#Test variance on
X = load_data()[0]


#Run code below on None PCA'd data.
'''
pca = PCA(n_components=250)
pca.fit(X)
#principalComponents = pca.fit_transform(X)
#variance = principalComponents.explained_variance_ratio_
print(pca.explained_variance_ratio_)
#print(va)
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
print(var)
'''