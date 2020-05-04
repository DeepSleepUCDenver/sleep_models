import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelSpreading
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from over_sample_data import load_all_data
from over_sample_data import load_known_data

n_features = 10

def do_sfs(x_tr, y_tr):
    sfs_kern = sfs(svm.SVC(kernel='rbf'),
               k_features=n_features,
               forward=True,
               floating=True,
               verbose=2,
               scoring='accuracy',
               cv=5)
    sfs_kern.fit(x_tr, y_tr)
    return sfs_kern


x_tr, y_tr, x_te, y_te, x_va, y_va = load_known_data()

best = do_sfs(x_tr, y_tr)

# examine the results

plot = plot_sfs(best.get_metric_dict())
                             
plot[1].figure.savefig("SFS-loss" + str(n_features) + ".png")

score = []
for i in range(1,11):
    score.append(best.get_metric_dict()[i]['avg_score'])

index = list(best.k_feature_idx_)
results = pd.DataFrame({'dex': index, 'score': score})
results = results.head(9)
#results = results.sort_values(by='score', ascending=False)

test_svm(x_te, y_te)

# make a more select dataset
# Filter the rest of the data
x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()
#x_obs, y_obs, x_nuls = load_data()
#keep = list(best.k_feature_idx_)
keep = results.dex.values
np.save('sfs_features', keep)
# keep = np.load('sfs_features.npy')
x_tr = x_tr[:,keep]
x_te = x_te[:,keep]
x_va = x_va[:,keep]

x_tr.shape

# # apply LabelSpreading
# label_spread = LabelSpreading(kernel='knn', alpha=0.8)
# label_spread.fit(x_obs, y_obs)
# x_all = np.concatenate([x_obs, x_nuls], axis=0)
# y_all = np.concatenate([y_obs, label_spread.predict(x_nuls)], axis=0)
# 
# x, y = shuffle(x_all, y_all, random_state=42)
# smpnum = min([sum(y==i) for i in range(1,6)])
# y_btr = y[y == 1][:smpnum]
# x_btr = x[y == 1][:smpnum]
# for i in range(2,6):
#     x_btr = np.concatenate([x_btr, x[y == i][:smpnum]])
#     y_btr = np.concatenate([y_btr, y[y == i][:smpnum]])
# 
# x_tr, x_te, y_tr, y_te = train_test_split(x_btr, y_btr, test_size = 0.20)

mod = svm.SVC(kernel='rbf')
mod.fit(x_tr, y_tr)
mod.score(x_te, y_te)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the example flights dataset and convert to long-form
#flights_long = sns.load_dataset("flights")
#flights = flights_long.pivot("month", "year", "passengers")
piv = .pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

disp = plot_confusion_matrix(mod, x_te, y_te,
                             cmap=plt.cm.Blues
                             ,normalize='true')
disp.ax_.set_title("RBF Kernel with " + str(n_features) + " best features no spreading")
cfm = disp.plot()
cfm.figure_.savefig("CM-SVM-RBF-" + str(n_features) + ".png")

disp = plot_confusion_matrix(mod, x_te, y_te,
                             cmap=plt.cm.Reds
                             ,normalize='true')

disp.ax_.set_title("RBF Kernel with " + str(n_features) + " best features")
cfm = disp.plot()
cfm.figure_.savefig("CM-SVM-RBF-" + str(n_features) + ".png")

