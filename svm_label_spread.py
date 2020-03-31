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


# Read am partition the matrix
data = pd.read_feather('./feature_stage_data_all.ftr')
x = data[data.columns[3:]]
y = data['stage']
x = x.values
x = normalize(x)
y = y.values

nnl = lambda a: np.invert(np.isnan(a))
nul = lambda a: np.isnan(a)
x_obs = x[nnl(y)]
y_obs = y[nnl(y)]

# apply LabelSpreading
x_nuls = x[nul(y)]
label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(x_obs, y_obs)
x_all = np.concatenate([x_obs, x_nuls], axis=0)
y_all = np.concatenate([y_obs, label_spread.predict(x_nuls)], axis=0)

def test_svm(x, y):
    x, y = shuffle(x, y, random_state=42)
    smpnum = min([sum(y==i) for i in range(1,6)])
    y_btr = y[y == 1][:smpnum]
    x_btr = x[y == 1][:smpnum]
    for i in range(2,6):
        x_btr = np.concatenate([x_btr, x[y == i][:smpnum]])
        y_btr = np.concatenate([y_btr, y[y == i][:smpnum]])
    x_tr, x_te, y_tr, y_te = train_test_split(x_btr, y_btr, test_size = 0.20)
    kerns = [
            'linear',
            'poly',
            'rbf',
            'sigmoid'
            ]

    kern_svms = [
            svm.SVC(kernel='linear'),
            svm.SVC(kernel='poly'),
            svm.SVC(kernel='rbf'),
            svm.SVC(kernel='sigmoid')
            ]
    for kern, mod in zip(kerns, kern_svms):
        # Build step forward feature selection
        sfs_kern = sfs(mod,
                   k_features=2,
                   forward=True,
                   floating=False,
                   verbose=2,
                   scoring='accuracy',
                   cv=5)
        print(kern)
        sfs_kern = sfs_kern.fit(x_tr, y_tr)
        #print(mod.score(x_te, y_te))
        break
    return sfs_kern


best = test_svm(x_obs, y_obs)

plot = plot_sfs(best.get_metric_dict())
import seaborn as sea
best.get_metric_dict()
plot = sea.lineplot(y='measure', x='index', hue='wave', data=data_joined)
plot.savefig("waveplot.png")
plot[1].figure.savefig("liner.png")

test_svm(x_all, y_all)
 
import matplotlib.pyplot as plt

data = pd.read_feather("../flat_data/Staging_Data_GT_190315.ftr")

#sea.lineplot(y='eeg', x='index', data=data[:500])
#plt.show()

eeg = data[['eeg']][:1000]
eog = data[['eog']][:1000]
emg = data[['emg']][:1000]
eeg.columns = ['measure']
eog.columns = ['measure'] 
emg.columns = ['measure']
eeg['wave'] = 'eeg'
eog['wave'] = 'eog'
emg['wave'] = 'emg'
data_joined = pd.concat([eeg, eog, emg], axis=0)
data_joined['index'] = data_joined.index
plot = sea.lineplot(y='measure', x='index', hue='wave', data=data_joined)
plot.figure.savefig("waveplot.png")


plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


