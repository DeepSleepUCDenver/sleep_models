import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelSpreading
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from imblearn.over_sampling import SMOTE
from oversample import load_all_data 


x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()

from sklearn import svm
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(x_tr, y_tr)
svm_linear.score(x_te, y_te)
svm_linear.score(x_va, y_va)

svm_poly = svm.SVC(kernel='poly')
svm_poly.fit(x_tr, y_tr)
svm_poly.score(x_te, y_te)
svm_poly.score(x_va, y_va)

svm_rbf  = svm.SVC(kernel='rbf')
svm_rbf.fit(x_tr, y_tr)
svm_rbf.score(x_te, y_te)
svm_rbf.score(x_va, y_va)

svm_sigmoid = svm.SVC(kernel='sigmoid')
svm_sigmoid.fit(x_tr, y_tr)
svm_sigmoid.score(x_te, y_te)
svm_sigmoid.score(x_va, y_va)


from sklearn.ensemble import RandomForestClassifier

rdf = RandomForestClassifier(max_depth=4, random_state=0)
rdf.fit(x_tr, y_tr)
rdf.score(x_te, y_te)
rdf.score(x_va, y_va)

from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier(random_state=0)
adb.fit(x_tr, y_tr)
adb.score(x_te, y_te)
adb.score(x_va, y_va)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_tr, y_tr)
knn.score(x_te, y_te)
knn.score(x_va, y_va)
knn.predict(x_va)

from sklearn.naive_bayes import GaussianNB

bay = GaussianNB()
bay.fit(x_tr, y_tr)
bay.score(x_te, y_te)
bay.score(x_va, y_va)


# example of semi-supervised gan for mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv1D
# from keras.layers import Conv1D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import TimeDistributed
from keras.layers import Activation
from matplotlib import pyplot

from keras import backend
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

def edu(x):
    r = []
    for i in x:
        m = 0
        for j, v in enumerate(i):
            if v > m :
                m = v
                c = j
        r.append(c)
    return r

# define the standalone supervised and unsupervised discriminator models
start = Input(shape=(22,))
fe = Dense(21)(start)
fe = Dense(15)(fe)
fe = Dense(10)(fe)
fe = Dense(5)(fe)
c_out_layer = Activation('sigmoid')(fe)
# define and compile supervised discriminator model
c_model = Model(start, c_out_layer)
c_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#c_model.compile(loss='sparce_binary_crossentropy', optimizer='adam', metrics=['accuracy'])
c_model.summary()


x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()
y_tr = pd.get_dummies(y_tr).values
y_te = pd.get_dummies(y_te).values
y_va = pd.get_dummies(y_va).values

c_model.fit(x_tr, y_tr)
hy_te = c_model.predict(x_te)
hy_va = c_model.predict(x_va)
accuracy_score(edu(hy_te), edu(y_te))
accuracy_score(edu(hy_va), edu(y_va))

# define the standalone supervised and unsupervised discriminator models
start = Input(shape=(22,))
fe = Dense(22)(start)
fe = Dense(44)(fe)
fe = Dense(20)(fe)
fe = Dense(15)(fe)
fe = Dense(10)(fe)
fe = Dense(8)(fe)
fe = Dense(5)(fe)
c_out_layer = Activation('LeakyReLU')(fe)
# define and compile supervised discriminator model
c_model = Model(start, c_out_layer)
c_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#c_model.compile(loss='sparce_binary_crossentropy', optimizer='adam', metrics=['accuracy'])
c_model.summary()


x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()
y_tr = pd.get_dummies(y_tr).values
y_te = pd.get_dummies(y_te).values
y_va = pd.get_dummies(y_va).values

c_model.fit(x_tr, y_tr)
hy_te = c_model.predict(x_te)
hy_va = c_model.predict(x_va)
accuracy_score(edu(hy_te), edu(y_te))
accuracy_score(edu(hy_va), edu(y_va))


disp = plot_confusion_matrix(c_model, x_te, y_te,
                             cmap=plt.cm.Blues
                             ,normalize='true')
disp.ax_.set_title("RBF Kernel with " + str(n_features) + " best features")
cfm = disp.plot()
cfm.figure_.savefig("gru1.png")

