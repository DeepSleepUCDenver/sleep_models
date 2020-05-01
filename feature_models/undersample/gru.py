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




# load the images
def load_data():
    data = pd.read_feather('./feature_stage_data.ftr')
    X = data[data.columns[3:]]
    y = pd.get_dummies(data['stage'])
    y = (data['stage'] - 1)
    X = X.values
    X = normalize(X)
    X = X[data.observation != 19]
    y = y[data.observation != 19]
    X_te = data[data.columns[3:]][data.observation == 19]
    y_te = data['stage'][data.observation == 19]
    #X = preprocessing.scale(X)
    y = y.values
    return [X, y, X_te, y_te]



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
n_classes=5
in_shape=(30000,3)
# image input
start = Input(shape=(21,))
# input layer nodes
fe = Dense(21)(start)
# input layer nodes
fe = Dense(15)(fe)
# input layer nodes
fe = Dense(10)(fe)
fe = Reshape((10,1))(fe)
#fe = LeakyReLU(alpha=0.2)(fe)
fe = Dense(n_classes)(fe)
fe = GRU(30, return_sequences=True)(fe)
fe = TimeDistributed(Dense(n_classes))(fe)
fe = Flatten()(fe)
fe = Dense(n_classes)(fe)
# output layer nodes
# supervised output
c_out_layer = Activation('sigmoid')(fe)
# define and compile supervised discriminator model
c_model = Model(start, c_out_layer)
c_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
c_model.compile(loss='sparce_binary_crossentropy', optimizer='adam', metrics=['accuracy'])
c_model.summary()


# size of the latent space
# create the discriminator models
x_tr, y_tr, x_te, y_te = load_data()
#type(y_tr)
y_tr = pd.get_dummies(y_tr).values

c_model.train_on_batch(x_tr, y_tr)
c_model.fit(x_tr, y_tr)
c_model.evaluate(x_tr, y_tr)
hy_tr = c_model.predict(x_tr)
accuracy_score(edu(hy_tr), edu(y_tr))


disp = plot_confusion_matrix(c_model, x_te, y_te,
                             cmap=plt.cm.Blues
                             ,normalize='true')
disp.ax_.set_title("RBF Kernel with " + str(n_features) + " best features")
cfm = disp.plot()
cfm.figure_.savefig("gru1.png")

