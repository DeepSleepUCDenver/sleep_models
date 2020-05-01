from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import warnings
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
# load all the layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import UpSampling2D 
from tensorflow.keras.layers import ZeroPadding2D # load model
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model, Model
# load back end
from tensorflow.keras import backend
# load optimizers
from tensorflow.keras.optimizers import Adam
# load other tensor stuff
import tensorflow as tf
# load other stuff stuff
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def educate_dummie(dum):
    out = []
    for r in dum:
        maxval = max(r)
        # print(maxval)
        for i, c in enumerate(r):
            if c == maxval:
                out.append(i)
                break
    return out

# GRU

data = pd.read_feather('./feature_stage_data.ftr')

X = data[data.columns[3:]]
X = X.values
X = preprocessing.normalize(X)
X = X.reshape((X.shape[0], X.shape[1], 1))

y = data['stage']
y = pd.get_dummies(y)
y = y.values
y = preprocessing.normalize(y)

x_start = Input(shape=(21,1))
x = x_start
x = GRU(21, activation='tanh', return_sequences=True)(x)
x = TimeDistributed(Dense(5))(x)
x = Flatten()(x)
x = Dense(50,activation = 'sigmoid')(x)
x = Dense(5,activation = 'sigmoid')(x)
mod = Model(inputs=x_start, outputs=x)
mod.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
mod.summary()


mod.train_on_batch(X, y)
ye = educate_dummie(y)
yp = educate_dummie(mod.predict(X))
accuracy_score(ye, yp)

# LSTM

data = pd.read_feather('./feature_stage_data.ftr')

X = data[data.columns[3:]]
X = X.values
X = preprocessing.normalize(X)
X = X.reshape((X.shape[0], X.shape[1], 1))

y = data['stage']
y = pd.get_dummies(y)
y = y.values
y = preprocessing.normalize(y)

x_start = Input(shape=(21,1))
x = x_start
x = LSTM(21, activation='tanh', return_sequences=True)(x)
x = TimeDistributed(Dense(5))(x)
x = Flatten()(x)
x = Dense(50,activation = 'sigmoid')(x)
x = Dense(5,activation = 'sigmoid')(x)
mod = Model(inputs=x_start, outputs=x)
mod.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
mod.summary()


mod.train_on_batch(X, y)
ye = educate_dummie(y)
yp = educate_dummie(mod.predict(X))
pd.concat([
        pd.DataFrame(ye, columns=['true']),
        pd.DataFrame(yp, columns=['pred'])], axis=1)
accuracy_score(ye, yp)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

data = pd.read_feather('./feature_stage_data.ftr')
X = data[data.columns[3:]]
y = data['stage']
X = X.values
X = preprocessing.normalize(X)
y = y.values

clf = AdaBoostClassifier(n_estimators=1000, random_state=0)

clf.fit(X,)

clf.score(X, y)
accuracy_score(X, y)
yp = clf.predict(X)
accuracy_score(y, yp)




import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm

data = pd.read_feather('./feature_stage_data.ftr')
X = data[data.columns[3:]]
y = data['stage']
X = X.values
X = preprocessing.normalize(X)
#X = preprocessing.scale(X)
y = y.values
y = pd.get_dummies(y)
y = y.loc[:,1]

clf = svm.SVC()
clf.fit(X, y)
clf.score(X, y)
yp = clf.predict(X)
accuracy_score(y, yp)



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm

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




# Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone # Trash Zone

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

model = Sequential()
model.add(Embedding(100, 9, input_length=1))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()




out = Activation('sigmoid', name='strong_out')(x)
#audio_context = Model(inputs=x_start, outputs=out)
#audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
#audio_context.summary()
# I don't think we need y

model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=21, output_dim=5))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(21))
model.add(layers.LSTM(10))

# Add a Dense layer with 10 units.
model.add(layers.Dense(4))

model.summary()


model.fit(input_data, target_data, batch_size=batch_size)
