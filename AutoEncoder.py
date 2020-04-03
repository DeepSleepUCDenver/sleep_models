import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib




#data = pd.read_feather("Staging_Data_GT_190315.ftr")
#
#
#eeg = data[['eeg']].to_numpy().transpose()
#eog = data[['eog']].to_numpy().transpose()
#emg = data[['emg']].to_numpy().transpose()
#
#data = np.vstack((eeg,eog,emg))
data = np.load('reshap_to_npy/all_labeled_data_X.npy')

window_length = data.shape[1]


#TODO: Normalize Data

#Encoder
input_window = Input(shape=(window_length,3))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # Full Dimension
x = BatchNormalization()(x)
x = MaxPooling1D(3, padding="same")(x)
x = Conv1D(1, 3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims... I'm not super convinced this is actually 3 dimensions

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # Latent space
x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 6 dims
x = Conv1D(16, 3, activation='relu', padding='same')(x) # 5 dims
x = BatchNormalization()(x)
x = UpSampling1D(3)(x) # 10 dims
decoded = Conv1D(3, 3, activation='sigmoid', padding='same')(x) # 10 dims
autoencoder = Model(input_window, decoded)
autoencoder.summary()


x_train = data
epochs = 2



autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_train, x_train))


#decoded_stocks = autoencoder.predict(x_test)


#Source
#https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e
#Look there for some good graphing code!
