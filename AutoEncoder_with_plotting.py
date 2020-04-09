import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt




#data = pd.read_feather("Staging_Data_GT_190315.ftr")
#
#
#eeg = data[['eeg']].to_numpy().transpose()
#eog = data[['eog']].to_numpy().transpose()
#emg = data[['emg']].to_numpy().transpose()
#
#data = np.vstack((eeg,eog,emg))
data = np.load('all_labeled_data_X.npy')

data_slice = data[0:1][:][:]
print(data.shape)
print(data_slice.shape)



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
epochs = 100



autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_train, x_train))


#decoded_stocks = autoencoder.predict(x_test)


def plot_latent_space(raw_input, encoder, name):
    latent_space = encoder.predict(raw_input)
    flat_latent_space = latent_space.flatten()
    X = []
    for i in range(len(flat_latent_space)):
        X.append(i)
    plt.clf()
    plt.plot(X,flat_latent_space)
    plt.savefig(name+'.png')

data_slice = data[0:1][:][:]

for i in range(1,200,30):
    data_to_plot =  data[i:i+1][:][:]
    plot_latent_space(data_to_plot, encoder, str(i))

encoder.save("Encoder_100.h5")

#Source
#https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e
#Look there for some good graphing code!

