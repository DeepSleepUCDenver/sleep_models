import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt


# Normalize, data
data = np.load('./reshap_to_npy/all_labeled_data_X.npy')
for i in range(3):
    print(i)
    mean = np.mean(data[:,:,i])
    sdev = np.std(data[:,:,i])
    data[:,:,i] = (data[:,:,i] - mean) / sdev

# # minmax, data
# data = np.load('./reshap_to_npy/all_labeled_data_X.npy')
# for i in range(3):
#     print(i)
#     minx = np.min(data[:,:,i])
#     maxx = np.max(data[:,:,i])
#     data[:,:,i] = (data[:,:,i] - minx) / (maxx - minx)
# 
# OH this was bad!

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
epochs = 25
epochs = 2



autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=(256*4),
                shuffle=True,
                validation_data=(x_train, x_train))



# rm-plot start 

import seaborn as sea
import matplotlib.pyplot as plt

# Real data
emg = pd.DataFrame(x_train[0,:1400,0])
eog = pd.DataFrame(x_train[0,:1400,1])
eeg = pd.DataFrame(x_train[0,:1400,2])
eeg.columns = ['measure']
eog.columns = ['measure'] 
emg.columns = ['measure']
eeg['wave'] = 'eeg'
eog['wave'] = 'eog'
emg['wave'] = 'emg'
data_joined = pd.concat([eeg, eog, emg], axis=0)
#data_joined['index'] = data_joined.index
real = data_joined

# simulated data
sim = autoencoder.predict(x_train[0:1,:,:])
sim.shape
emg = pd.DataFrame(sim[0,:1400,0])
eog = pd.DataFrame(sim[0,:1400,1])
eeg = pd.DataFrame(sim[0,:1400,2])
eeg.columns = ['measure']
eog.columns = ['measure'] 
emg.columns = ['measure']
eeg['wave'] = 'sim_eeg'
eog['wave'] = 'sim_eog'
emg['wave'] = 'sim_emg'
data_joined = pd.concat([eeg, eog, emg], axis=0)
#data_joined['index'] = data_joined.index
sim = data_joined

pdta = pd.concat([real, sim], axis=0)
pdta['index'] = pdta.index

plt.clf()
plot = sea.lineplot(
        y='measure', 
        x='index', 
        hue='wave', 
        data=pdta)


plot.lines[3].set_linestyle("--")
plot.lines[4].set_linestyle("--")
plot.lines[5].set_linestyle("--")
plot.figure.savefig("compare_plot.png")

# rm-plot end 
#Source
#https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e
#Look there for some good graphing code!

