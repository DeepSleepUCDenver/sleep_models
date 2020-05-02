# Import the scripts
import pandas as pd
import seaborn as sea
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

