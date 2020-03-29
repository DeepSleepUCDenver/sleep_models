# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import average_downsampling_sliding_window

data = pd.read_feather("Staging_Data_GT_190315.ftr")



for i in range(1,15):
    down_sample_data = average_downsampling_sliding_window.down_sample(i, data)

    test = down_sample_data[0]
    eeg = down_sample_data[0][['eeg']][:1000]
    eog = down_sample_data[0][['eog']][:1000]
    emg = down_sample_data[0][['emg']][:1000]
    eeg.columns = ['measure']
    eog.columns = ['measure']
    emg.columns = ['measure']
    eeg['wave'] = 'eeg'
    eog['wave'] = 'eog'
    emg['wave'] = 'emg'

    data_joined_2 = pd.concat([eeg, eog, emg], axis=0)
    data_joined_2['index'] = data_joined_2.index
    plt.figure()
    plot = sea.lineplot(y='measure', x='index', hue='wave', data=data_joined_2)
    print(data_joined_2.shape)
    plot.figure.savefig("waveplot_downsample" + str(i) + ".png")
