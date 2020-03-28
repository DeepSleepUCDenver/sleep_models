# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_feather("Staging_Data_GT_190315.ftr")


eeg = data[['eeg']]
eog = data[['eog']]
emg = data[['emg']]

'''
window_size - The number of elements to average. This is a non-overlapping window. 
data - data straight from a feather file. 

The following are the 3 channels (in the order they are stacked, with the first element listed is on the top.:
eeg, eog, emg
'''
def down_sample(window_size, data):
    data_length = len(data[['eog']])
    eeg = data[['eeg']]
    #blah = eeg.rolling(5).mean()
    index = pd.date_range('1/1/2000', periods=data_length, freq='T') #You have to add timestamps in order to use pandas resampling method
    final_resampled_data = []

    for item in ["eeg","eog", "emg"]:
        list_eeg = data[item].values.tolist()
        series = pd.Series(list_eeg, index=index)
        window_size = str(window_size)
        series = series.resample(window_size+'T', label='right').mean()
        array_series = series.to_numpy()
        final_resampled_data.append(array_series)


    return np.vstack((final_resampled_data[0],final_resampled_data[1],final_resampled_data[2]))


meh =  down_sample(5, data)

