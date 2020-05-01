# Import the scripts
import pandas as pd
import numpy as np
from mat_read import read_raw, read_feature
import gc
path_to_mat = '../raw_data'

files = ["190315"
       , "190427"
       , "190617"
       , "190624"
       , "190710"
       , "190716"
       , "190723"
       , "190724"
       , "190730"
       , "191001"
       , "191008"
       , "191009"
       , "191108"
       , "191209"
       , "191211"
       , "191213"
       , "191217"
       , "191218"
       , "200128"
       , "200129"]

print('The order of the data will be emg, eog, eeg')
for i, f in enumerate(files):
    print("Apending: ", f)
    data = read_raw(path_to_mat + '/Staging_Data_GT_' + f + ".mat", )
    x = np.zeros([data.shape[0], len(data.emg[0]), 3], dtype= 'float32')
    y = data.stage.values
    for j, r in data.iterrows():
        x[j, :, 0] = r.emg
        x[j, :, 1] = r.eog
        x[j, :, 2] = r.eeg
    del(data)
    gc.collect()
    np.save('all_labeled_data_X_' + str(i) + '.npy', x)
    np.save('all_labeled_data_y_' + str(i) + '.npy', y)

print('You can now run the combo script!')
