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

for i, f in enumerate(files):
    print("Apending: ", f)
    data = read_raw(path_to_mat + '/Staging_Data_GT_' + f + ".mat", )
    data.columns
    x = np.zeros([data.shape[0], len(data.raw[0])], dtype= 'float32')
    #x.shape
    y = data.stage.values
    for j, r in data.iterrows():
        x[j, :] = r.emg
    del(data)
    gc.collect()
    np.save('all_raw_data_X_' + str(i) + '.npy', x)
    np.save('all_raw_data_y_' + str(i) + '.npy', y)
    #print('The order of the data will be emg, eog, eeg')

print('You can now run the combo script!')
