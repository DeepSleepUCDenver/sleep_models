from scipy.io import loadmat
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# We set the log-level to 'WARNING' so the output is less verbose
mne.set_log_level('WARNING')


def read_raw(mat_path, unlist=False):
    mat = loadmat(mat_path)
    cols = ['id', 'raw', 'eeg','eog', 'emg', 'stage'] 
    data = pd.DataFrame()
    for i, epox in enumerate(mat['epochs']):
        epo = epox[0][0][0]
        size = len(epo[1])
        df2 = pd.DataFrame()
        m = 0
        if(unlist):
            for co, ar in zip(cols, epo):
                l = len([v.item() for v in ar])
                m = max(l,m)
            for co, ar in zip(cols, epo):
                tmp = [v.item() for v in ar]
                df1 = pd.DataFrame()
                if len(tmp) == 1:
                    df1[co] = [tmp[0] for i in range(m)]
                else:
                    df1[co] = tmp
                df2 = pd.concat([df1,df2], axis=1)
        else:
            for co, ar in zip(cols, epo):
                tmp = [v.item() for v in ar]
                if len(tmp) == 1:
                    df1 = pd.DataFrame({co: tmp})
                else:
                    df1 = pd.DataFrame({co: [tmp]})
                df2 = pd.concat([df1,df2], axis=1)
                # print(df2)
        if data.empty:
            data = df2
        else:
            data = pd.concat([data, df2], axis=0)
    data = data.reset_index()
    return(data)


def read_feature(mat_path):
    mat = loadmat(mat_path)
    features = []
    for i, f in enumerate(mat['feature']):
        features.append(f)
    cols = [str(x) for x in range(len(f))]
    data = pd.DataFrame(features, columns=cols)
    return data

