import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import itr
# from sklearn.semi_supervised import label_propagation
from imblearn.over_sampling import SMOTE



def load_all_data():
    # Read am partition the matrix
    data = pd.read_feather('../feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[[i in [8, 9] for i in o.values]]
    y_va = y[[i in [8, 9] for i in o.values]]
    x = x[[i not in [8, 9] for i in o.values]]
    y = y[[i not in [8, 9] for i in o.values]]
    o.unique()
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_obs = x[nnl(y)]
    y_obs = y[nnl(y)]
    x_nuls = x[nul(y)]
    
    # Undersample the stages
    x_obs, y_obs = shuffle(x_obs, y_obs, random_state=42)
    smpnum = min([sum(y_obs==i) for i in range(1,6)])
    y_obs_us = y[y == 1][:smpnum]
    x_obs_us = x[y == 1][:smpnum]
    for i in range(2,6):
        x_obs_us = np.concatenate([x_obs_us, x[y == i][:smpnum]])
        y_obs_us = np.concatenate([y_obs_us, y[y == i][:smpnum]])

    # apply Label Spreading
    label_spread = LabelPropagation(kernel='knn')
    label_spread.fit(x_obs_us, y_obs_v)
    x_all = np.concatenate([x_obs, x_nuls], axis=0)
    y_all = np.concatenate([y_obs, label_spread.predict(x_nuls)], axis=0)
    
    # Undersample the stages
    x, y = shuffle(x, y, random_state=42)
    smpnum = min([sum(y==i) for i in range(1,6)])
    y_btr = y[y == 1][:smpnum]
    x_btr = x[y == 1][:smpnum]
    for i in range(2,6):
        x_btr = np.concatenate([x_btr, x[y == i][:smpnum]])
        y_btr = np.concatenate([y_btr, y[y == i][:smpnum]])
    x_tr, x_te, y_tr, y_te = train_test_split(x_btr, y_btr, test_size = 0.20)
    # Over sample the stages
    #zen = SMOTE(random_state=8675309)
    #x, y = zen.fit_resample(x_all, y_all)
    #x, y = shuffle(x, y, random_state=42)
    #x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
    return x_tr, y_tr, x_te, y_te, x_va, y_va


def load_known_data():
    # Read am partition the matrix
    data = pd.read_feather('../feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[[i in [8, 9] for i in o.values]]
    y_va = y[[i in [8, 9] for i in o.values]]
    x = x[[i not in [8, 9] for i in o.values]]
    y = y[[i not in [8, 9] for i in o.values]]
    o.unique()
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_all = x[nnl(y)]
    y_all = y[nnl(y)]
    
    ## apply Label Spreading
    #x_nuls = x[nul(y)]
    #label_spread = LabelPropagation(kernel='knn')
    #label_spread.fit(x_obs, y_obs)
    #x_all = np.concatenate([x_obs, x_nuls], axis=0)
    #y_all = np.concatenate([y_obs, label_spread.predict(x_nuls)], axis=0)
    
    # Undersample the stages
    x, y = shuffle(x, y, random_state=42)
    smpnum = min([sum(y==i) for i in range(1,6)])
    y_btr = y[y == 1][:smpnum]
    x_btr = x[y == 1][:smpnum]
    for i in range(2,6):
        x_btr = np.concatenate([x_btr, x[y == i][:smpnum]])
        y_btr = np.concatenate([y_btr, y[y == i][:smpnum]])
    x_tr, x_te, y_tr, y_te = train_test_split(x_btr, y_btr, test_size = 0.20)
    # Over sample the stages
    #zen = SMOTE(random_state=8675309)
    #x, y = zen.fit_resample(x_all, y_all)
    #x, y = shuffle(x, y, random_state=42)
    #x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
    return x_tr, y_tr, x_te, y_te, x_va, y_va


def load_psdo_label_data(algo):
    # Read am partition the matrix
    data = pd.read_feather('../feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[[i in [8, 9] for i in o.values]]
    y_va = y[[i in [8, 9] for i in o.values]]
    x = x[[i not in [8, 9] for i in o.values]]
    y = y[[i not in [8, 9] for i in o.values]]
    o.unique()
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_obs = x[nnl(y)]
    y_obs = y[nnl(y)]
    x_nuls = x[nul(y)]
    
    # Undersample the stages
    x_obs, y_obs = shuffle(x_obs, y_obs, random_state=42)
    smpnum = min([sum(y_obs==i) for i in range(1,6)])
    y_obs_us = y[y == 1][:smpnum]
    x_obs_us = x[y == 1][:smpnum]
    for i in range(2,6):
        x_obs_us = np.concatenate([x_obs_us, x[y == i][:smpnum]])
        y_obs_us = np.concatenate([y_obs_us, y[y == i][:smpnum]])

    # apply Pseudo Label Spreading
    model = PseudoLabeler(
        algo,
        x_obs_os,
        y_obs_os,
        x_nuls
    )
    model.fit(x_obs_os, y_obs_os)
    x_all = np.concatenate([x_obs, x_nuls], axis=0)
    y_nuls = model.predict(x_nuls)
    y_all = np.concatenate([y_obs, y_nuls], axis=0)
    
    # Undersample the stages
    x, y = shuffle(x_all, y_all, random_state=42)
    smpnum = min([sum(y==i) for i in range(1,6)])
    y_btr = y[y == 1][:smpnum]
    x_btr = x[y == 1][:smpnum]
    for i in range(2,6):
        x_btr = np.concatenate([x_btr, x[y == i][:smpnum]])
        y_btr = np.concatenate([y_btr, y[y == i][:smpnum]])
    x_tr, x_te, y_tr, y_te = train_test_split(x_btr, y_btr, test_size = 0.20)
    
    return x_tr, y_tr, x_te, y_te, x_va, y_va


def load_iter_psdo_label_data(algo):
    # Read am partition the matrix
    data = pd.read_feather('../feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[[i in [8, 9] for i in o.values]]
    y_va = y[[i in [8, 9] for i in o.values]]
    x = x[[i not in [8, 9] for i in o.values]]
    y = y[[i not in [8, 9] for i in o.values]]
    o.unique()
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_obs = x[nnl(y)]
    y_obs = y[nnl(y)]
    x_nuls = x[nul(y)]
    
    # Undersample the stages
    x_obs, y_obs = shuffle(x_obs, y_obs, random_state=42)
    smpnum = min([sum(y_obs==i) for i in range(1,6)])
    y_obs_us = y[y == 1][:smpnum]
    x_obs_us = x[y == 1][:smpnum]
    for i in range(2,6):
        x_obs_us = np.concatenate([x_obs_us, x[y == i][:smpnum]])
        y_obs_us = np.concatenate([y_obs_us, y[y == i][:smpnum]])

    # apply Pseudo Label Spreading
    model = PseudoLabeler(
        algo,
        x_obs_os,
        y_obs_os,
        x_nuls
    )
    model.fit(x_obs_os, y_obs_os)
    x_all = np.concatenate([x_obs, x_nuls], axis=0)
    y_nuls = model.predict(x_nuls)
    y_all = np.concatenate([y_obs, y_nuls], axis=0)
    
    # Undersample the stages
    x, y = shuffle(x_all, y_all, random_state=42)
    smpnum = min([sum(y==i) for i in range(1,6)])
    y_btr = y[y == 1][:smpnum]
    x_btr = x[y == 1][:smpnum]
    for i in range(2,6):
        x_btr = np.concatenate([x_btr, x[y == i][:smpnum]])
        y_btr = np.concatenate([y_btr, y[y == i][:smpnum]])
    x_tr, x_te, y_tr, y_te = train_test_split(x_btr, y_btr, test_size = 0.20)
    
    return x_tr, y_tr, x_te, y_te, x_va, y_va



