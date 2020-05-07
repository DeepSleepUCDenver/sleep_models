import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
# from sklearn.semi_supervised import label_propagation
from imblearn.over_sampling import SMOTE
from PseudoLabel.PseudoLabel import PseudoLabeler
from PseudoLabel.PseudoLabel import Iterative_PseudoLabeler




def load_known_data():
    # Read am partition the matrix
    data = pd.read_feather('../feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[4977:4977+3000]
    y_va = y[4977:4977+3000]
    x = np.concatenate((x[:4977],x[4977+3000:]))
    y = np.concatenate((y[:4977],y[4977+3000:]))
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)

    x = x[nnl(y)]
    y = y[nnl(y)]
    
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
    return x_tr, y_tr, x_te, y_te, x_va, y_va

def load_all_data():
    # Read am partition the matrix
    data = pd.read_feather('../feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[4977:4977+3000]
    y_va = y[4977:4977+3000]
    x = np.concatenate((x[:4977],x[4977+3000:]))
    y = np.concatenate((y[:4977],y[4977+3000:]))
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_obs = x[nnl(y)]
    y_obs = y[nnl(y)]
    
    # apply Label Spreading
    x_nuls = x[nul(y)]
    label_spread = LabelPropagation(kernel='knn')
    label_spread.fit(x_obs, y_obs)
    x = np.concatenate([x_obs, x_nuls], axis=0)
    y = np.concatenate([y_obs, label_spread.predict(x_nuls)], axis=0)
    
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
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
    x_va = x[4977:4977+3000]
    y_va = y[4977:4977+3000]
    x = np.concatenate((x[:4977],x[4977+3000:]))
    y = np.concatenate((y[:4977],y[4977+3000:]))
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_obs = x[nnl(y)]
    y_obs = y[nnl(y)]
    x_nuls = x[nul(y)]
    
    # apply Pseudo Label Spreading
    model = PseudoLabeler(
        algo,
        x_obs,
        y_obs,
        x_nuls
    )
    model.fit(x_obs, y_obs)
    x_all = np.concatenate([x_obs, x_nuls], axis=0)
    y_nuls = model.predict(x_nuls)
    y_all = np.concatenate([y_obs, y_nuls], axis=0)
    
    x, y = shuffle(x_all, y_all, random_state=42)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
    return x_tr, y_tr, x_te, y_te, x_va, y_va


def load_iter_psdo_label_data(algo, n_splits):
    # Read am partition the matrix
    data = pd.read_feather('../feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[4977:4977+3000]
    y_va = y[4977:4977+3000]
    x = np.concatenate((x[:4977],x[4977+3000:]))
    y = np.concatenate((y[:4977],y[4977+3000:]))
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_obs = x[nnl(y)]
    y_obs = y[nnl(y)]
    x_nuls = x[nul(y)]
    
    # apply Pseudo Label Spreading
    model = Iterative_PseudoLabeler(
        algo,
        x_obs,
        y_obs,
        x_nuls,
        n_splits
    )
    model.fit()
    x_all = np.concatenate([x_obs, x_nuls], axis=0)
    y_nuls = model.predict(x_nuls)
    y_all = np.concatenate([y_obs, y_nuls], axis=0)
    
    x, y = shuffle(x_all, y_all, random_state=42)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)

    return x_tr, y_tr, x_te, y_te, x_va, y_va
