import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
# from sklearn.semi_supervised import label_propagation
from imblearn.over_sampling import SMOTE



def load_all_data():
    # Read am partition the matrix
    data = pd.read_feather('./feature_stage_data_all.ftr')
    x = data[data.columns[3:]]
    y = data['stage']
    o = data.observation
    x = x.values
    x = normalize(x)
    y = y.values
    x_va = x[[i in [8, 9] for i in o.values]]
    y_va = y[[i in [8, 9] for i in o.values]]
    o.unique()
    
    
    nnl = lambda a: np.invert(np.isnan(a))
    nul = lambda a: np.isnan(a)
    x_obs = x[nnl(y)]
    y_obs = y[nnl(y)]
    
    # apply Label Spreading
    x_nuls = x[nul(y)]
    label_spread = LabelPropagation(kernel='knn')
    label_spread.fit(x_obs, y_obs)
    x_all = np.concatenate([x_obs, x_nuls], axis=0)
    y_all = np.concatenate([y_obs, label_spread.predict(x_nuls)], axis=0)
    
    # Over sample the stages
    zen = SMOTE(random_state=8675309)
    x, y = zen.fit_resample(x_all, y_all)
    x, y = shuffle(x, y, random_state=42)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
    return x_tr, y_tr, x_te, y_te, x_va, y_va


load_all_data()
