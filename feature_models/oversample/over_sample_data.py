import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
# from sklearn.semi_supervised import label_propagation
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier
from PseudoLabel.PseudoLabel import PseudoLabeler
from sklearn.neighbors import KNeighborsClassifier


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
    
    # Over sample the stages
    zen = SMOTE(random_state=8675309)
    x_obs_os, y_obs_os = zen.fit_resample(x_obs, y_obs)

    # apply Label Spreading
    label_spread = LabelPropagation(kernel='knn')
    label_spread.fit(x_obs_os, y_obs_os)
    x_all = np.concatenate([x_obs, x_nuls], axis=0)
    y_all = np.concatenate([y_obs, label_spread.predict(x_nuls)], axis=0)
    
    # Over sample the stages
    zen = SMOTE(random_state=8675309)
    x, y = zen.fit_resample(x_all, y_all)
    x, y = shuffle(x, y, random_state=42)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
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
    
    # Over sample the stages
    zen = SMOTE(random_state=8675309)
    x, y = zen.fit_resample(x_all, y_all)
    x, y = shuffle(x, y, random_state=42)
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
    
    # Over sample the stages for labeling
    zen = SMOTE(random_state=8675309)
    x_obs_os, y_obs_os = zen.fit_resample(x_obs, y_obs)

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
    
    # Over sample the stages
    zen = SMOTE(random_state=8675309)
    x, y = zen.fit_resample(x_all, y_all)
    x, y = shuffle(x, y, random_state=42)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20)
    return x_tr, y_tr, x_te, y_te, x_va, y_va





#  x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()
#  
#  
#  #x_tr, y_tr, x_te
#  
#  #Random Forest Example.
#  #x_tr and y_tr are the labeled data
#  #x_te is unlabaled data
#  
#  #Random Forest Example
#  #---------------------
#  model = PseudoLabeler(
#      RandomForestClassifier(n_estimators=2),
#      x_tr,
#      y_tr,
#      x_te
#  )
#  model.fit(x_tr, y_tr)
#  print(model.predict(x_te))
#  #---------------------
#  
#  #KNN Classifier
#  model1 = PseudoLabeler(
#      KNeighborsClassifier(),
#      x_tr,
#      y_tr,
#      x_te
#  )
#  model1.fit(x_tr, y_tr)
#  print(model1.predict(x_te))
#  

