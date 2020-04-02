import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelSpreading
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from imblearn.over_sampling import SMOTE
from oversample import load_all_data 


x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()

from sklearn import svm
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(x_tr, y_tr)
svm_linear.score(x_te, y_te)
svm_linear.score(x_va, y_va)

svm_poly = svm.SVC(kernel='poly')
svm_poly.fit(x_tr, y_tr)
svm_poly.score(x_te, y_te)
svm_poly.score(x_va, y_va)

svm_rbf  = svm.SVC(kernel='rbf')
svm_rbf.fit(x_tr, y_tr)
svm_rbf.score(x_te, y_te)
svm_rbf.score(x_va, y_va)

svm_sigmoid = svm.SVC(kernel='sigmoid')
svm_sigmoid.fit(x_tr, y_tr)
svm_sigmoid.score(x_te, y_te)
svm_sigmoid.score(x_va, y_va)


from sklearn.ensemble import RandomForestClassifier

rdf = RandomForestClassifier(max_depth=4, random_state=0)
rdf.fit(x_tr, y_tr)
rdf.score(x_te, y_te)
rdf.score(x_va, y_va)

from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier(random_state=0)
adb.fit(x_tr, y_tr)
adb.score(x_te, y_te)
adb.score(x_va, y_va)

