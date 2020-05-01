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
from balance_sample_data import load_all_data 
from balance_sample_data import load_known_data 

model_name = []
test_accuracy = []
train_accuracy = []
validation_accuracy = []
label_prop = []

x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()
x_tr.shape
x_te.shape
x_va.shape


from imblearn.ensemble import BalancedRandomForestClassifier

# model_name.append("Balanced Random Forest")
# label_prop.append("Label Propagation")
# rfb = BalancedRandomForestClassifier()
# rfb.fit(x_tr, y_tr)
# train_accuracy.append(     rfb.score(x_tr, y_tr))
# test_accuracy.append(      rfb.score(x_te, y_te))
# validation_accuracy.append(rfb.score(x_va, y_va))
# 
# model_name.append("Balanced Random Forest 7")
# label_prop.append("Label Propagation")
# rfb = BalancedRandomForestClassifier(max_depth=7)
# rfb.fit(x_tr, y_tr)
# train_accuracy.append(     rfb.score(x_tr, y_tr))
# test_accuracy.append(      rfb.score(x_te, y_te))
# validation_accuracy.append(rfb.score(x_va, y_va))

model_name.append("Balanced Random Forest")
label_prop.append("Label Propagation")
rfb = BalancedRandomForestClassifier(max_depth=2)
rfb.fit(x_tr, y_tr)
train_accuracy.append(     rfb.score(x_tr, y_tr))
test_accuracy.append(      rfb.score(x_te, y_te))
validation_accuracy.append(rfb.score(x_va, y_va))


from imblearn.ensemble import RUSBoostClassifier

model_name.append("RUS Boost")
label_prop.append("Label Propagation")
clf = RUSBoostClassifier(random_state=0)
clf.fit(x_tr, y_tr)
train_accuracy.append(     clf.score(x_tr, y_tr))
test_accuracy.append(      clf.score(x_te, y_te))
validation_accuracy.append(clf.score(x_va, y_va))

x_tr, y_tr, x_te, y_te, x_va, y_va = load_known_data()

from imblearn.ensemble import BalancedRandomForestClassifier

# model_name.append("Balanced Random Forest")
# label_prop.append("Label Propagation")
# rfb = BalancedRandomForestClassifier()
# rfb.fit(x_tr, y_tr)
# train_accuracy.append(     rfb.score(x_tr, y_tr))
# test_accuracy.append(      rfb.score(x_te, y_te))
# validation_accuracy.append(rfb.score(x_va, y_va))
# 
# model_name.append("Balanced Random Forest 7")
# label_prop.append("Label Propagation")
# rfb = BalancedRandomForestClassifier(max_depth=7)
# rfb.fit(x_tr, y_tr)
# train_accuracy.append(     rfb.score(x_tr, y_tr))
# test_accuracy.append(      rfb.score(x_te, y_te))
# validation_accuracy.append(rfb.score(x_va, y_va))

model_name.append("Balanced Random Forest")
label_prop.append("Label Propagation")
rfb = BalancedRandomForestClassifier(max_depth=2)
rfb.fit(x_tr, y_tr)
train_accuracy.append(     rfb.score(x_tr, y_tr))
test_accuracy.append(      rfb.score(x_te, y_te))
validation_accuracy.append(rfb.score(x_va, y_va))


from imblearn.ensemble import RUSBoostClassifier

model_name.append("RUS Boost")
label_prop.append("Label Propagation")
clf = RUSBoostClassifier(random_state=0)
clf.fit(x_tr, y_tr)
train_accuracy.append(     clf.score(x_tr, y_tr))
test_accuracy.append(      clf.score(x_te, y_te))
validation_accuracy.append(clf.score(x_va, y_va))

results = pd.DataFrame({
    'Model': model_name,
    'Label Propagation': label_prop,
    'Test Accuracy': test_accuracy,
    'Train Accuracy': train_accuracy,
    'Validation Accuracy': validation_accuracy
})
results.T
results.to_csv('./balance_sample_results.csv')


