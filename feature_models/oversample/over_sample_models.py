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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from over_sample_data import load_all_data 
from over_sample_data import load_known_data 
from over_sample_data import load_psdo_label_data
from over_sample_data import load_iter_psdo_label_data

model_name = []
test_accuracy = []
train_accuracy = []
validation_accuracy = []
label_prop = []


model_name.append("Random Forest")
label_prop.append("Pseudo Iterative")
rdf = RandomForestClassifier(max_depth=4, random_state=0)
x_tr, y_tr, x_te, y_te, x_va, y_va = load_iter_psdo_label_data(rdf, 50)
rdf.fit(x_tr, y_tr)
train_accuracy.append(     rdf.score(x_tr, y_tr))
test_accuracy.append(      rdf.score(x_te, y_te))
validation_accuracy.append(rdf.score(x_va, y_va))

model_name.append("Naive Bayes")
label_prop.append("Pseudo Iterative")
bay = GaussianNB()
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(bay)
bay.fit(x_tr, y_tr)
train_accuracy.append(     bay.score(x_tr, y_tr))
test_accuracy.append(      bay.score(x_te, y_te))
validation_accuracy.append(bay.score(x_va, y_va))



results = pd.DataFrame({
    'Model': model_name,
    'Label Propagation': label_prop,
    'Test Accuracy': test_accuracy,
    'Train Accuracy': train_accuracy,
    'Validation Accuracy': validation_accuracy
})
old_results = pd.read_csv('./over_sample_results.csv')
pd.merge(old_results, results, on=['Model', 'Label Propagation'], how='outer', validate="one_to_one")


x_tr, y_tr, x_te, y_te, x_va, y_va = load_all_data()
x_tr.shape
x_te.shape
x_va.shape


model_name.append("SVM Linear Kernel")
label_prop.append("Label Propagation")
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(x_tr, y_tr)
train_accuracy.append(     svm_linear.score(x_tr, y_tr))
test_accuracy.append(      svm_linear.score(x_te, y_te))
validation_accuracy.append(svm_linear.score(x_va, y_va))

model_name.append("SVM Poly Kernel")
label_prop.append("Label Propagation")
svm_poly = svm.SVC(kernel='poly')
svm_poly.fit(x_tr, y_tr)
train_accuracy.append(     svm_poly.score(x_tr, y_tr))
test_accuracy.append(      svm_poly.score(x_te, y_te))
validation_accuracy.append(svm_poly.score(x_va, y_va))

model_name.append("SVM RBF Kernel")
svm_rbf  = svm.SVC(kernel='rbf')
label_prop.append("Label Propagation")
svm_rbf.fit(x_tr, y_tr)
train_accuracy.append(     svm_rbf.score(x_tr, y_tr))
test_accuracy.append(      svm_rbf.score(x_te, y_te))
validation_accuracy.append(svm_rbf.score(x_va, y_va))

model_name.append("SVM sigmoid Kernel")
label_prop.append("Label Propagation")
svm_sigmoid = svm.SVC(kernel='sigmoid')
svm_sigmoid.fit(x_tr, y_tr)
train_accuracy.append(     svm_sigmoid.score(x_tr, y_tr))
test_accuracy.append(      svm_sigmoid.score(x_te, y_te))
validation_accuracy.append(svm_sigmoid.score(x_va, y_va))



model_name.append("Random Forest")
label_prop.append("Label Propagation")
rdf = RandomForestClassifier(max_depth=4, random_state=0)
rdf.fit(x_tr, y_tr)
train_accuracy.append(     rdf.score(x_tr, y_tr))
test_accuracy.append(      rdf.score(x_te, y_te))
validation_accuracy.append(rdf.score(x_va, y_va))


model_name.append("Ada-Boost")
label_prop.append("Label Propagation")
adb = AdaBoostClassifier(random_state=0)
adb.fit(x_tr, y_tr)
train_accuracy.append(     adb.score(x_tr, y_tr))
test_accuracy.append(      adb.score(x_te, y_te))
validation_accuracy.append(adb.score(x_va, y_va))

# this one takes to long
# model_name.append("Ada-Boost SVM")
# label_prop.append("Label Propagation")
# adbs = AdaBoostClassifier(base_estimator=svm.SVC(probability=True, kernel='rbf'), random_state=0)
# adbs.fit(x_tr, y_tr)
# test_accuracy.append(      adb.score(x_te, y_te))
# validation_accuracy.append(adb.score(x_va, y_va))
# validation_accuracy.append(adbs.score(x_va, y_va))


model_name.append("KNN")
label_prop.append("Label Propagation")
knn = KNeighborsClassifier()
knn.fit(x_tr, y_tr)
train_accuracy.append(     knn.score(x_tr, y_tr))
test_accuracy.append(      knn.score(x_te, y_te))
validation_accuracy.append(knn.score(x_va, y_va))
#knn.predict(x_va)


model_name.append("Naive Bayes")
label_prop.append("Label Propagation")
bay = GaussianNB()
bay.fit(x_tr, y_tr)
train_accuracy.append(     bay.score(x_tr, y_tr))
test_accuracy.append(      bay.score(x_te, y_te))
validation_accuracy.append(bay.score(x_va, y_va))


results = pd.DataFrame({
    'Model': model_name,
    'Label Propagation': label_prop,
    'Test Accuracy': test_accuracy,
    'Train Accuracy': train_accuracy,
    'Validation Accuracy': validation_accuracy
})

results.to_csv('./over_sample_results.csv')

x_tr, y_tr, x_te, y_te, x_va, y_va = load_known_data()

model_name.append("SVM Linear Kernel")
label_prop.append("No Propagation")
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(x_tr, y_tr)
train_accuracy.append(     svm_linear.score(x_tr, y_tr))
test_accuracy.append(      svm_linear.score(x_te, y_te))
validation_accuracy.append(svm_linear.score(x_va, y_va))

model_name.append("SVM Poly Kernel")
label_prop.append("No Propagation")
svm_poly = svm.SVC(kernel='poly')
svm_poly.fit(x_tr, y_tr)
train_accuracy.append(     svm_poly.score(x_tr, y_tr))
test_accuracy.append(      svm_poly.score(x_te, y_te))
validation_accuracy.append(svm_poly.score(x_va, y_va))

model_name.append("SVM RBF Kernel")
svm_rbf  = svm.SVC(kernel='rbf')
label_prop.append("No Propagation")
svm_rbf.fit(x_tr, y_tr)
train_accuracy.append(     svm_rbf.score(x_tr, y_tr))
test_accuracy.append(      svm_rbf.score(x_te, y_te))
validation_accuracy.append(svm_rbf.score(x_va, y_va))

model_name.append("SVM sigmoid Kernel")
label_prop.append("No Propagation")
svm_sigmoid = svm.SVC(kernel='sigmoid')
svm_sigmoid.fit(x_tr, y_tr)
train_accuracy.append(     svm_sigmoid.score(x_tr, y_tr))
test_accuracy.append(      svm_sigmoid.score(x_te, y_te))
validation_accuracy.append(svm_sigmoid.score(x_va, y_va))



model_name.append("Random Forest")
label_prop.append("No Propagation")
rdf = RandomForestClassifier(max_depth=4, random_state=0)
rdf.fit(x_tr, y_tr)
train_accuracy.append(     rdf.score(x_tr, y_tr))
test_accuracy.append(      rdf.score(x_te, y_te))
validation_accuracy.append(rdf.score(x_va, y_va))


model_name.append("Ada-Boost")
label_prop.append("No Propagation")
adb = AdaBoostClassifier(random_state=0)
adb.fit(x_tr, y_tr)
train_accuracy.append(     adb.score(x_tr, y_tr))
test_accuracy.append(      adb.score(x_te, y_te))
validation_accuracy.append(adb.score(x_va, y_va))

# this one takes to long
# model_name.append("Ada-Boost SVM")
# label_prop.append("Label Propagation")
# adbs = AdaBoostClassifier(base_estimator=svm.SVC(probability=True, kernel='rbf'), random_state=0)
# adbs.fit(x_tr, y_tr)
# train_accuracy.append(     adbs.score(x_tr, y_tr))
# test_accuracy.append(      adbs.score(x_te, y_te))
# validation_accuracy.append(adbs.score(x_va, y_va))
# from sklearn.neighbors import KNeighborsClassifier

model_name.append("KNN")
label_prop.append("No Propagation")
knn = KNeighborsClassifier()
knn.fit(x_tr, y_tr)
train_accuracy.append(     knn.score(x_tr, y_tr))
test_accuracy.append(      knn.score(x_te, y_te))
validation_accuracy.append(knn.score(x_va, y_va))
#knn.predict(x_va)


model_name.append("Naive Bayes")
label_prop.append("No Propagation")
bay = GaussianNB()
bay.fit(x_tr, y_tr)
train_accuracy.append(     bay.score(x_tr, y_tr))
test_accuracy.append(      bay.score(x_te, y_te))
validation_accuracy.append(bay.score(x_va, y_va))


results = pd.DataFrame({
    'Model': model_name,
    'Label Propagation': label_prop,
    'Test Accuracy': test_accuracy,
    'Train Accuracy': train_accuracy,
    'Validation Accuracy': validation_accuracy
})
results.to_csv('./over_sample_results.csv')
# d = pd.read_csv('./over_sample_results.csv')


model_name.append("SVM Linear Kernel")
label_prop.append("Pseudo Labels")
svm_linear = svm.SVC(kernel='linear')
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(svm_linear)
svm_linear.fit(x_tr, y_tr)
train_accuracy.append(     svm_linear.score(x_tr, y_tr))
test_accuracy.append(      svm_linear.score(x_te, y_te))
validation_accuracy.append(svm_linear.score(x_va, y_va))

model_name.append("SVM Poly Kernel")
label_prop.append("Pseudo Labels")
svm_poly = svm.SVC(kernel='poly')
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(svm_poly)
svm_poly.fit(x_tr, y_tr)
train_accuracy.append(     svm_poly.score(x_tr, y_tr))
test_accuracy.append(      svm_poly.score(x_te, y_te))
validation_accuracy.append(svm_poly.score(x_va, y_va))

model_name.append("SVM RBF Kernel")
label_prop.append("Pseudo Labels")
svm_rbf  = svm.SVC(kernel='rbf')
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(svm_rbf)
svm_rbf.fit(x_tr, y_tr)
train_accuracy.append(     svm_rbf.score(x_tr, y_tr))
test_accuracy.append(      svm_rbf.score(x_te, y_te))
validation_accuracy.append(svm_rbf.score(x_va, y_va))

model_name.append("SVM sigmoid Kernel")
label_prop.append("Pseudo Labels")
svm_sigmoid = svm.SVC(kernel='sigmoid')
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(svm_sigmoid)
svm_sigmoid.fit(x_tr, y_tr)
train_accuracy.append(     svm_sigmoid.score(x_tr, y_tr))
test_accuracy.append(      svm_sigmoid.score(x_te, y_te))
validation_accuracy.append(svm_sigmoid.score(x_va, y_va))



model_name.append("Random Forest")
label_prop.append("Pseudo Labels")
rdf = RandomForestClassifier(max_depth=4, random_state=0)
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(rdf)
rdf.fit(x_tr, y_tr)
train_accuracy.append(     rdf.score(x_tr, y_tr))
test_accuracy.append(      rdf.score(x_te, y_te))
validation_accuracy.append(rdf.score(x_va, y_va))


model_name.append("Ada-Boost")
label_prop.append("Pseudo Labels")
adb = AdaBoostClassifier(random_state=0)
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(adb)
adb.fit(x_tr, y_tr)
train_accuracy.append(     adb.score(x_tr, y_tr))
test_accuracy.append(      adb.score(x_te, y_te))
validation_accuracy.append(adb.score(x_va, y_va))

# this one takes to long
# model_name.append("Ada-Boost SVM")
# label_prop.append("Label Propagation")
# adbs = AdaBoostClassifier(base_estimator=svm.SVC(probability=True, kernel='rbf'), random_state=0)
# adbs.fit(x_tr, y_tr)
# train_accuracy.append(     adbs.score(x_tr, y_tr))
# test_accuracy.append(      adbs.score(x_te, y_te))
# validation_accuracy.append(adbs.score(x_va, y_va))
# from sklearn.neighbors import KNeighborsClassifier

model_name.append("KNN")
label_prop.append("Pseudo Labels")
knn = KNeighborsClassifier()
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(knn)
knn.fit(x_tr, y_tr)
train_accuracy.append(     knn.score(x_tr, y_tr))
test_accuracy.append(      knn.score(x_te, y_te))
validation_accuracy.append(knn.score(x_va, y_va))
#knn.predict(x_va)


model_name.append("Naive Bayes")
label_prop.append("Pseudo Labels")
bay = GaussianNB()
x_tr, y_tr, x_te, y_te, x_va, y_va = load_psdo_label_data(bay)
bay.fit(x_tr, y_tr)
train_accuracy.append(     bay.score(x_tr, y_tr))
test_accuracy.append(      bay.score(x_te, y_te))
validation_accuracy.append(bay.score(x_va, y_va))


results = pd.DataFrame({
    'Model': model_name,
    'Label Propagation': label_prop,
    'Test Accuracy': test_accuracy,
    'Train Accuracy': train_accuracy,
    'Validation Accuracy': validation_accuracy
})
results.to_csv('./over_sample_results.csv')

