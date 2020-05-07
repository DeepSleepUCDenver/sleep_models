# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

d1 = pd.read_csv('./oversample/over_sample_results_svm.csv')
shape_tmp = []
for i, j in zip(['Train Accuracy', 'Test Accuracy', 'Validation Accuracy'], ['a Train Accuracy', 'b Test Accuracy', 'c Validation Accuracy']):
    tmp = d1[['Model', 'Label Propagation']].copy()
    tmp['Partition'] = i
    tmp['sort'] = j
    tmp['Score'] = d1[i]
    shape_tmp.append(tmp)

d1 = pd.concat(shape_tmp)
#d1['Model'] = d1['Model'].apply(lambda x : x + " (over)")
d1['Partition'] = d1['Partition'].apply(lambda x : x + " (over)")

d2 = pd.read_csv('./undersample/under_sample_results_svm.csv')
shape_tmp = []
for i, j in zip(['Train Accuracy', 'Test Accuracy', 'Validation Accuracy'], ['a Train Accuracy', 'b Test Accuracy', 'c Validation Accuracy']):
    tmp = d2[['Model', 'Label Propagation']].copy()
    tmp['Partition'] = i
    tmp['sort'] = j
    tmp['Score'] = d2[i]
    shape_tmp.append(tmp)

d2 = pd.concat(shape_tmp)
#d2['Model'] = d2['Model'].apply(lambda x : x + " (under)")
d2['Partition'] = d2['Partition'].apply(lambda x : x + " (under)")

'Model'
#d1['Label Propagation'].unique()


data = pd.concat([d1, d2])
data['SSL'] = data['Label Propagation']
data['SSL'].unique()
data = data[data['SSL'] == 'No Propagation']
data.columns
data = data.sort_values(by = "sort")
data.reset_index(inplace = True, drop=True)
data.loc[1].T


#color='Blues',
plt.clf()
plt.figure(figsize=(15,5))
sea.set(style="whitegrid")
# set colors
sea.set_palette("Reds")
colo1 = sea.color_palette()[-3:]
sea.set_palette("Blues")
colo2 = sea.color_palette()[-3:]
colo = []
for c1, c2 in zip(colo1,colo2):
    colo.append(c1)
    colo.append(c2)

colo.reverse()
sea.set_palette(colo)
# Make plot
plot = sea.barplot(
    y='Score', 
    x='Model', 
    hue="Partition", 
    data=data,
    ci=None,
    orient='v'
    )
plot.set_title("Model Performance, Undersampled VS Oversampled")
#change_width(plot, .35)
plt.ylim(0, .9)
plt.savefig("model_performance_under_vs_over.png")

