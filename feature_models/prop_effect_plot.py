# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

#color='Blues',
plt.clf()
plt.figure(figsize=(20,5))
sea.set(style="whitegrid")
# set colors
colo = []
sea.set_palette("Greys")
colo += sea.color_palette()[-3:]
sea.set_palette("Blues")
colo += sea.color_palette()[-3:]
sea.set_palette("Greens")
colo += sea.color_palette()[-3:]
sea.set_palette("Oranges")
colo += sea.color_palette()[-3:]
sea.set_palette("Reds")
colo += sea.color_palette()[-3:]
colo.reverse()

d1 = pd.read_csv('undersample/under_sample_results.csv')

shape_tmp = []
for i, j in zip(['Train Accuracy', 'Test Accuracy', 'Validation Accuracy'], ['a Train Accuracy', 'b Test Accuracy', 'c Validation Accuracy']):
    tmp = d1[['Model', 'Label Propagation']].copy()
    tmp['Partition'] = i
    tmp['sort'] = j
    tmp['Score'] = d1[i]
    shape_tmp.append(tmp)

d1 = pd.concat(shape_tmp)

data = pd.concat([d1])
data['Label Propagation'].unique()
data.columns
data['Label Propagation'] = ['SKLearn Propagation' if x['Label Propagation'] == 'Label Propagation' else x['Label Propagation'] for _, x in data.iterrows()]
data['SSL'] = [x['Label Propagation'] + " " + x['Partition']  for _, x in data[['Label Propagation', 'Partition']].iterrows()]
data['sort'] = [x['Label Propagation'] + " " + x['sort']  for _, x in data.iterrows()]
data['sort'] = ['y' + x['sort'] if x['Label Propagation'] == 'SKLearn Propagation' else x['sort']  for _, x in data.iterrows()]
data['sort'] = ['z' + x['sort'] if x['Label Propagation'] == 'No Propagation' else x['sort']  for _, x in data.iterrows()]
data = data.sort_values(by = "sort")
data.reset_index(inplace = True, drop=True)
data.loc[1].T

maximum = data[data['Partition'] == 'Validation Accuracy']['Score'].max()
print(data[data['Partition'] == 'Validation Accuracy'].loc[maximum == data['Score']].T)


plt.clf()
plt.figure(figsize=(15,5))
sea.set(style="whitegrid")
sea.set_palette(colo)
# Make plot
plot = sea.barplot(
    y='Score', 
    x='Model', 
    hue="SSL", 
    data=data[[x['Model'] == 'KNN' or x['Model'] == 'Naive Bayes' for _, x in data.iterrows()]],
    ci=None,
    orient='v'
   )
# resize figure box to -> put the legend out of the figure
box = plot.get_position() # get position of figure
plot.set_position([box.x0, box.y0, box.width * 0.75, box.height]) # resize position
# Put a legend to the right side
plot.legend(loc='top right', bbox_to_anchor=(1, 1), ncol=1)
plot.set_title("Model Performance Across Forms of Label Propagation")
plt.ylim(0, .9)
plt.savefig("model_performance_label_prop_undersampled_ne.png")


plt.clf()
plt.figure(figsize=(15,5))
sea.set(style="whitegrid")
sea.set_palette(colo)
# Make plot
plot = sea.barplot(
    y='Score', 
    x='Model', 
    hue="SSL", 
    data=data[[x['Model'] == 'Random Forest' or x['Model'] == 'Ada-Boost' for _, x in data.iterrows()]],
    ci=None,
    orient='v'
   )
# resize figure box to -> put the legend out of the figure
box = plot.get_position() # get position of figure
plot.set_position([box.x0, box.y0, box.width * 0.75, box.height]) # resize position
# Put a legend to the right side
plot.legend(loc='top right', bbox_to_anchor=(1, 1), ncol=1)
plot.set_title("Model Performance Across Forms of Label Propagation")
plt.ylim(0, .9)
plt.savefig("model_performance_label_prop_undersampled_e_.png")
# Import the scripts





d2 = pd.read_csv('./balancing/balance_sample_results_prop.csv')

shape_tmp = []
for i, j in zip(['Train Accuracy', 'Test Accuracy', 'Validation Accuracy'], ['a Train Accuracy', 'b Test Accuracy', 'c Validation Accuracy']):
    tmp = d2[['Model', 'Label Propagation']].copy()
    tmp['Partition'] = i
    tmp['sort'] = j
    tmp['Score'] = d2[i]
    shape_tmp.append(tmp)

d2 = pd.concat(shape_tmp)



data = pd.concat([d2])
data['Label Propagation'].unique()
data.columns
data['Label Propagation'] = ['SKLearn Propagation' if x['Label Propagation'] == 'Label Propagation' else x['Label Propagation'] for _, x in data.iterrows()]
data['SSL'] = [x['Label Propagation'] + " " + x['Partition']  for _, x in data[['Label Propagation', 'Partition']].iterrows()]
#data['sort'] = data['SSL']
data['sort'] = [x['Label Propagation'] + " " + x['sort']  for _, x in data.iterrows()]
data['sort'] = ['y' + x['sort'] if x['Label Propagation'] == 'SKLearn Propagation' else x['sort']  for _, x in data.iterrows()]
data['sort'] = ['z' + x['sort'] if x['Label Propagation'] == 'No Propagation' else x['sort']  for _, x in data.iterrows()]
data = data.sort_values(by = "sort")
data.reset_index(inplace = True, drop=True)
data.loc[1].T


plt.clf()
plt.figure(figsize=(15,5))
sea.set(style="whitegrid")
sea.set_palette(colo)
# Make plot
plot = sea.barplot(
    y='Score', 
    x='Model', 
    hue="SSL", 
    data=data,
    ci=None,
    orient='v'
    )

# resize figure box to -> put the legend out of the figure
box = plot.get_position() # get position of figure
plot.set_position([box.x0, box.y0, box.width * 0.75, box.height]) # resize position
# Put a legend to the right side
plot.legend(loc='top right', bbox_to_anchor=(1, 1), ncol=1)

plot.set_title("Model Performance Across Forms of Label Propagation")
#change_width(plot, .35)
plt.ylim(0, .9)
plt.savefig("model_performance_label_prop_balanced.png")

maximum = data[data['Partition'] == 'Validation Accuracy']['Score'].max()
print(data[data['Partition'] == 'Validation Accuracy'].loc[maximum == data['Score']].T)
dat = pd.concat([d1,d2])

data=dat[dat['Partition'] == 'Validation Accuracy']
print(data.sort_values(by='Score'))
