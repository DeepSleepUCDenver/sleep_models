# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

d1 = pd.read_csv('./oversample/over_sample_results.csv')
d2 = pd.read_csv('./undersample/under_sample_results.csv')
d3 = pd.read_csv('./balancing/balance_sample_results.csv')
d1["Balancing"] = "Oversampled"
d2["Balancing"] = "Undersampled"
d3["Balancing"] = "Algorithm"
#d1['Label Propagation'].unique()

data = pd.concat([d1, d2, d3])
data.reset_index(inplace = True, drop=True)
#data['metric'] = data['metric'].apply(lambda x: 
#        'Allocate/free 1000000 memory chunks' if x == 'Allocate/free 1000000 memory chunks (4-128 bytes)' else x)
data.columns
data.loc[1].T
shape_tmp = []
for i in ['Test Accuracy', 'Train Accuracy', 'Validation Accuracy']:
    tmp = data[['Model', 'Label Propagation','Balancing']].copy()
    tmp['Partition'] = i
    tmp['Score'] = data[i]
    shape_tmp.append(tmp)

data = pd.concat(shape_tmp)
data['SSL'] = data['Label Propagation']
data['SSL'].unique()
data.columns
data.loc[1].T

#color='Blues',
sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,12))
plt.subplots_adjust(left=0.21)
sea.set(style="whitegrid")
sea.set_palette("Reds")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "No Propagation" and i['Balancing'] == "Undersampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, No Propagation, Undersampled")
plt.xlim(0, .9)
plt.savefig("model_performance_No_Propagation_Undersampled.png")

sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,12))
plt.subplots_adjust(left=0.21)
sea.set(style="whitegrid")
sea.set_palette("Blues")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "Label Propagation" and i['Balancing'] == "Undersampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, Label Propagation, Undersampled")
plt.xlim(0, .9)
plt.savefig("model_performance_Label_Propagation_Undersampled.png")

sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,12))
plt.subplots_adjust(left=0.21)
sea.set(style="whitegrid")
sea.set_palette("Reds")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "No Propagation" and i['Balancing'] == "Oversampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, No Propagation, Oversampled")
plt.xlim(0, .9)
plt.savefig("model_performance_No_Propagation_Oversampled.png")

sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,12))
plt.subplots_adjust(left=0.21)
sea.set(style="whitegrid")
sea.set_palette("Blues")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "Label Propagation" and i['Balancing'] == "Oversampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, Label Propagation, Oversampled")
plt.xlim(0, .9)
plt.savefig("model_performance_Label_Propagation_Oversampled.png")

sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,12))
plt.subplots_adjust(left=0.21)
sea.set(style="whitegrid")
sea.set_palette("Blues")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "Pseudo Labels" and i['Balancing'] == "Oversampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, Pseudo Labels, Oversampled")
plt.xlim(0, .9)
plt.savefig("model_performance_Pseudo_Labels_Oversampled.png")

sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,3))
plt.subplots_adjust(left=0.21)
sea.set(style="whitegrid")
sea.set_palette("Blues")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "Pseudo Iterative" and i['Balancing'] == "Oversampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, Iterative Pseudo Labels, Oversampled")
plt.xlim(0, .9)
plt.savefig("model_performance_Iterative_Labels_Oversampled.png")

sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,3))
plt.subplots_adjust(left=0.26)
sea.set(style="whitegrid")
sea.set_palette("Reds")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "No Propagation" and i['Balancing'] == "Algorithm") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, No Propagation, Balancing Algorithm")
plt.xlim(0, .9)
plt.savefig("model_performance_No_Propagation_Balancing_Algorithm.png")

sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,3))
plt.subplots_adjust(left=0.26)
sea.set(style="whitegrid")
sea.set_palette("Blues")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "Label Propagation" and i['Balancing'] == "Algorithm") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance, Label Propagation, Balancing Algorithm")
plt.xlim(0, .9)
plt.savefig("model_performance_Label_Propagation_Balancing_Algorithm.png")

