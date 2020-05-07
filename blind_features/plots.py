# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

d1 = pd.read_csv('./oversample/over_sample_results.csv')
#d2 = pd.read_csv('./undersample/under_sample_results.csv')
#d3 = pd.read_csv('./balancing/balance_sample_results.csv')
d1["Balancing"] = "Oversampled"
#d2["Balancing"] = "Undersampled"
#d3["Balancing"] = "Algorithm"
#d1['Label Propagation'].unique()

#data = pd.concat([d1, d2, d3])
data = pd.concat([d1])
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


sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(8,12))
plt.subplots_adjust(left=0.21)
sea.set(style="whitegrid")
sea.set_palette("Greens")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['SSL'] == "No Propagation" and i['Balancing'] == "Oversampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Blind Model Performance, Oversampled")
plt.xlim(0, .9)
plt.savefig("blind_model_performance__Oversampled.png")


