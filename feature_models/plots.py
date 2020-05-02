# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

d1 = pd.read_csv('./oversample/over_sample_results_2.csv')
d1_2 = pd.read_csv('./oversample/over_sample_results_1.csv')
d2 = pd.read_csv('./undersample/under_sample_results.csv')
d3 = pd.read_csv('./balancing/balance_sample_results.csv')
d1["Balancing"] = "Oversampled"
d2["Balancing"] = "Undersampled"
d3["Balancing"] = "Algorithm"

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
data.columns
data.loc[1].T

#color='Blues',
sea.set_palette(sea.color_palette()[-3:])
plt.clf()
plt.figure(figsize=(7,10))
plt.subplots_adjust(left=0.35)
sea.set(style="whitegrid")
sea.set_palette("Blues")
sea.set_palette(sea.color_palette()[-3:])
plot = sea.barplot(
    x='Score', 
    y='Model', 
    hue="Partition", 
    data=data[[(i['Label Propagation'] == "No Propagation" and i['Balancing'] == "Undersampled") for _, i in data.iterrows()]],
    ci=None,
    orient='h'
    )
plot.set_title("Model Performance on ")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig("model_performance.png")

for i, met in enumerate(data.metric.unique()):
    plt.clf()
    plt.figure(figsize=(9,5))
    plt.subplots_adjust(left=0.09)
    plot = sea.boxplot(y='measure', x='Modification', data=data[data.metric == met])
    plot.set_title(met)
    yl = data[data.metric == met]['units'].values[0] + "  <Lower is Better>"
    plot.set_ylabel(yl)
    plot.figure.savefig("mods_" + str(i) + ".png", transparent=True)

plt.close('all')

plt.clf()
plt.figure(figsize=(8,5))
plot = sea.lineplot(
        y='heat', 
        x='Iteration', 
        hue="Modification",  
        data=data
        )
plot.set_title("Temperature Over Testing Cycle")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig("heat_vs_time_vs_mods.png", transparent=True)

for i, met in enumerate(data.metric.unique()):
    plt.clf()
    plt.figure(figsize=(5,4))
    plot = sea.scatterplot(
            y='heat', 
            x='measure', 
            style='GPU Memory', 
            hue='Priority',  
            data=data[data["metric"] == met]
            )
    plot.set_title(met+" (Speed vs Temperature)")
    xl = data[data.metric == met]['units'].values[0] + "  <Lower is Better>"
    plot.set_xlabel(xl)
    plot.set_ylabel("Temperature (Celsius)")
    plt.savefig("heat_vs_speed_" + str(i) + ".png", transparent=True)

plt.clf()
plot = sea.scatterplot(
       y='heat', 
       x='volt', 
       hue='Modification',  
       data=data
       )
plt.savefig("heat_vs_volt.png", transparent=True)

