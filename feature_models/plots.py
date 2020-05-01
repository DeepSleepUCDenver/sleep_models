# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

d1 = pd.read_csv('./baserun.csv')
d2 = pd.read_csv('./armfreq900_run_4-30.csv')
d3 = pd.read_csv('./core500freq.csv')
d4 = pd.read_csv('./sdram500freq.csv')
d5 = pd.read_csv('./core_ov_run_max_safe.csv')
d6 = pd.read_csv('./cor_ov_run_2.csv')
d1["Modification"] = "Factory_clock"
d2["Modification"] = "arm_freq_900"
d3["Modification"] = "sdram_freq_500"
d4["Modification"] = "core_freq_500"
d5["Modification"] = "overvolt_1.3v"
d6["Modification"] = "overvolt_1.4v"

data = pd.concat([d1, d2, d3, d4, d5, d6])
data.reset_index(inplace = True, drop=True)
data['Iteration'] = data.index
data['metric'] = data['metric'].apply(lambda x: 
        'Allocate/free 1000000 memory chunks' if x == 'Allocate/free 1000000 memory chunks (4-128 bytes)' else x)
data.columns
data.loc[1].T

sea.set_palette("Dark2")
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

