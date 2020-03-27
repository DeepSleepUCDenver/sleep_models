# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

data = pd.read_feather("./flat_data/Staging_Data_GT_190315.ftr")


sea.lineplot(y='eeg', x='index', data=data[:500])
plt.show()

eeg = data[['eeg']][:500]
eog = data[['eog']][:500]
emg = data[['emg']][:500]
eeg.columns = ['measure']
eog.columns = ['measure'] 
emg.columns = ['measure']
eeg['wave'] = 'eeg'
eog['wave'] = 'eog'
emg['wave'] = 'emg'
data_joined = pd.concat([eeg, eog, emg], axis=0)
data_joined['index'] = data_joined.index
sea.lineplot(y='measure', x='index', hue='wave', data=data_joined)
plt.show()

data = pd.read_feather('./feature_stage_data.ftr')
data = pd.concat([data.iloc[:,3:], data.iloc[:,1]], axis=1)
dfs = [x for _, x in data.groupby(data.stage)]
len(dfs)
smpl = pd.DataFrame(columns=data.columns)
for df in dfs:
    s = df.sample(n=100, random_state=74)
    smpl = pd.concat([smpl, s], axis=0)

smpl.stage 
smpl.shape

plot = sea.pairplot(data=smpl, hue='stage') # sns.pairplot(df, hue='species', size=2.5)
plot.savefig("pairplot.png")
