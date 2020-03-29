# Import the scripts
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

data = pd.read_feather('../feature_stage_data.ftr')
data = pd.concat([data.iloc[:,3:], data.iloc[:,1]], axis=1)
dfs = [x for _, x in data.groupby(data.stage)]
smpl = pd.DataFrame(columns=data.columns)
for df in dfs:
    s = df.sample(n=100, random_state=74)
    smpl = pd.concat([smpl, s], axis=0)


plot = sea.pairplot(data=smpl, hue='stage') # sns.pairplot(df, hue='species', size=2.5)
plot.savefig("pairplot.png")
