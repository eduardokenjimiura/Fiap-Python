import pandas as pd 
import numpy as np

df = pd.read_csv('automobile-mod.csv' , sep=';', header=None)
#df.loc[:, df.isna().any()]
#res = df.apply(lambda x: x.fillna("") if x. in 'biufc' else x.fillna('.'))
df.loc[:, [0,1]] = df.loc[:, [0,1]].fillna("")#trplcce na coluna esecifica 
print(df.columns[df.isna().any()].tolist())


#print(df.isnull().any())
#zdf[''] = df[''].fillna(0, inplace=True) 
#print(df.select_dtypes(include=['float64']))
#print(df.columns[df.isnull().any()].tolist())
