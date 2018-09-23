import pandas as pd 
import numpy as np
df = pd.read_excel('beer_consumption.xlsx', sheetname='Cerveja')
print(df.convert_objects(convert_numeric=True).dtypes)
 
print(df.isnull().any())
#print(df.columns[df.isnull().any()].tolist())
dict_gen = dict(zip(range(len(df.columns)),df.columns))