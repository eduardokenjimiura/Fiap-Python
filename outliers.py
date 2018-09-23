
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
##, names=['city_mpg', 'highway_mpg', 'horsepower', 'price', 'width']

df = pd.read_csv('automobile-mod.csv' , sep=';' )
df['volume'] =df['length']*df['width']*df['height']
df['averageofconsume'] = (df['city_mpg']+df['highway_mpg'])/2
df = df.sort_values(by=['volume'])
#sns.boxplot(x = df['city_mpg'] )
print(df[['horsepower','averageofconsume']])
#plt.hist([df['horsepower'],df['engine_size']], color=['g','r'])
sns.lmplot(data = df ,y='price',x='volume' , order =1)
#sns.distplot(df["horsepower"])
plt.show()