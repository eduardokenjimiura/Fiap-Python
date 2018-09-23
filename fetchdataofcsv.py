# import pandas as pd 

# df = pd.read_csv('automobile-mod.csv' , sep=';', header=None)
# print(df.head(10))

import pandas as pd 
##, names=['city_mpg', 'highway_mpg', 'horsepower', 'price', 'width']
df = pd.read_csv('automobile-mod.csv' , sep=';' )
df['volume'] =df['length']*df['width']*df['height']
df['averageofconsume'] = (df['city_mpg']+df['highway_mpg'])/2
#print(df[['make', 'city_mpg','highway_mpg','averageofconsume','volume','horsepower' ,'price' ]].sort_values(by=['volume']))
#print(df.mode())
print(df.mode() )
#quanto maior o horse power, menor o consumo para um range de volume parecido 
#maior sera o preco do carro.
