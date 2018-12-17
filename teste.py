# import pandas as pd 435334534534534535353534534534534535

# df = pd.read_csv('automobile-mod.csv' , sep=';', header=None)
# print(df.head(10))

import pandas as pd 
##, names=['city_mpg', 'highway_mpg', 'horsepower', 'price', 'width']
df = pd.read_csv('automobile-mod.csv' , sep=';' )
df['averageofconsume'] = (df['city_mpg']+df['highway_mpg'])/2
print(df.sort_values(by=['averageofconsume']))
 
#quanto maior o horse power, menor o consumo para um range de volume parecido 
#maior sera o preco do carro.


