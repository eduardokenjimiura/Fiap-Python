import sys
import pandas as pd 
import numpy as np

def mapear_serie(serie):
    dict_gen = {}
    #sort value os array.
    uniquedataframe = sorted(serie.unique())
    #build a dictonary, from sorted array
    dict_gen = dict(zip(range(len(uniquedataframe)),uniquedataframe))
    return dict_gen

def main(argv):
    #bring value of csv
    df = pd.read_csv('automobile-mod.csv' , sep=';', header=None)
    dic= mapear_serie(df[0])
    print(dic)
    pass

if __name__ == "__main__":
    main(sys.argv) 

    #class_week={True:1, False:0}
    #df[''] = df[''].map(class_week) ---> muda o valor de acordo com o map