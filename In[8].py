import numpy as np
import pandas as pd 
import sys
 
 
def identificacao_outlier(df,column_name):
    print(column_name)
    #values of first quartiles
    q1 = np.percentile(df[column_name], 25)
    #values of third quartiles
    q3=  np.percentile(df[column_name], 75)
    # dif of first quantir and third
    iqr = q3 - q1
    outlier_lower = q1 - (iqr * 1.5)
    outlier_upper = q3 + (iqr * 1.5)
    return (outlier_upper + outlier_lower)

def main(argv):
    #bring value of csv
    df = pd.read_csv('automobile-mod.csv' , sep=';' )
    dic= identificacao_outlier(df,'height')
    print(dic)
    pass

if __name__ == "__main__":
    main(sys.argv) 
