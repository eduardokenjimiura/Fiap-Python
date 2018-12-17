 
#C    rie subsets de treinamento e teste utilizado uma razao adequada de tamanho. Utilze o train_test_split passando como parametros
 
from sklearn.model_selection import train_test_split
import pandas as pd ssssss
import sys
 
def SplitSubset(df,column_name):
    y = df[column_name].values #column of predict, in this case price.
    x = df.drop(column_name, axis=1).values
    #spliting the dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test 




def main(argv):
    dataset = pd.read_csv('automobile-mod.csv' , sep=';' )
    dataset.head(20)
    #bring value of csv
    x_train, x_test, y_train, y_test  = SplitSubset(dataset,'price')
    print(x_train, x_test, y_train, y_test)
    pass

if __name__ == "__main__":
    main(sys.argv) 
