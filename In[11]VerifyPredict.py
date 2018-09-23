# Crie subsets de treinamento e teste utilizado uma razao adequada de tamanho. Utilze o train_test_split passando como parametros
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model


# variaveis preditoreas fazer grafico verificando o comportamento, analise de correlacao => atributo preditor


def SplitSubset(df, column_namePredict, feature_col_names):
    x = df[feature_col_names].values
    y = df[column_namePredict].values
    # spliting the dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def RemoveNullValues(df, column_name):
    # get mode from RPM
    df[column_name] = df[column_name].replace(
        np.nan, df[column_name].mode()[0])
    return df

# make	fuel_type	aspiration	number_of_doors	body_style	drive_wheels
# engine_location	wheel_base	length	width	height	curb_weight
# engine_type	number_of_cylinders	engine_size	fuel_system	bore	stroke
# compression_ratio	horsepower	peak_rpm	city_mpg	highway_mpg	price


def main(argv):
    predicted_class_names = ['price']

    predictedClassNameOfHighwayMpg = ['highway_mpg']

    # bring value of csv
    dataset = pd.read_csv('automobile-mod.csv', sep=';')
    receber esse cara como datase  print('sssss', dataset.loc[dataset['city_mpg'].notnull(), ['length', 'width', 'height','horsepower', 'compression_ratio', 'stroke', 'bore' ]]  )
    dsmpg = dataset[dataset.filter(like='city_mpg').isnull().any(1)]
    print(dsmpg)
    #print(dataset[dataset.city_mpg != np.nan])
    #dsmpg = dataset[dataset.filter(like='city_mpg').isnull().any(1)]
    #dataset = dataset.dropna(axis=0,subset=['city_mpg'])
    # null values from mpg use to find city mpg values
    #print(dataset.filter(like ='city_mpg').isnull().any(1))
    dataset = dataset[dataset['city_mpg'].isna()]
    #dataset = RemoveNullValues(dataset, "city_mpg")
    #dataset = dataset.drop(dataset.index[[38]])
     #dataset = dataset[dataset['city_mpg'].notnull()]
    #dataset = dataset.reset_index(drop=True)
     #dataset = RemoveNullValues(dataset, "city_mpg")
    
   # print(dataset[dataset.filter(like='city_mpg').isnull().any(1)])
   
    predictedClassNameOfCityMpg = ['city_mpg']
    featureColMpgValues = ['length', 'width', 'height',
                           'horsepower', 'compression_ratio', 'stroke', 'bore']

    x_trainmpg, x_testmpg, y_trainmpg, y_testmpg = SplitSubset(
        dataset, predictedClassNameOfCityMpg, featureColMpgValues)
    lr_modelMpg = linear_model.LinearRegression()
    lr_modelMpg.fit(np.array(x_trainmpg), np.array(y_trainmpg).ravel())
    y_predmpg = lr_modelMpg.predict(x_testmpg)
    print('R2 score  city mpg Mpg: %.2f' % r2_score(y_testmpg, y_predmpg))
    print('R2 score   city mpg lMpg: %.2f' %
          lr_modelMpg.score(x_testmpg, y_testmpg))
    # how to calculate city_mpg from veicule, based in anothers variables?

    # how to calculate highway_mpg from veicule, based in anothers variables?
    dataset = RemoveNullValues(dataset, "highway_mpg")

    # Volum of car
    dataset['volume'] = dataset['length']*dataset['width']*dataset['height']
    # change null values to modal
    dataset = RemoveNullValues(dataset, "peak_rpm")

    feature_col_names = ['peak_rpm', 'city_mpg',
                         'highway_mpg', 'horsepower', 'volume']

    x_train, x_test, y_train, y_test = SplitSubset(
        dataset, predicted_class_names, feature_col_names)
    lr_model = linear_model.LinearRegression()
    lr_model.fit(np.array(x_train), np.array(y_train).ravel())
    y_pred = lr_model.predict(x_test)
    #print('R2 score: %.2f' % r2_score(y_test, y_pred))
  #  print('R2 score: %.2f' % lr_model.score(x_test, y_test))


pass


if __name__ == "__main__":
    main(sys.argv)
