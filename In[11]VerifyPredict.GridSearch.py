#  Crie subsets de treinamento e teste utilizado uma razao adequada de tamanho. Utilze o train_test_split passando como parametros
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import GridSearchCV


# variaveis preditoreas fazer grafico verificando o comportamento, analise de correlacao => atributo preditor


def SplitSubset(df, column_namePredict, feature_col_names):
    x = df[feature_col_names].values
    y = df[column_namePredict].values
    # spliting the dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)
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


def mapear_serie(serie):
    dict_gen = {}
    # sort value os array.
    uniquedataframe = sorted(serie.unique())
    # build a dictonary, from sorted array
    dict_gen = dict(zip(uniquedataframe, range(len(uniquedataframe))))
    return dict_gen


def linearModel(x_train, x_test, y_train, y_test):
    lr_model = linear_model.LinearRegression()
    lr_model.fit(np.array(x_train), np.array(y_train).ravel())
    y_pred = lr_model.predict(x_test)
    print('R2 score linearModel: %.2f' % r2_score(y_test, y_pred))
    print('R2 score linearModel: %.2f' % lr_model.score(x_test, y_test))

def decisionTree(x_train, x_test, y_train, y_test):
    lr_model = tree.DecisionTreeRegressor()
    lr_model.fit(np.array(x_train), np.array(y_train).ravel())
    y_pred = lr_model.predict(x_test)
    print('R2 score decisionTree: %.2f' % r2_score(y_test, y_pred))
    print('R2 score decisionTree: %.2f' % lr_model.score(x_test, y_test))

def linearRidge(x_train, x_test, y_train, y_test):

    lr_model = linear_model.Ridge()
    lr_model.fit(np.array(x_train), np.array(y_train).ravel())
    y_pred = lr_model.predict(x_test)
    print('R2 score linearRidge: %.2f' % r2_score(y_test, y_pred))
    print('R2 score linearRidge: %.2f' % lr_model.score(x_test, y_test))

def GridSearchCVAn(x_train, x_test, y_train, y_test,parameters):
    lr_model = linear_model.LinearRegression()
    opt_model_lr = GridSearchCV(lr_model, parameters, scoring='r2')
    opt_model_lr.fit(x_train, y_train.ravel())
    y_pred = lr_model.predict(x_test)
    print('R2 score GridSearchCV: %.2f' % r2_score(y_test, y_pred))
    print('R2 score GridSearchCV: %.2f' % lr_model.score(x_test, y_test))
    return ''

def main(argv):
    predicted_class_names = ['price']

    dataset = pd.read_csv('automobile-mod.csv', sep=';')

    RemoveNullValues(dataset, "length")
    RemoveNullValues(dataset, "width")
    RemoveNullValues(dataset, "horsepower")
    RemoveNullValues(dataset, "height")
    RemoveNullValues(dataset, "compression_ratio")
    RemoveNullValues(dataset, "stroke")
    RemoveNullValues(dataset, "bore")
    RemoveNullValues(dataset, "engine_size")
    RemoveNullValues(dataset, "peak_rpm")
    RemoveNullValues(dataset, "city_mpg")
    dataset = RemoveNullValues(dataset, "highway_mpg")
    print((mapear_serie(dataset['number_of_cylinders'])))
    dataset["number_of_cylinders"] = dataset["number_of_cylinders"].map(
        (mapear_serie(dataset['number_of_cylinders'])))
    dataset["drive_wheels"] = dataset["drive_wheels"].map(
        (mapear_serie(dataset['drive_wheels'])))

    #print( dataset["number_of_cylinders"])
    # Volum of car
    dataset['volume'] = dataset['length']*dataset['width']*dataset['height']
    feature_col_names = ['city_mpg', 'highway_mpg',
                         'horsepower', 'volume', 'engine_size']
    x_train, x_test, y_train, y_test = SplitSubset(dataset, predicted_class_names, feature_col_names)
    linearModel(x_train, x_test, y_train, y_test)
    decisionTree(x_train, x_test, y_train, y_test)
    linearRidge(x_train, x_test, y_train, y_test)
    GridSearchCVAn(x_train, x_test, y_train, y_test,feature_col_names)

pass


if __name__ == "__main__":
    main(sys.argv)
