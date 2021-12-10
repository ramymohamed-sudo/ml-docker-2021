



import datetime
import sys
import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from random import randint
import requests
import zipfile
from io import StringIO
import os
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import urllib


pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = datetime.datetime.now()

model_dir = './' 
logs_directory = "logs-h5-models"
path = os.path.join(model_dir, logs_directory)
os.mkdir(path)



if not ('CMAPSSData' in os.listdir(model_dir)):
    data_directory = "CMAPSSData"
    path = os.path.join(model_dir, logs_directory)
    os.mkdir(path)

    # Download the file from `url` and save it locally under `file_name`:
    url1 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD001.txt'
    url2 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD002.txt'
    url3 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD003.txt'
    url4 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD004.txt'
    url5 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD001.txt'
    url6 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD002.txt'
    url7 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD003.txt'
    url8 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD004.txt'
    url9 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD001.txt'
    url10 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD002.txt'
    url11 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD003.txt'
    url12 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD004.txt'

    urllib.request.urlretrieve(url1, data_directory+'/train_FD001.txt')
    urllib.request.urlretrieve(url2, data_directory+'/train_FD002.txt')
    urllib.request.urlretrieve(url3, data_directory+'/train_FD003.txt')
    urllib.request.urlretrieve(url4, data_directory+'/train_FD004.txt')
    urllib.request.urlretrieve(url5, data_directory+'/test_FD001.txt')
    urllib.request.urlretrieve(url6, data_directory+'/test_FD002.txt')
    urllib.request.urlretrieve(url7, data_directory+'/test_FD003.txt')
    urllib.request.urlretrieve(url8, data_directory+'/test_FD004.txt')
    urllib.request.urlretrieve(url9, data_directory+'/RUL_FD001.txt')
    urllib.request.urlretrieve(url10, data_directory+'/RUL_FD002.txt')
    urllib.request.urlretrieve(url11, data_directory+'/RUL_FD003.txt')
    urllib.request.urlretrieve(url12, data_directory+'/RUL_FD004.txt')

    print("data is downloaded")


""" --------------- HyperParameters -------------- """
R_early = 130
feature_range = [0, 1]
names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3',
         's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
         's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
         's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26']
selc_sensors = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13',
                's14', 's15', 's17', 's20', 's21']
cols_to_drop = sorted(list(set(names[2::]) - set(selc_sensors)))
# window_size for 1D = [30, 20, 30, 15]     Here it is 31-2 = 29??
# optimizer, loss, activation functions, other parameters




# convert series to supervised learning
def series_to_supervised(data, window_size=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    # print("n_vars", n_vars)  # 18
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(window_size, 0, -1):        # window_size=29 for engine no 1
        # print(f"i is {i}")
        cols.append(df.shift(i))
        # print("cols", cols)
        names += [(df.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
        # print("names", names)

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(df.columns[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(df.columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_data(data_path):
    operational_settings = ['setting{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['s{}'.format(i + 1) for i in range(26)]
    cols = ['engine_id', 'cycle'] + operational_settings + sensor_columns
    data = pd.read_csv(data_path, sep=' ', header=None, names=cols)
    data = data.drop(cols_to_drop, axis=1)
    # data['index'] = data.index
    # data.index = data['index']
    # data['time'] = pd.date_range('1/1/2000', periods=data.shape[0], freq='600s')  # noqa
    # print(f'Loaded data with {data.shape[0]} Recordings')    # Recordings\n{} Engines     # noqa
    # print(f"Number of engines {len(data['engine_id'].unique())}")
    # print('21 Sensor Measurements and 3 Operational Settings')
    return data


def load_train_data(train_file, test_file):
    """ 1. Load training data """
    training_data = load_data(train_file)
    # print("data.head():\n", training_data.head())
    # print("data.shape:\n", training_data.shape)
    # print("data.var():\n", training_data.drop(
    #     ['engine_id', 'cycle'], axis=1).var())
    num_engine = max(training_data['engine_id'])

    """ 2. Load test data to get window_size"""
    test_data = load_data(test_file)
    max_window_size = min(test_data.groupby('engine_id')['cycle'].max())
    window_size = max_window_size - 2   # 31 - 2 =29

    """ 3. Remove columns that are not useful for prediction """

    """ 4. Convert time series to features. """
    # df is the training data
    df_train = pd.DataFrame()

    for i in range(num_engine):
        df1 = training_data[training_data['engine_id'] == i+1]
        max_cycle = max(df1['cycle'])
        # print("df1.shape", df1.shape)   # df1.shape (192, 18)
        # print("df1 max_cycle", max_cycle)   # df1 max_cycle 192
        # Calculate Y (RUL)
        # df1['RUL'] = max_cycle - df1['cycle']
        df1['RUL'] = df1['cycle'].apply(lambda x: max_cycle-x)
        df1['RUL'] = df1['RUL'].apply(lambda x: R_early if x > R_early else x)
        df2 = df1.drop(['engine_id'], axis=1)
        df3 = series_to_supervised(df2, window_size, 1)
        df_train = df_train.append(df3)     # dataframes under each other

    df_train.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    for col in df_train.columns:
        if col.startswith('RUL'):
            df_train.drop([col], axis=1, inplace=True)

    return df_train


def load_test_data(test_file, rul_file):
    test_data = load_data(test_file)
    max_window_size = min(test_data.groupby('engine_id')['cycle'].max())
    window_size = max_window_size - 2   # 31 - 2 =29

    """ Load RUL """
    data_RUL = pd.read_csv(rul_file,  header=None, names=["RUL"])
    num_engine_t = data_RUL.shape[0]

    # df_t is the testing data
    df_test = pd.DataFrame()
    df_test_RUL = pd.DataFrame()
    for i in range(num_engine_t):
        df1 = test_data[test_data['engine_id'] == i+1]
        max_cycle = max(df1['cycle']) + data_RUL.iloc[i, 0]
        # Calculate Y (RUL)
        df1['RUL'] = max_cycle - df1['cycle']
        df1['RUL'] = df1['RUL'].apply(lambda x: R_early if x > R_early else x)
        # df2 = df1.drop(['engine_id', 'cycle'], axis=1)
        df2 = df1.drop(['engine_id'], axis=1)
        df3 = series_to_supervised(df2, window_size, 1)
        df_test = df_test.append(df3)
        df_test_RUL = df_test_RUL.append(df3.tail(1))   # of last row

    df_test.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    df_test_RUL.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    for col in df_test.columns:
        if col.startswith('RUL'):
            df_test.drop([col], axis=1, inplace=True)
            df_test_RUL.drop([col], axis=1, inplace=True)

    return df_test, data_RUL, df_test_RUL


def normalize_data(df_train, df_test, df_test_RUL):
    scaler = MinMaxScaler(feature_range=feature_range)

    train_values = df_train.drop('Y', axis=1).values  # only normalize X, not y     # noqa
    train_values = train_values.astype('float32')
    scaled_train = scaler.fit_transform(train_values)

    values_test = df_test.drop('Y', axis=1).values  # only normalize X, not y   # noqa
    values_test = values_test.astype('float32')
    scaled_test = scaler.transform(values_test)

    values_test_RUL = df_test_RUL.drop('Y', axis=1).values  # only normalize X, not y   # noqa
    values_test_RUL = values_test_RUL.astype('float32')
    scaled_test_RUL = scaler.transform(values_test_RUL)

    return scaled_train, scaled_test, scaled_test_RUL

