


import os
from re import X
import sys
import csv
from matplotlib import pyplot as plt
import time
import datetime
from tensorflow.python.keras.engine.input_layer import Input
from load_data_4_files_1D_2D import load_train_data, load_test_data, normalize_data

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dropout, MaxPooling1D, Conv1D
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.models import Sequential, Model
import keras.backend as K
import keras_tuner as kt
# print(help(kt.Hyperband()))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = datetime.datetime.now()

from numpy.random import seed
seed(42)
tf.random.set_seed(42)


""" .......... Data preprocesing """
N_ft = 15
dim1_list = [30, 20, 30, 15]     # or FD001 to FD004
""" Model used modular or FUNC   """
learning_rate = 0.001  # then 0.0001
F_N = 10    # FN ﬁlters are used in each layer
F_L = 4     # and the ﬁlter size is FL× 1
epochs = 100    # 250   500
dropout_rate = 0.5
# n_FC = 100
batch_size = 512
MODEL = 'MODULAR'   # 'FUNC'
ARCH = 'RNN'    # "LSTM" "CLASSIC" ...


model_dir = './'
file_path = model_dir + 'CMAPSSData/'
if not ('CMAPSSData' in os.listdir(model_dir)):
    file_path = './scripts/CMAPSSData/'

logs_and_h5_path = model_dir+'logs-h5-models/'
if not ('logs-h5-models' in os.listdir(model_dir)):
    logs_and_h5_path = './scripts/logs-h5-models/'
print("logs_and_h5_path", logs_and_h5_path)

train_file = [file_path+f"train_FD00{i}.txt" for i in [1, 2, 3, 4]]
test_file = [file_path+f"test_FD00{i}.txt" for i in [1, 2, 3, 4]]
rul_file = [file_path+f"RUL_FD00{i}.txt" for i in [1, 2, 3, 4]]


class StoreModelHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None, k=''):
        if ('lr' not in logs.keys()):
            logs.setdefault('lr', 0)
            logs['lr'] = K.get_value(self.model.optimizer.lr)

        if not (f'rnn_with_scale_epochs_{epochs}.csv' in os.listdir(logs_and_h5_path)):     # noqa
            with open(logs_and_h5_path+f'rnn_with_scale_epochs_{epochs}.csv','a') as f:     # noqa
                y = csv.DictWriter(f, logs.keys())
                y.writeheader()

        with open(logs_and_h5_path+f'rnn_with_scale_epochs_{epochs}.csv','a') as f:     # noqa
            y = csv.DictWriter(f, logs.keys())
            y.writerow(logs)


es_callback = EarlyStopping(monitor="val_root_mean_squared_error",
                                      verbose=1,
                                      patience=5,
                                      mode="auto",
                                      baseline=None,
                                      restore_best_weights=False)
mc_callback = ModelCheckpoint('best_model.h5',
                               monitor='val_root_mean_squared_error',
                               save_best_only=True)

callback_list = [StoreModelHistory(), es_callback, mc_callback]


def model_builder(hp, units, n_FC, activation_rnn, activation_dense, lr, dropout, MODEL=MODEL):

    if MODEL == 'MODULAR':
        print("MODULAR Keras is used .....")
        print("dim2", dim2)
        dim3 = N_ft
        model = Sequential()
        model.add(LSTM(units=units,
                       input_shape=(dim2, dim3),
                       return_sequences=False,
                       activation=activation_rnn))
        if dropout:
            model.add(Dropout(rate=dropout_rate))

        model.add(Dense(units=n_FC, activation=activation_dense))
        model.add(Dense(1, activation=None))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=opt,        # 'adam'
                        metrics=['mean_absolute_error', 'RootMeanSquaredError'])      # noqa
        # metrics=[keras.metrics.MeanSquaredError()]
        # metrics=[keras.metrics.RootMeanSquaredError()]

    elif MODEL == 'FUNC':
        print("FUNCTIONAL Keras is used .....")
        dim3 = N_ft
        (T, D) = (dim2, N_ft)
        inputs = Input(shape=(T, D))
        x = LSTM(LSTM(units=units,
                      input_shape=(dim2, dim3),
                      return_sequences=False,
                      activation=activation_rnn))(inputs)

        x = Dense(n_FC, activation=activation_dense)(x)
        outputs = Dense(1, activation=None)(x)
        model = Model(inputs=inputs, outputs=outputs, name="lstm_1l_model")             # noqa

    return model


def model_tuner(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    n_FC = hp.Int("units", min_value=32, max_value=512, step=32)
    activation_rnn = hp.Choice("activation", ["relu", "tanh"])
    activation_dense = hp.Choice("activation", ["relu", "tanh"])
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")      # noqa
    dropout = hp.Boolean("dropout", default=False)

    # call existing model-building code with the hyperparameter values.
    model = model_builder(hp, units, n_FC, activation_rnn, activation_dense, lr, dropout)

    return model


for i in range(len(train_file)):
    print(f"train file number {i+1}")
    sequence_length = dim1_list[i]
    dim2 = sequence_length
    print("sequence_length", sequence_length)

    df_train = load_train_data(train_file[i], test_file[i], sequence_length)
    df_test, data_RUL, df_test_RUL = load_test_data(test_file[i], rul_file[i], sequence_length)     # noqa

    train_X, test_X, test_X_RUL = normalize_data(df_train, df_test, df_test_RUL)        # noqa
    train_X = train_X.reshape(train_X.shape[0], sequence_length, -1)
    test_X = test_X.reshape(test_X.shape[0], sequence_length, -1)
    test_X_RUL = test_X_RUL.reshape(test_X_RUL.shape[0], sequence_length, -1)

    print("train_X.shape", train_X.shape)
    print("test_X.shape", test_X.shape)
    print("test_X_RUL.shape", test_X_RUL.shape)

    train_y = df_train['Y'].values.astype('float32')
    test_y = df_test['Y'].values.astype('float32')
    test_y_RUL = df_test_RUL['Y'].values.astype('float32')

    print("train_y.shape", train_y.shape)
    print("test_y.shape", test_y.shape)
    print("test_y_RUL.shape", test_y_RUL.shape)

    # You can quickly test if the model builds successfully.
    # model = model_builder(hp=kt.HyperParameters())
    model = model_tuner(hp=kt.HyperParameters())

    tuner = kt.RandomSearch(hypermodel=model_tuner,
                            objective=kt.Objective("val_root_mean_squared_error", direction="min"),     # noqa
                            max_trials=5,
                            executions_per_trial=2,
                            overwrite=True,
                            directory=logs_and_h5_path,
                            project_name="rnn__one_layer",)

    print("tuner.search_space_summary()", tuner.search_space_summary())
    tuner.search(train_X, train_y, epochs=epochs,
                 validation_data=(test_X_RUL, test_y_RUL),
                 callbacks= callback_list,      # [StoreModelHistory()]
                 verbose=2,
                 batch_size = batch_size,
                 shuffle=True)
    
    bestModels = tuner.get_best_models(num_models=1)
    print("bestModels", bestModels)
    highestScoreModel = bestModels[0]
    print("highestScoreModel.summary()", highestScoreModel.summary())
    
    # keras.utils.plot_model(model, "LSTM.png")
    # plt.show()
    # for layer in model.layers:
    # print(layer.output_shape)
    sys.exit()



# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
# optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)




# tf.keras.layers.Reshape(target_shape, **kwargs)
# https://keras.io/api/layers/reshaping_layers/reshape/
# https://keras.io/api/layers/reshaping_layers/flatten/
# model.output_shape

# prediction = tf.reshape(output, [-1])
# y_flat = tf.reshape(Y, [-1])
# h = prediction - y_flat
# cost_function = tf.reduce_sum(tf.square(h))
# RMSE = tf.sqrt(tf.reduce_mean(tf.square(h)))
# optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)
# saver = tf.train.Saver()
# training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
