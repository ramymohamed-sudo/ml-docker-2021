

import sys
import os

import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sqlite3.test import regression

""" what about decision tree """
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
# Multi-layer neural network MLP

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix     # noqa
from sklearn.metrics import accuracy_score, r2_score     # noqa
from sklearn.metrics import mean_absolute_error, mean_squared_error     # noqa
from sklearn.metrics import fbeta_score, make_scorer

# from sklearn.utils import shuffle
from load_data import load_train_data, load_test_data, normalize_data
import pickle
from joblib import dump, load

test_size = 0.2

model_dir = '../'   # for docker
file_path = model_dir + 'CMAPSSData/'
if not ('CMAPSSData' in os.listdir(model_dir)):
    file_path = './scripts/CMAPSSData/'

# model_dir_for_logs_and_h5 = model_dir+'logs-h5-models/'
model_dir_for_logs_and_h5 = './'
# if not ('logs-h5-models' in os.listdir(model_dir)):
#     model_dir_for_logs_and_h5 = './scripts/logs-h5-models/'

train_file = [file_path+f"train_FD00{i}.txt" for i in [1, 2, 3, 4]]
test_file = [file_path+f"test_FD00{i}.txt" for i in [1, 2, 3, 4]]
rul_file = [file_path+f"RUL_FD00{i}.txt" for i in [1, 2, 3, 4]]



""" Regression models """
# 0- example of cross-validation random forest without Gridsearch
# rf = RandomForestClassifier(n_jobs=-1)   # njobs for parallel tasks
# k_fold = KFold(n_splits=5)
# cross_val_score(rf,x_features,data['label'],cv=k_fold,scoring='accuracy',n_jobs=-1)

# 1- example of custom scoring:
# ftwo_scorer = make_scorer(fbeta_score, beta=2)
# grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer, cv=5)

""" Model names """
model_names = ['LinearRegression', 'DecisionTreeRegressor',
               'RandomForestRegressor', 'SVR',
               'ExtraTreesRegressor', 'GradientBoostingRegressor']

""" Model1: Linear Regression """
linear_model_reg = LinearRegression(n_jobs=-1)
# positive bool, default=False When set to True, forces the coefficients to be positive     # noqa
# n_jobs int, default=None The number of jobs to use for the computation.
# fit_interceptbool, default=True - Whether to calculate the intercept for this model. If set to False, # noqa
# no intercept will be used in calculations (i.e. data is expected to be centered).     # noqa


""" Model2: Decision Tree Model """
dt_model_reg = DecisionTreeRegressor()
# splitter{“best”, “random”}, default=”best” - The strategy used to choose the split at each node.
# max_depth int, default=None; The maximum depth of the tree. If None, then nodes are expanded until    # noqa
# all leaves are pure or until all leaves contain less than min_samples_split samples.  # noqa
# min_samples_split int or float, default=2; The minimum number of samples required to split an internal node:
# min_samples_leaf int or float, default=1 The minimum number of samples required to be at a leaf node.  # noqa
# max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto” The number of features to consider when     # noqa
# max_leaf_nodes int, default=None - Grow a tree with max_leaf_nodes in best-first fashion.  # noqa
# Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.     # noqa


""" Model3: Random Forest Model """
rf_model_reg = RandomForestRegressor(random_state=42, n_jobs=-1)   # default 'mse' loss function       # noqa
# n_estimatorsint, default=100 The number of trees in the forest.
# max_depth int, default=None - The maximum depth of the tree.
# min_samples_split int or float, default=2 The minimum number of samples required to split an internal node:     # noqa
# min_samples_leaf int or float, default=1 The minimum number of samples required to be at a leaf node.  # noqa
# max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto” The number of features to consider when     # noqa
# looking for the best split: If int, then consider max_features features at each split.     # noqa
# n_jobs int,

""" Model4: SVM Model """
svr_model_reg = SVR()   # loss{‘epsilon_insensitive’, ‘squared_epsilon_insensitive’}, default=’epsilon_insensitive’     # noqa
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
# Specifies the kernel type to be used in the algorithm. It must be one of
# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given,
# ‘rbf’ will be used. The kernel is the function used to map a lower dimensional data into a higher dimensional data.    # noqa
# degreeint, default=3 - Degree of the polynomial kernel function (‘poly’).
# Cfloat, default=1.0 Regularization parameter. The strength of the regularization is inversely proportional to C.   # noqa
# Must be strictly positive. The penalty is a squared l2 penalty.
# gamma{‘scale’, ‘auto’} or float, default=’scale’ - Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.     # noqa
# if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, if ‘auto’, uses 1 / n_features.    # noqa


""" Model4_1: Linear Support Vector Regression. """
# svr_model_reg = LinearSVR()
# Similar to SVR with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm,
# so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
# min_samples_split int or float, default=2. The minimum number of samples required to split an internal node:


""" Model5: ExtraTreesRegressor Model """
extra_tree_model_reg = ExtraTreesRegressor()
# This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) 
# on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.




""" Model6: GradientBoostingRegressor Model """
grad_boost_model_reg = GradientBoostingRegressor()
# Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.    # noqa
# criterion {‘friedman_mse’, ‘squared_error’, ‘mse’, ‘mae’}, default=’friedman_mse’ The function to measure the quality of a split.     # noqa
# subsample float, default=1.0 The fraction of samples to be used for fitting the individual base learners.         # noqa
# If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators.    # noqa
# Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.    # noqa

""" Bias - Variance - Regularization """

""" Pipeline """
reg_pipeline = Pipeline([("regression", rf_model_reg)])
#  for NLP Pipeline([("TF-IDF", tf_idf_model), ("regression", rf_model_reg)])

model1_reg = {"regression": [linear_model_reg]}

model2_reg = {"regression": [dt_model_reg],
              "regression__splitter": ["best", "random"],
              "regression__max_features": ["auto", "sqrt", "log2"]}   # max_features??
# max_depth int, default=None
# min_samples_split int or float, default=2
# min_samples_leaf int or float, default=1
# max_features int, float or {“auto”, “sqrt”, “log2”}, default=None

model3_reg = {"regression": [rf_model_reg],
              "regression__n_estimators": [10, 100],
              "regression__max_features": ["auto", "sqrt", "log2"]}   # max_features??
# n_estimators int, default=100
# max_depth int, default=None
# min_samples_split int or float, default=2
# min_samples_leaf int or float, default=1
# max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto”


model4_reg = {"regression": [svr_model_reg],
              "regression__C": [0.001, 0.01, 0.1, 1, 10],
              "regression__gamma": ['auto', 'scale'],
              'regression__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}
# degree int, default=3 (considered only if kernel=poly)
# gamma{‘scale’, ‘auto’} or float, default=’scale’
# C float, default=1.0 Regularization parameter.


model5_reg = {"regression": [extra_tree_model_reg],
              "regression__n_estimators": [10, 100],
              "regression__min_samples_leaf": [1, 2]}
# n_estimators  int, default=100
# max_depth int, default=None
# min_samples_split int or float, default=2
# min_samples_leaf  int or float, default=1


model6_reg = {"regression": [grad_boost_model_reg],
              "regression__n_estimators": [10, 100],
              "regression__learning_rate": [0.001, 0.01, 0.1],
              "regression__subsample": [0.5, 1]}

# loss{‘squared_error’, ‘absolute_error’, ‘huber’, ‘quantile’}, default=’squared_error’
# learning_rate float, default=0.1
# n_estimators  int, default=100
# subsample float, default=1.0
# (learning_rate=0.1,n_estimators=100,subsample=1)




""" Custom scoring 0 """
# print(sorted(sklearn.metrics.SCORERS.keys()))

# 2- example of custom scoring:
score_mse = make_scorer(mean_squared_error, greater_is_better=False)

""" Custom scoring 1 """
def my_custom_loss_func(y_true, y_pred):
    test_mser = np.sum(np.square(y_true-y_pred))/len(y_true)
    return test_mser

""" Custom scoring 2 """
scoring = {'accuracy': make_scorer(mean_squared_error, greater_is_better=False),
           'prec': 'precision'}


for i in range(len(train_file)):
    print(f"train file number {i+1}")
    df_train = load_train_data(train_file[i], test_file[i])
    df_test, data_RUL, df_test_RUL = load_test_data(test_file[i], rul_file[i])
    train_X, test_X, test_X_RUL = normalize_data(df_train, df_test, df_test_RUL)        # noqa
    train_y = df_train['Y'].values.astype('float32')
    test_y = df_test['Y'].values.astype('float32')
    test_y_RUL = df_test_RUL['Y'].values.astype('float32')

    print("train_y.shape", train_y.shape)
    print("test_y.shape", test_y.shape)
    print("test_y_RUL.shape", test_y_RUL.shape)
    print("test_X_RUL.shape", test_X_RUL.shape)

    print("before train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(train_X,
                                                        train_y,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        shuffle=True)
    # print("shape of X_train", X_train.shape)
    # print("shape of X_test", X_test.shape)
    # print("shape of y_train", y_train.shape)
    # print("shape of y_test", y_test.shape)

    search_space_reg = [model1_reg, model2_reg, model3_reg, model4_reg, model5_reg, model6_reg]       # noqa

    for j in range(len(search_space_reg)):
        print(f"\n >>>>>>>>>>>>>>>> train file {i+1} using model {model_names[j]}")
        clf_reg = GridSearchCV(reg_pipeline,
                               search_space_reg[j],
                               cv=5, n_jobs=-1,
                               scoring=score_mse)
                                # 'neg_mean_squared_error'  score_mse     # noqa 
        # scoring: By default, sklearn.metrics.accuracy_score for classification and sklearn.metrics.r2_score for regression    # noqa 

        best = clf_reg.fit(X_train, y_train)
        print(f"best.best_params_: {best.best_params_} and best.best_score_: {best.best_score_}")   # noqa
        means = clf_reg.cv_results_['mean_test_score']  # list of mean_test_score for each model     # noqa
        print("means......", means)

        # cv stands for cross-validations
        for mean, params in zip(means, clf_reg.cv_results_['params']):
            print("mean score is % 0.3f for % r " % (mean, params))

        y_pred_RUL = clf_reg.predict(test_X_RUL)
        data_RUL['pred_URL'] = y_pred_RUL
        print("data_RUL.head()\n", data_RUL.head())

        y_pred = clf_reg.predict(X_test)
        mean_abs_err = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')       # noqa
        mean_sqrd_err = mean_squared_error(y_test, y_pred, multioutput='uniform_average', squared=True)      # noqa
        root_mean_sqrd_err = sqrt(mean_squared_error(y_test, y_pred))

        print(f"Final mean_absolute_error is {mean_abs_err}")
        print(f"Final mean_squared_error is {mean_sqrd_err}")
        print(f"Root mean_squared_error is {root_mean_sqrd_err}")

        # with open(model_dir_for_logs_and_h5+"lr_rf_svr_.txt", "a") as f:
        #     f.write(f"For train file {i+1}, regressor type {model_names[j]}\n")
        #     f.write(f"Final mean_absolute_error is {mean_abs_err}\n")
        #     f.write(f"Final mean_squared_error is {mean_sqrd_err}\n")
        #     f.write(f"Root mean_squared_error is {root_mean_sqrd_err}\n")
        #     f.write("\n")

        dump(clf_reg, model_dir_for_logs_and_h5+f"clf_reg_{i+1}_{model_names[j]}.joblib")      # noqa




# print(f.read()) to read everything






# Bar plot of MAE/RMSE measures of our proposed CBLSTMs without dropout, CLSTM and CBLSTM under
# three different datasets: C1, C4 and C6, respectively.
# Finally, three tool life tests named C1, C4 and C6 were selected as our dataset. Each test contains 315 data samples, while each data sample has a corresponding flank wear. 
# For training/testing splitting, a three-fold setting is adopted such that two tests are used as the training domain and the other one is used as the testing domain. For example, when C4 and C6 are used as the training datasets, C1 will be adopted as the testing dataset. This splitting is denoted as c1.
# Our task is defined as the prediction of tool wear depth based on the sensory input.
# Total wear is plotted versus the number of cuts


# Table 3. MAE for compared methods on these three datasets. Bold face indicates the best performances.
# RNN, Recurrent Neural Network.

# Table 4. RMSE for compared methods on these three datasets. Bold face indicates the best performance


def plot_RUL(df):
    df['RUL'] = df['RUL'].astype(float)
    df[['RUL', 'pred_URL']].plot()
    # df.plot("RUL", figsize=(15, 5))
    plt.grid()
    plt.xlim(0, 100)
    plt.ylim(0, 160)
    plt.show()

# df = data_RUL.sort_values(by=['RUL'], ignore_index=True)
# print("df.head()\n", df.head())
# plot_RUL(df)