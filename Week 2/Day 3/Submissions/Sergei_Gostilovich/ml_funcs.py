import os
from time import time
from glob import glob

import pandas as pd

import joblib
from dask.distributed import Client
from sklearn.externals import joblib

from ml_classes import MLNameSpace, OutLog




def get_xy_from_df(df):
    X_list, y_list, data_list = take_xy_data_lists()
    X = df[X_list]
    y = df[y_list]
    return X, y


def take_xy_data_lists():
    return MLNameSpace.X_list, MLNameSpace.y_list, MLNameSpace.data_list

def take_read_dir():
    read_dir = os.path.join(MLNameSpace.data_dir, MLNameSpace.flight_dir)
    return read_dir


def readData(read_dir):
    i = 0
    print("Read starting")
    for path in glob(os.path.join(read_dir, '*.csv')):

        if i == 0:
            df = pd.read_csv(path)
            i = 1
        else:
            df = df.append(pd.read_csv(path))
        print('File: ', path, ' has read')
    print("Read ended")
    return df


def do_grid_search(param_grid, params, X, y, regressor, searchCV_type,   log_out, client=None):
    log_tol = MLNameSpace.log_tol

    print('GridSearch start')
    estimator = regressor.estimator
    estimator.set_params(random_state=0, fit_intercept=False)
    searchCV = searchCV_type(estimator, param_grid, **params)

    if client is None:
        client = Client(processes=False)

    tic = time()
    with joblib.parallel_backend('dask'):
        searchCV.fit(X, y)
    toc = time()
    log_out.set_param(grid_time=round(toc - tic, log_tol))
    print('GridSearch stop, time = ', log_out.grid_time)
    best_estimator = searchCV.best_estimator_
    estimator.set_params(**best_estimator.get_params())
    print(estimator)
    return estimator

def do_train(X, y, regressor, log_out):
    log_tol = MLNameSpace.log_tol
    estimator = regressor.estimator
    print('Estimator train start')
    tic = time()
    estimator.fit(X, y)
    toc = time()
    log_out.set_param(train_time=round(toc - tic, log_tol))
    print('Train stop, time = ', log_out.train_time)
    return estimator






