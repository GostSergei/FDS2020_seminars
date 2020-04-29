from time import time
import joblib
from dask.distributed import Client
from sklearn.externals import joblib

import pandas as pd
import numpy as np
import dask.array as da
import os
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob


from sklearn.linear_model import Ridge as sk_regressor
from sklearn.model_selection import train_test_split, GridSearchCV as sk_searchCV
from dask_ml.linear_model import LinearRegression as dask_regressor

from ml_classes import MLNameSpace, OutLog

def get_xy_from_df(df):
    X_list, y_list = take_xy_lists()
    X = df[X_list]
    y = df[y_list]
    return X, y


def take_xy_lists():
    return MLNameSpace.X_list, MLNameSpace.y_list

def take_read_dir():
    read_dir = os.path.join(MLNameSpace.data_dir, MLNameSpace.flight_dir)
    return read_dir

def readData(read_dir):
    df = pd.read_csv(os.path.join(read_dir, '1990.csv'))
    return df


def do_grid_search(param_grid, params, X, y, regressor, searchCV_type,   log_out, client=None):
    log_tol = 3


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
    estimator = searchCV.best_estimator_
    print(estimator)
    return estimator, log_out




