import numpy as np
import dask.array as da

from abc import ABC, abstractmethod
from enum import Enum

from sklearn.linear_model import Ridge as sk_estimator
from dask_ml.linear_model import LinearRegression as dask_estimator

# hard data MLNameSpace
class MLNameSpace:
    X_list = ['DayOfWeek','Distance', 'ArrDelay','CRSElapsedTime','Month', 'ArrTime', 'DepTime']
    y_list = ['DepDelay']
    data_list = X_list + y_list +['Year']
    data_dir = 'data'
    flight_dir = 'nycflights'
    log_tol = 3

#Enum for RegType
class RegType(Enum):
    NONE = 'NoneType'
    SK = 'sk_reg'
    DASK = 'dask_reg'

# class to contain and log out necessary info
class OutLog():
    log_Hat = 'regType, grid_time, train_time, score'

    def __init__(self, **kwargs):
        self.regType = RegType.NONE
        self.grid_time = -1
        self.train_time = -1
        self.score = -1
        self.set_param(**kwargs)
        self.log_Hat = OutLog.log_Hat


    def set_param(self,**kwargs):
        if 'regType' in kwargs:
            self.regType = kwargs['regType']
        if 'grid_time' in kwargs:
            self.grid_time = kwargs['grid_time']
        if 'train_time' in kwargs:
            self.train_time = kwargs['train_time']
        if 'score' in kwargs:
            self.score = kwargs['score']

    def print_log(self, log_Hat = False):
        if log_Hat == True: self.print_log_hat()
        print_list = [self.regType.value, self.grid_time,
                      self.train_time, self.score]
        print(*print_list, sep=', ')

    def print_log_hat(self):
        print(self.log_Hat)


### Regressor class intarface

class AbsRegressor(ABC):
    pass
# for Sklearn
class AbsSkRegressor(AbsRegressor):
    type = RegType.NONE

    @abstractmethod
    def __init__(self,):
        self.type = AbsSkRegressor.type
    pass

    def prepare_xy(self,X_for_learn, y_for_learn):
        X = X_for_learn
        y = y_for_learn.iloc[:, 0]
        return X, y

class SkLinRegressor(AbsSkRegressor):

    def __init__(self,):
        super(SkLinRegressor, self).__init__()
        self.estimator = sk_estimator()
    pass

# for Dask
class AbsDaskRegressor(AbsRegressor):
    type = RegType.DASK

    @abstractmethod
    def __init__(self,):
        self.type = AbsDaskRegressor.type
    pass

    def prepare_xy(self,X_for_learn, y_for_learn):
        X = da.array(np.array((X_for_learn)))
        y = da.array(np.array(y_for_learn.iloc[:, 0]))
        return X, y

class DaskLinRegressor(AbsDaskRegressor):

    def __init__(self,):
        super(DaskLinRegressor, self).__init__()
        self.estimator = dask_estimator()
    pass