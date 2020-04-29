from time import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as sk_regressor
from sklearn.model_selection import train_test_split, GridSearchCV as sk_searchCV

from dask_ml.linear_model import LinearRegression as dask_regressor
from dask_ml.model_selection import GridSearchCV as dask_searchCV
import dask.dataframe as dd
import dask.array as da
from dask import delayed

import joblib
from dask.distributed import Client
from sklearn.externals import joblib

from ml_classes import MLNameSpace, SkLinRegressor, DaskLinRegressor, OutLog, RegType
from ml_funcs import readData, take_xy_lists, take_read_dir, get_xy_from_df, do_grid_search
import dask_ml
import dask



log_tol = 3
#prepearing data

data_df = readData(take_read_dir())
X_list, y_list = take_xy_lists()

df = data_df[y_list + X_list].dropna()

#######
df = df.iloc[:20000]

df_valid = df.iloc[-1:5001:-1]
df_for_learn = df.iloc[0:-5000]

X_for_learn, y_for_learn = get_xy_from_df(df_for_learn)
X_valid, y_valid = get_xy_from_df(df_valid)


#GridSearch all
param_grid = {'max_iter': [100, 1000]}
# grid_param = {'max_iter': [100,1000],
#               'tol': [10**-3, 10**-4]}

params = { 'cv': 2, 'n_jobs': -1}

#GridSearch Sklearn
X = X_for_learn
y = y_for_learn.iloc[:, 0]



client = Client(processes=False)
regressor = SkLinRegressor()
X, y = regressor.prepare_xy(X_for_learn, y_for_learn)

log_sk = OutLog(regType=RegType.SK)

print('GridSearch Sklearn start')

estimator, log_sk = do_grid_search(param_grid, params, X, y, regressor, sk_searchCV, log_sk,  client)


print('Estimator train start')
tic = time()
estimator.fit(X, y)
toc = time()
log_sk.set_param(train_time = round(toc - tic, log_tol))
print('Train stop, time = ', log_sk.train_time)
X_test, y_test = regressor.prepare_xy(X_valid, y_valid)
score = estimator.score(X_test, y_test)
log_sk.set_param(score=score)
print('Estimator scope = ', score)











#GridSearch dask_ml

regressor = DaskLinRegressor()
X, y = regressor.prepare_xy(X_for_learn, y_for_learn)
log_Dask = OutLog(regType=RegType.DASK)



print('GridSearch dask start')
estimator, log_Dask = do_grid_search(param_grid, params, X, y, regressor, sk_searchCV, log_Dask,  client)
print(estimator)
print('Estimator train start')
tic = time()
estimator.fit(X, y)
toc = time()
log_Dask.set_param(train_time = round(toc - tic, log_tol))
print('Train stop, time = ', log_Dask.train_time)
X_test, y_test = regressor.prepare_xy(X_valid, y_valid)
score = estimator.score(X_test, y_test)
log_Dask.set_param(score=score)
print('Estimator scope = ', score)


print()
log_sk.print_log(True)
log_Dask.print_log()






































from dask_ml.xgboost import XGBRegressor

# import numpy as np
# from sklearn.linear_model import LinearRegression

from distributed import Client
# client = Client().close()
# client = Client()

# est = XGBRegressor()
# X = np.array([[0], [1], [2], [3]])
# y = np.array([0, 1, 1, 4])
#
# from dask_glm.datasets import make_regression
# X, y = make_regression(n_samples=10)
# est1 = est.fit(X, y)

import dask_ml
import dask

# clf = KMeans(init_max_iter=3, oversampling_factor=10)
import dask.array as da
# X = da.from_array[[0, 0], [0, 1], [10,10]]

# clf.fit(X)

from dask_ml.linear_model import LinearRegression
from dask_glm.datasets import make_regression



#############################################

# from dask.distributed import Client
# print(1)
# client = Client(processes= False)
#
# # client = Client('scheduler-address:8787')
#
#
# print(2)
# import dask.dataframe as dd
# df = dd.read_csv(r'.\data\nycflights\1990.csv')
# print(df.head())
#
# # Split into training and testing data
# X_list = MLNameSpace.X_list
# y_list = MLNameSpace.y_list
# data_df = df;
# df = data_df[y_list + X_list].dropna()
#
# print(df.head())
# X = df[X_list]
# y = df[y_list].DepDelay
#
# print(X.head())
# print(y.head())
# # from xgboost import XGBRegressor  # change import
# from dask_ml.xgboost import XGBRegressor
#
# print(3)
# est = XGBRegressor()
# print(4)
# est.fit(X, y)
#
# print(5)
#
#
#
# print('yes')


