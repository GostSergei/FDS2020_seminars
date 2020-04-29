from sklearn.model_selection import GridSearchCV as sk_searchCV
from dask.distributed import Client

from ml_classes import MLNameSpace, SkLinRegressor, DaskLinRegressor, OutLog, RegType
from ml_funcs import readData, take_xy_data_lists, take_read_dir, get_xy_from_df, do_grid_search, do_train


def main():
    # working with data
    data_df = readData(take_read_dir())

    X_list, y_list, data_list = take_xy_data_lists()
    df = data_df[data_list].dropna()

    # We try to predict the last year by knowing the data from previous years
    last_year = df.Year.max()
    df_valid = df[df.Year == last_year]
    df_for_learn = df[df.Year != last_year]

    X_for_learn, y_for_learn = get_xy_from_df(df_for_learn)
    X_valid, y_valid = get_xy_from_df(df_valid)

    # Creating client
    client = Client(processes=False)

    # GridSearch parameters
    param_grid = {'max_iter': [100, 1000],
                  'tol': [10**-3, 10**-4]}

    params = { 'cv': 2, 'verbose': 1, 'n_jobs': -1}


    # GridSearch Sklearn
    regressor = SkLinRegressor()
    X, y = regressor.prepare_xy(X_for_learn, y_for_learn)
    log_sk = OutLog(regType=RegType.SK)

    print('Sklearn start')
    do_grid_search(param_grid, params, X, y, regressor, sk_searchCV, log_sk,  client)
    do_train(X, y, regressor, log_sk)

    X_test, y_test = regressor.prepare_xy(X_valid, y_valid)
    score = regressor.estimator.score(X_test, y_test)
    log_sk.set_param(score=score)
    print('Estimator scope = ', score)
    print('Sklearn ended')



    # GridSearch dask_ml
    regressor = DaskLinRegressor()
    X, y = regressor.prepare_xy(X_for_learn, y_for_learn)
    log_Dask = OutLog(regType=RegType.DASK)

    print('Dask start')
    do_grid_search(param_grid, params, X, y, regressor, sk_searchCV, log_Dask,  client)
    do_train(X, y, regressor, log_Dask)

    X_test, y_test = regressor.prepare_xy(X_valid, y_valid)
    score = regressor.estimator.score(X_test, y_test)
    log_Dask.set_param(score=score)
    print('Estimator scope = ', score)

    # Output
    print()
    log_sk.print_log(True)
    log_Dask.print_log()


if __name__ == '__main__':
    main()

































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


