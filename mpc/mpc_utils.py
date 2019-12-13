import sys,os
import numpy
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from numpy import genfromtxt

from sklearn.svm import SVR


class mpc_regression(object):
    def __init__(self):
        self.mpc_regressor = None
        self.mpc_datapath = os.environ['PROJ_HOME_3'] + "/mpc/test_samps_400_with_cost.csv"

    def setup(self):
        X = genfromtxt(self.mpc_datapath, delimiter=',', skip_header=1, usecols=(1,2,3))
        Y = genfromtxt(self.mpc_datapath, delimiter=',', skip_header=1, usecols=(6))
        # print('X:', X)
        # print('shape of X:', X.shape)
        print('shape of Y:', Y.shape)
        num = X.shape[0]
        train_num = int(0.75 * num)

        X_train = X[0:train_num, :]
        Y_train = Y[0:train_num]

        X_test  = X[train_num:, :]
        Y_test  = Y[train_num:]

        # clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-5)
        # clf.fit(X_train, Y_train)
        # print("score:", clf.score(X_train, Y_train))

        # clf = SVR(C=1.0, epsilon=0.2)
        # clf.fit(X_train, Y_train)
        # score = clf.score(X_train, Y_train)
        # print("SVR scoring:", score)

        # # Create linear regression object
        # poly = PolynomialFeatures(degree=10)
        # # Poly features generation
        # X_train = poly.fit_transform(X_train)
        # X_test = poly.fit_transform(X_test)
        # # Fit the model
        # model = linear_model.LinearRegression()
        # model.fit(X_train, Y_train)

        # print("model scoring:", model.score(X_train, Y_train))
        # print("model scoring on test:", model.score(X_test, Y_test))

mpc_reg = mpc_regression()
mpc_reg.setup()