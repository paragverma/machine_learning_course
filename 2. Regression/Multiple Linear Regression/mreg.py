# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.formula.api as sm

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, :]

ols = sm.OLS(endog = y, exog = X_opt).fit()

stats = ols.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
ols = sm.OLS(endog = y, exog = X_opt).fit()
ols.summary()

X_opt = X[:, [0, 3, 4, 5]]
ols = sm.OLS(endog = y, exog = X_opt).fit()
ols.summary()

X_opt = X[:, [0, 3, 5]]
ols = sm.OLS(endog = y, exog = X_opt).fit()
ols.summary()

X_opt = X[:, [0, 3]]
ols = sm.OLS(endog = y, exog = X_opt).fit()
ols.summary()

X_opt = X[:, [3]]
lro = LinearRegression()
X_opt_train, X_opt_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

lro.fit(X_opt_train, y_train)
y_opt_pred = lro.predict(X_opt_test)