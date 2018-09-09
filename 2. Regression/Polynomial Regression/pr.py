# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

linreg = LinearRegression()
linreg.fit(X, y)
y_pred = linreg.predict(X)

polreg = PolynomialFeatures(degree = 4)
X_poly = polreg.fit_transform(X, y)

pol_l_reg = LinearRegression()
pol_l_reg.fit(X_poly, y)

#y_poly_pred = pol_l_reg.predict(X)


plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Linear Reg')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, pol_l_reg.predict(polreg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Reg')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

linreg.predict(6.5)
pol_l_reg.predict(polreg.fit_transform(6.5))
