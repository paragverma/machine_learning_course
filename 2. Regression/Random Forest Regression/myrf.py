# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X, y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('DT Reg')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

regressor.predict(6.5)