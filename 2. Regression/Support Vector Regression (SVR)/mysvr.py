# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

scalerX = StandardScaler()
X = scalerX.fit_transform(X)
scalerY = StandardScaler()
y = scalerY.fit_transform(y)


regressor = SVR(kernel = 'rbf') #Gaussian SVR

regressor.fit(X, y)



plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('SVR Reg')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

y_pred = scalerY.inverse_transform(regressor.predict(scalerX.transform(np.array([[6.5]]))))
