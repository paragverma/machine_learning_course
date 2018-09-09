# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linear_regression.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Training)')
plt.xlabel('Experience years')
plt.ylabel('Salary')

plt.show()

plt.close()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')

plt.title('Salary vs Experience (Test)')
plt.xlabel('Experience years')
plt.ylabel('Salary')

plt.show()