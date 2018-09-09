# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



dataset = pd.read_csv('Social_Network_Ads.csv')

#X = dataset.iloc[:, :-1].values
X = dataset.iloc[:, [2, 3]]
y = dataset.iloc[:, 4].values

#labelencoder = LabelEncoder()
#X[:, 1] = labelencoder.fit_transform(X[:, 1])
#X = X.astype(float)
#y = y.astype(float)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)

logreg = LogisticRegression(random_state = 0)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
