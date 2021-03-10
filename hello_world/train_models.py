from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import joblib

import pandas as pd
import numpy as np

# importing the boston housing dataset
boston_df = datasets.load_boston()

x = boston_df['data']
y = boston_df['target']

print('[INFO] started training the models')
# Linear Model
linear_model_filename = 'lm.pkl'
lm = linear_model.LinearRegression()
lm.fit(x, y)

joblib.dump(lm, linear_model_filename)
print('[INFO] completed training the linear model')

#Support Vector Machine
svm_filename = 'svm.pkl'
svm = svm.SVR()
svm.fit(x, y)

joblib.dump(svm, svm_filename)
print('[INFO] completed training the svm model')

# Random Forest
random_forest_filename = 'randomforest.pkl'
rf_m = RandomForestRegressor()
rf_m.fit(x, y)

joblib.dump(rf_m, random_forest_filename)
print('[INFO] completed training the random forest model')


print('[INFO] Trained and saved the models')