#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the datasets
dataset = pd.read_csv("C:/Users/jites/Desktop/Train week 2.csv")
X_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:,5].values
X_test = pd.read_csv("C:/Users/jites/Desktop/Test week 2")

#Fitting Multiple linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)
