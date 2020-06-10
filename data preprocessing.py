import pandas as pd
import numpy as np
import matplotlib as plt


dataset = pd.read_csv("C:/Users/jites/Desktop/data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
# print("X = ", X)
# print("y = ", y)


from sklearn.impute import SimpleImputer                              #impute - library, SimpleImputer - Class
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')   #replacing missing values with mean
imputer = imputer.fit(X[:, 1:3])                                      #fitting our object imputer into the dataset
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ct = ColumnTransformer([("country", OneHotEncoder(), [1] )])
# X = ct.fit_transform(X)
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])                         #Encoding countries into 0,1,2

# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)                                   #Encoding yes or no into 1,0


##splitting into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''##Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_test)'''
