# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
#Load the Dataset
df = pd.read_csv(path)

#Display first 5 rows
print(df.head())

#Store Independent Variables and Dependent Variable
X = df.drop(['Price'], 1)
y = df['Price']

#Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

#Correlation between Independent Variables
corr = X_train.corr()
print(corr)



# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here

#Instantiate Linear Regression Model
regressor = LinearRegression()

#Fit training data on model
regressor.fit(X_train, y_train)

#Make Predictions on X_test
y_pred = regressor.predict(X_test)

#Find Coefficient of Determination between ytest and ypred
r2 = r2_score(y_test, y_pred)
print("The Coefficient of Determination between y_test and y_pred is {}.".format(round(r2,2)))


# --------------
from sklearn.linear_model import Lasso

# Code starts here

#Instantiate a Lasso Regression Model
lasso = Lasso()

#Fit the model on training data
lasso.fit(X_train, y_train)

#Make predictions using lasso model on X_test
lasso_pred = lasso.predict(X_test)

#Coefficient of Determination between LassoPred and ytest
r2_lasso = r2_score(y_test, lasso_pred)
print("The coefficient of Determination between y_test and lasso_pred is {}.".format(round(r2_lasso,2)))
print("Using a lasso model has not improved the r2 score, it is still the same.")



# --------------
from sklearn.linear_model import Ridge

# Code starts here

#Instantiate a Model for Ridge Regression
ridge = Ridge()

#Fit training data on the model
ridge.fit(X_train, y_train)

#Make predictions from the model using X_test
ridge_pred = ridge.predict(X_test)

#Coefficient of Determination between ytest and ridgePred
r2_ridge = r2_score(y_test, ridge_pred)
print("The Coefficient of Determination between y_test and ridge_pred is {}.".format(round(r2_ridge,2)))
print("Using ridge regression has not improved the r2 score, it is still the same.")



# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here

#Instantiate a model for Linear Regression
regressor = LinearRegression()

#Cross Validation Score for X_train and y_train
score = cross_val_score(regressor, X_train, y_train, cv = 10)

#Mean of cross validation scores
mean_score = np.mean(score)
print("The mean cross validation score for ou Linear Regression Model of Melbourne Housing Prices is {}.".format(round(mean_score,2)))


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here

#Initiate pipeline for Polynomial Features
model = make_pipeline(PolynomialFeatures(2), LinearRegression())

#Fit the model on training data
model.fit(X_train, y_train)

#Making predictions from the model using X_test
y_pred = model.predict(X_test)

#Coefficient of Determination between ytest and ypred
r2_poly = r2_score(y_test, y_pred)
print("The Coefficient of Determination between y_test and y_pred is {}.".format(round(r2_poly,2)))
print("Using polynomial regression has improved the r2 score. for our Melbourne Hosuing Prices model.")


