# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
#load the dataset
df = pd.read_csv(path)

#view the dataset
print(df.head())

#store independent and dependent variables
X = df.drop(['list_price'], 1)
y = df['list_price']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)




# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here 
cols = X_train.columns
fig, axes = plt.subplots(nrows = 3, ncols = 3)

for i in range(0,3):
    for j in range(0,3):
        col = cols[i*3 + j]
        axes[i,j].scatter(X_train[col], y_train)

plt.tight_layout()
plt.show()






# code ends here



# --------------
# Code starts here

#correlation between independent variables
corr = X_train.corr()
print(corr)

#dropping variables highly correlated with each other
X_train = X_train.drop(['play_star_rating', 'val_star_rating'], 1)
X_test = X_test.drop(['play_star_rating', 'val_star_rating'], 1)




# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here

#initializing linear regression model
regressor = LinearRegression()

#fit model to training data
regressor.fit(X_train, y_train)

#make predictions using X_test
y_pred = regressor.predict(X_test)

#mean squared error between ytest and ypred
mse = mean_squared_error(y_test, y_pred)
print(mse)

#r^2 score between ytest and ypred
r2 =r2_score(y_test, y_pred)
print(r2)



# Code ends here


# --------------
# Code starts here

#residual between ytest and ypred
residual = y_test - y_pred

#histogram of residual values
residual.plot(kind = 'hist', bins = 30)
plt.xlabel('Residuals')
plt.title('Histogram of Residual Values')
plt.show()




# Code ends here


