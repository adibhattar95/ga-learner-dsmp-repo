# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here

#Load the dataset
df = pd.read_csv(path)

#Split the data into target and features
X = df.drop(['customerID', 'Churn'], 1)
y = df['Churn']

#Split the data into test and train dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)




# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here

#Replace missing values in test and train, column TotalCharges
X_train['TotalCharges'] = X_train['TotalCharges'].replace(' ', np.nan)
X_test['TotalCharges'] = X_test['TotalCharges'].replace(' ', np.nan)

#Change dtype of TotalCharges to float
X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)
X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)

#Fill in missing values with mean of TotalCharges
X_train['TotalCharges'] = X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean())
X_test['TotalCharges'] = X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean())

#Check for missing values in the dataset
print(X_train.isnull().sum())

#Initialize LabelEncoder
le = LabelEncoder()

#Split numerical and categorical features
num_train = X_train.select_dtypes(include = np.number)
cat_train = X_train.select_dtypes(exclude = np.number)
num_test = X_test.select_dtypes(include = np.number)
cat_test = X_test.select_dtypes(exclude = np.number)

#Label encode categorical features of X_train and X_test
for column in list(cat_train):
    cat_train[column] = le.fit_transform(cat_train[column])

for column in list(cat_test):
    cat_test[column] = le.fit_transform(cat_test[column])

#Combine numrical and categorical features
X_train = pd.concat([num_train, cat_train], 1)
X_test = pd.concat([num_test, cat_test], 1)


#Relabel output columns
y_train = y_train.replace({'No' : 0, 'Yes' : 1})
y_test = y_test.replace({'No' : 0, 'Yes' : 1})









# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here

#View X_train, X_test, y_train, y_test
print(X_train.head())
print('='*20)
print(X_test.head())
print('='*20)
print(y_train.head())
print('='*20)
print(y_test.head())

#Initialize an adaboost classifier
ada_model = AdaBoostClassifier(random_state = 0)

#Fit model on training data
ada_model.fit(X_train, y_train)

#Make prediction from the model using X_test
y_pred = ada_model.predict(X_test)

#Find accuracy score between y_test and y_pred
ada_score = accuracy_score(y_test, y_pred)
print('The accuracy score for AdaBoost is {}.'.format(round(ada_score, 3)))

#Find the confusion matrix of the model
ada_cm = confusion_matrix(y_test, y_pred)
print(ada_cm)

#Find the classification report of the model
ada_cr = classification_report(y_test, y_pred)
print(ada_cr)



# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here

#Initialize a XG Boost classifier
xgb_model = XGBClassifier(random_state = 0)

#Fit model on training data
xgb_model.fit(X_train, y_train)

#Make prediction from the model using X_test
y_pred = xgb_model.predict(X_test)

#Find accuracy score between y_test and y_pred
xgb_score = accuracy_score(y_test, y_pred)
print('The accuracy score for XGBoost is {}.'.format(round(xgb_score, 3)))

#Find the confusion matrix of the model
xgb_cm = confusion_matrix(y_test, y_pred)
print(xgb_cm)

#Find the classification report of the model
xgb_cr = classification_report(y_test, y_pred)
print(xgb_cr)

#Initialize GridSearch CV
clf_model = GridSearchCV(estimator = xgb_model, param_grid = parameters)

#Fit model on training data
clf_model.fit(X_train, y_train)

#Make prediction from the model using X_test
y_pred = clf_model.predict(X_test)

#Find accuracy score between y_test and y_pred
clf_score = accuracy_score(y_test, y_pred)
print('The accuracy score for XGBoost after gridsearch is {}.'.format(round(clf_score, 3)))

#Find the confusion matrix of the model
clf_cm = confusion_matrix(y_test, y_pred)
print(clf_cm)

#Find the classification report of the model
clf_cr = classification_report(y_test, y_pred)
print(clf_cr)









