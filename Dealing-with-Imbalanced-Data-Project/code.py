# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here

#Load and View the dataset
df = pd.read_csv(path)
print(df.head())
print("="*20)
print(df.info())

#Remove $ and , from columns
columns = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']

for column in columns:
    df[column] = df[column].astype('str')
for column in columns:
    df[column] = df[column].apply(lambda x : x.replace('$', ''))

for column in columns:
    df[column] = df[column].apply(lambda x : x.replace(',', ''))

#Split dataset in target and features
X = df.drop(['CLAIM_FLAG'], 1)
y = df['CLAIM_FLAG']

#View count of target variable
count = y.value_counts()
print(count)

#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

# Code ends here


# --------------
# Code starts here

#Make object features as float features
for column in columns:
    X_train[column] = X_train[column].astype('float')

for column in columns:
    X_test[column] = X_test[column].astype('float')

#Check for null values
print(X_train.isnull().sum())
print(X_test.isnull().sum())


# Code ends here


# --------------
# Code starts here
#Drop nan values from features

X_train = X_train[X_train['YOJ'].notna()]
X_train = X_train[X_train['OCCUPATION'].notna()]
X_test = X_test[X_test['YOJ'].notna()]
X_test = X_test[X_test['OCCUPATION'].notna()]

#Updating index of y_train and y_test
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

#Replacing nan values in X_train
columns2 = ['AGE', 'CAR_AGE', 'INCOME', 'HOME_VAL']
for column in columns2:
    X_train[column].fillna(X_train[column].mean(), inplace = True)

for column in columns2:
    X_test[column].fillna(X_test[column].mean(), inplace = True)



# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here

#Initialize labelencoder
le = LabelEncoder()

#Label encode categorical columns
for column in columns:
    X_train[column] = le.fit_transform(X_train[column].astype(str))

for column in columns:
    X_test[column] = le.fit_transform(X_test[column].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

#Initiate a logistic regression model
model = LogisticRegression(random_state = 6)

#Fit model on training data
model.fit(X_train, y_train)

#Make prediction using X_test
y_pred = model.predict(X_test)

#Calculate accuracy score
score = accuracy_score(y_test, y_pred)
print(score)



# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

#Initiate smote
smote = SMOTE(random_state = 9)

#Fit smote on training data
X_train, y_train = smote.fit_sample(X_train, y_train)

#Initiate standard scaler
scaler = StandardScaler()

#Standardize features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# Code ends here


# --------------
# Code Starts here

#Initiate a logistic regression model
model = LogisticRegression()

#Fit model on training data
model.fit(X_train, y_train)

#Make prediction using X_test
y_pred = model.predict(X_test)

#Calculate accuracy score
score = accuracy_score(y_test, y_pred)
print(score)


# Code ends here


