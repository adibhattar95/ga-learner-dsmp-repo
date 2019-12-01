# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here

#Load and view the dataset
df = pd.read_csv(path)
print(df.head())

#Create X and y variables
X = df.drop(['insuranceclaim'], 1)
y = df['insuranceclaim']

#Create Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 6)




# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here

#Check for outliers in BMI
plt.figure(figsize = (10,10))
X_train['bmi'].plot(kind = 'box')
plt.title('Boxplot of BMI')
plt.show()

#Quantile of BMI
q_value = X_train['bmi'].quantile(0.95)
print(q_value)

#Value Counts of y_train
print(y_train.value_counts())

# Code ends here


# --------------
# Code starts here

#Correlation between features in X_train
relation = X_train.corr()
print(relation)

#Pairplot for features in X_train
sns.pairplot(X_train)
plt.show()


# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here

#List of Columns
cols = ['children', 'sex', 'region', 'smoker']

#Create a subplot
fig, axes = plt.subplots(nrows = 2, ncols = 2)

#Countplot for features with target

for i in range(0,2):
    for j in range(0,2):
        col = cols[i*2+j]
        sns.countplot(x = X_train[col], hue = y_train, ax = axes[i, j])


# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here

#Instantiate a Logistic Regression Model
lr = LogisticRegression(random_state = 9)

#Apply GridSearchCV on logistic regression model 
grid = GridSearchCV(estimator = lr, param_grid = parameters)

#Fit the model on training data
grid.fit(X_train, y_train)

#Calculate y_pred from X_Test
y_pred = grid.predict(X_test)

#Check accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here

#Calculate roc auc score of the model
score = roc_auc_score(y_test, y_pred)
print(score)

#Predict probability
y_pred_proba = grid.predict_proba(X_test)[:, 1]

#Calculate fpr and tpr
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

#Roc Auc Score of y_test and y_pred_proba
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(roc_auc)

#Plot auc curve
plt.plot(fpr, tpr, label = 'Logistic model, auc =' + str(roc_auc))
plt.show()


# Code ends here


