# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here

#Load the Dataset
df = pd.read_csv(path)

#View the dataset
print(df.head(5))

#Split Features and Target Variable
X = df.drop(['attr1089'], 1)
y = df['attr1089']

#Split into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

#Initialize minmaxscaler
scaler = MinMaxScaler()

#Standardize X_train and X_test
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#Initialize a logistic regression model
lr = LogisticRegression()

#Fit the model on training data
lr.fit(X_train, y_train)

#Make prediction using X_test
y_pred = lr.predict(X_test)

#View roc_score
roc_score = roc_auc_score(y_test, y_pred)
print(roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier

#Initialize a Decision Tree
dt = DecisionTreeClassifier(random_state = 4)

#Fit the model on training data
dt.fit(X_train, y_train)

#Make prediction using X_test
y_pred = dt.predict(X_test)

#View roc_score
roc_score = roc_auc_score(y_test, y_pred)
print(roc_score)
print(classification_report(y_test, y_pred))


# --------------
from sklearn.ensemble import RandomForestClassifier


# Code strats here

#Initiate a random forest
rfc = RandomForestClassifier(random_state = 4)

#Fit the model on training data
rfc.fit(X_train, y_train)

#Make prediction using X_test
y_pred = rfc.predict(X_test)

#View roc_score
roc_score = roc_auc_score(y_test, y_pred)
print(roc_score)
print(classification_report(y_test, y_pred))

# Code ends here


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code starts here

#Initiate a bagging classifier
bagging_clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 100, max_samples = 100, random_state = 0)

#Fit the model on training data
bagging_clf.fit(X_train, y_train)

#View the accuracy score of the model
score_bagging = bagging_clf.score(X_test, y_test)
print(score_bagging)
y_pred = bagging_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here

#Initiate a voting classifier
voting_clf_hard = VotingClassifier(estimators = model_list, voting = 'hard')

#Fit the model on training data
voting_clf_hard.fit(X_train, y_train)

#View the accuracy score of the model
hard_voting_score = voting_clf_hard.score(X_test, y_test)
print(hard_voting_score)
y_pred = voting_clf_hard.predict(X_test)
print(classification_report(y_test, y_pred))


# Code ends here


