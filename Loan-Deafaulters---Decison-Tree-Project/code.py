# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
#Load the dataset
data = pd.read_csv(path)

#Split the DataFrame into Featurs and Target
X = data.drop(['customer.id', 'paid.back.loan'], 1)
y = data['paid.back.loan']

#Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
#View value counts of target variable
fully_paid = y_train.value_counts()

#View bar graph of fully_paid
fully_paid.plot(kind = 'bar')
plt.xlabel('Load Paid Back')
plt.ylabel('No. of Customers')
plt.title('Graph of Customers Paying Back loan or not')
plt.show()

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
#Remove '%' from interest rate column and change dtype to float
X_train['int.rate'] = X_train['int.rate'].apply(lambda x: x.replace('%', ''))
X_train['int.rate'] = X_train['int.rate'].astype(float)
X_train['int.rate'] = X_train['int.rate']/100

X_test['int.rate'] = X_test['int.rate'].apply(lambda x: x.replace('%', ''))
X_test['int.rate'] = X_test['int.rate'].astype(float)
X_test['int.rate'] = X_test['int.rate']/100

#Seperate numerical and categorical features
num_df = X_train.select_dtypes(include = np.number)
cat_df = X_train.select_dtypes(exclude = np.number)

# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
#List of numerical features
cols = list(num_df)

#View boxplot for numerical features and target variable
fig, axes = plt.subplots(nrows = 9, ncols = 1)

plt.figure(figsize = (10,10))
for i in range(0, 9):
    sns.boxplot(x = y_train, y = num_df[cols[i]], ax = axes[i])
plt.tight_layout()
plt.show()


# Code ends here


# --------------
# Code starts here
#List of categorical features
cols = list(cat_df)

#View countplot between categorical features and target variable
plt.figure(figsize = (20, 20))
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 20))

for i in range(0, 2):
    for j in range(0, 2):
        sns.countplot(x = X_train[cols[i*2 + j]], hue = y_train, ax = axes[i, j])
        fig.tight_layout()
plt.show()

# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

#Code starts here

#Looping through categorical columns
for col in cat_df.columns:
    
    #Filling null values with 'NA'
    X_train[col].fillna('NA',inplace=True)
    
    #Initalising a label encoder object
    le=LabelEncoder()
    
    #Fitting and transforming the column in X_train with 'le'
    X_train[col]=le.fit_transform(X_train[col]) 
    
    #Filling null values with 'NA'
    X_test[col].fillna('NA',inplace=True)
    
    #Fitting the column in X_test with 'le'
    X_test[col]=le.transform(X_test[col]) 

# Replacing the values of y_train
y_train.replace({'No':0,'Yes':1},inplace=True)

# Replacing the values of y_test
y_test.replace({'No':0,'Yes':1},inplace=True)

#Initialising 'Decision Tree' model    
model=DecisionTreeClassifier(random_state=0)

#Training the 'Decision Tree' model
model.fit(X_train, y_train)

#Finding the accuracy of 'Decision Tree' model
acc=model.score(X_test, y_test)

#Printing the accuracy
print(acc)

#Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
#Initialize a new DecisionTreeClassifier
model_2 = DecisionTreeClassifier(random_state = 0)

#Create a GridSearchCV to find optimal model
p_tree = GridSearchCV(estimator = model_2, param_grid = parameter_grid, cv = 5)

#Fit training data on GridSearchCV
p_tree.fit(X_train, y_train)

#Find accuracy of optimal model
acc_2 = p_tree.score(X_test, y_test)
print("The accuracy of our model using GridSearchCV is {}.".format(acc_2))



# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
#View how the decision tree looks like
dot_data = export_graphviz(decision_tree = p_tree.best_estimator_, out_file = None, feature_names = X.columns, filled = True, class_names = ['load_paid_back_yes', 'load_paid_back_no'])

graph_big = pydotplus.graph_from_dot_data(dot_data)



# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


