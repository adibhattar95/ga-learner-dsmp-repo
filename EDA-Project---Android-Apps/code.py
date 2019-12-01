# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here

#Load the Dataset
data = pd.read_csv(path)

#Replace NaN values in Rating column will 0
data['Rating'] = data['Rating'].dropna()

#Histogram of Rating Column
fig = plt.figure(figsize = (10, 10))
data['Rating'].plot(kind = 'hist')
plt.xlabel('Rating')
plt.ylabel('No. of Ratings')
plt.title('Distribution of Ratings for Android Apps')
plt.show()

#Removing Ratings greater than 5
data = data[data['Rating'] <= 5]

#Histogram of Rating Column after removing greater than 5
fig = plt.figure(figsize = (10, 10))
data['Rating'].plot(kind = 'hist')
plt.xlabel('Rating')
plt.ylabel('No. of Ratings')
plt.title('Distribution of Ratings for Android Apps')
plt.show()

#Code ends here


# --------------
# code starts here

#Total null values in each column
total_null = data.isnull().sum()

#Percent of null values in each column
percent_null = (total_null/data.isnull().count())

#Concatenate total and percent of null valuies and print it out
missing_data = pd.concat([total_null, percent_null], axis = 1, keys = ['Total', 'Percent'])
print(missing_data)

#Dropping null values
data.dropna(inplace = True)

#Check for null values again
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis = 1, keys = ['Total', 'Percent'])
print(missing_data_1)



# code ends here


# --------------

#Code starts here

#Check for relation between Category and Rating
fig = plt.figure(figsize = (10, 10))
sns.catplot(x = 'Category', y = 'Rating', data = data, kind = 'box', height = 10)
plt.xticks(rotation = 90)
plt.xlabel('Category')
plt.ylabel('Rating')
plt.title('Rating vs Category [BoxPlot]')
plt.show()


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

#See distribution of Installs Column
print(data['Installs'].value_counts())

#Remove , an + from Installs column
data['Installs'] = data['Installs'].apply(lambda x: x.replace(',', ''))
data['Installs'] = data['Installs'].apply(lambda x: x.replace('+', ''))


#Convert Installs datatype to Int
data['Installs'] = data['Installs'].astype(int)

#Creating a label encoder object
le = LabelEncoder()

#Transforming values of Installs Column using Label Encoder
data['Installs'] = le.fit_transform(data['Installs'])

#Plot Regplot between Installs and Rating
fig = plt.figure(figsize = (10, 10))
sns.regplot(x = 'Installs', y = 'Rating', data = data)
plt.xlabel('Installs')
plt.ylabel('Rating')
plt.title('Rating vs Installs [RegPlot]')
plt.show()


#Code ends here



# --------------
#Code starts here

#View distribution of Price Column
print(data['Price'].value_counts())

#Remove $ sign from Price Column
data['Price'] = data['Price'].apply(lambda x: x.replace('$', ''))

#Convert Price Column dtype to float
data['Price'] = data['Price'].astype(float)

#Plot RegPlot between Rating and Price
fig = plt.figure(figsize = (10, 10))
sns.regplot(x = 'Price', y = 'Rating', data = data)
plt.xlabel('Price')
plt.ylabel('Rating')
plt.title('Rating vs Price [RegPlot]')
plt.show()


#Code ends here


# --------------

#Code starts here

#View Unique values of Genre Column
print(data['Genres'].unique())
print('='*50)

#Split Genre Column by ';', and store only first column back in Genre
data['Genres'] = data['Genres'].str.split(';').str[0]

#Grouping Genres by Rating
gr_mean = data[['Genres', 'Rating']].groupby(['Genres'], as_index = False).mean()

#View the statistics of gr_mean
print(gr_mean.describe())
print('='*50)

#Sort gr_mean by Rating
gr_mean = gr_mean.sort_values('Rating')

#Print First and Last value of gr_mean
print(gr_mean.head(1))
print('='*50)
print(gr_mean.tail(1))


#Code ends here


# --------------

#Code starts here

#View Last Updated Column values
print(data['Last Updated'].value_counts())

#Convert Last Updated to datetime format
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

#Max value in Last Updated Column
max_date = data['Last Updated'].max()

#Create a new column Last Updated Days
data['Last Updated Days'] =  (max_date - data['Last Updated']).dt.days

#PLot RegPlot between Last Updated Days and Rating
fig = plt.figure(figsize = (10, 10))
sns.regplot(x = 'Last Updated Days', y = 'Rating', data = data)
plt.xlabel('Last Updated')
plt.ylabel('Rating')
plt.title('Rating vs Last Updated [RegPlot]')
plt.show()

#Code ends here


