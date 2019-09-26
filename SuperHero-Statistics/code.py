# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path

#Code starts here 
data = pd.read_csv(path)
data['Gender'].replace('-', 'Agender', inplace = True)
gender_count = data['Gender'].value_counts()

gender_count.plot(kind = 'bar')
plt.xlabel('Gender Type')
plt.ylabel('No. of SuperHeroes/Villians')
plt.title('SuperHeroes/Villains by Gender')
plt.show()





# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
labels = ['Good', 'Bad', 'Neutral']
plt.pie(alignment, explode = (0.1, 0.1, 0.1), labels = labels)
plt.title('Character Alignment')
plt.show()


# --------------
#Code starts here
sc_df = data[['Strength', 'Combat']]
sc_covariance = sc_df['Strength'].cov(sc_df['Combat'])
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()
sc_pearson = sc_covariance/(sc_combat*sc_strength)

ic_df = data[['Intelligence', 'Combat']]
ic_covariance =ic_df['Intelligence'].cov(ic_df['Combat'])
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()
ic_pearson = ic_covariance/(ic_combat*ic_intelligence)

print(sc_pearson)
print(ic_pearson)


# --------------
#Code starts here
total_high = np.quantile(data['Total'], 0.99)
super_best = data[data['Total'] > total_high]

super_best_names = list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(1,3,  figsize = (20,10))

super_best['Intelligence'].plot(kind = 'box', ax = ax_1)
ax_1.set_title('Intelligence')

super_best['Speed'].plot(kind = 'box', ax = ax_2)
ax_2.set_title('Speed')

super_best['Power'].plot(kind = 'box', ax = ax_3)
ax_3.set_title('Power')

plt.show()
print('Intelligence has the largest spread of the three attributes shown.')


