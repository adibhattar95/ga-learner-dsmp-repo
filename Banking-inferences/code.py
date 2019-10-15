# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data = pd.read_csv(path)

data_sample = data.sample(n = sample_size, random_state = 0)

sample_mean = round(data_sample['installment'].mean(),2)
print("The sample mean of installment is {}.".format(sample_mean))
sample_std = round(data_sample['installment'].std(),2)
print("The sample standard deviation of installment is {}.".format(sample_std))

margin_of_error = z_critical*(sample_std/math.sqrt(sample_size))

confidence_interval = round(sample_mean - margin_of_error,2) , round(sample_mean + margin_of_error,2)
print("The confidence interval at 95% for installment is {}.".format(confidence_interval))

true_mean = round(data['installment'].mean(),2)
print('The true mean for installment is {}.'.format(true_mean))

print("The true mean of installment falls in the range of the confidence interval at 95%.")





# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig, axes = plt.subplots(nrows = 3, ncols = 1)

for i in range(len(sample_size)):
    m = []
    for j in range(1000):
        sample_data = data['installment'].sample(n = sample_size[i])
        sample_mean = sample_data.mean()
        m.append(sample_mean)
    mean_series = pd.Series(m)

    mean_series.plot(kind = 'hist', ax = axes[i])

plt.tight_layout()
plt.xlabel('Installment Amount')

plt.show()
print('Looking at the histogram, it does seem like installment does follow the central limit theorem.')





# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].apply(lambda x: x.strip('%'))

data = data.astype({'int.rate': np.float64}, inplace = True)
data['int.rate'] = data['int.rate'].apply(lambda x: x/100)

z_statistic, p_value = ztest(data[data['purpose'] == 'small_business']['int.rate'], value = data['int.rate'].mean(), alternative = 'larger')

print('The z_statistic for our hypothesis test is {}.'.format(round(z_statistic,2)))
print('The p_value of our hypothesis test is {}.'.format(p_value))
print("As the p_value is less than 0.05, we can reject the null hypothesis and conclude that the interest rate charged for taking out a loan for the purpose of small business is higher as compared to the interest rates for other types of loans.")



# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value = ztest(x1 = data[data['paid.back.loan'] == 'No']['installment'], x2 = data[data['paid.back.loan'] == 'Yes']['installment'])

print('The z_statistic for out hypothesis test is {}.'.format(round(z_statistic,2)))
print('The p_value for our hypothesis test is {}.'.format(p_value))
print('As the p_value is less than 0.05, we can reject the null hypothesis and conclude that there is a difference in installments being paid by loan defaulters and loan non-defaulters.')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan'] == 'Yes']['purpose'].value_counts()
no = data[data['paid.back.loan'] == 'No']['purpose'].value_counts()

observed = pd.concat([yes.transpose(), no.transpose()], axis = 1, keys = ['Yes', 'No'])
print(observed)

chi2, p, dof, ex = chi2_contingency(observed)
print('The chi-squared statistic is {}.'.format(chi2))
print('The p value of the test is {}.'.format(p))
print('The degrees of freedom for the test is {}.'.format(dof))

print('The critical value is {}.'.format(critical_value))
print('As the chi-squared statistic is greater than the critical value, we can reject the null hypothesis and conclude that distribution of purpose for loan defaulters and non defaulters is different.')






