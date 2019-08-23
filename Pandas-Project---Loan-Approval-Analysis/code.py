# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 

bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)
 



# code starts here






# code ends here


# --------------
# code starts here
banks = bank.drop(columns = "Loan_ID")

null_values = banks.isnull().sum()

print(null_values)

bank_mode = banks.mode().iloc[0]

banks.fillna(bank_mode, inplace = True)
print(banks.isnull().sum())


#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks, index = ["Gender", "Married", "Self_Employed"],
values = "LoanAmount")

print(avg_loan_amount)



# code ends here



# --------------
# code starts here

loan_approved_se = len(banks[(banks["Self_Employed"] == "Yes")
 & (banks["Loan_Status"] == "Y")])

loan_approved_nse = len(banks[(banks["Self_Employed"] == "No")
 & (banks["Loan_Status"] == "Y")])

loan_status = len(banks["Loan_Status"])

percentage_se = (loan_approved_se/loan_status)*100
percentage_nse = (loan_approved_nse/loan_status)*100

print(percentage_se)
print(percentage_nse)





# code ends here


# --------------
# code starts here
loan_term = banks["Loan_Amount_Term"].apply(lambda x: int(x)/12)

big_loan_term = len(loan_term[loan_term >= 25])
print(big_loan_term)





# code ends here


# --------------
# code starts here

loan_groupby = banks.groupby("Loan_Status")

loan_groupby = banks.groupby("Loan_Status")[["ApplicantIncome", "Credit_History"]]
mean_values = loan_groupby.mean()
print(mean_values)





# code ends here


