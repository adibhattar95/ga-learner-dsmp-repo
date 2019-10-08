# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
fico = len(df[df['fico'] > 700])
total = len(df)
p_a = fico/total
print(p_a)

debt_consolidation = len(df[df['purpose'] == 'debt_consolidation'])
p_b = debt_consolidation/total
print(p_b)

df1 = pd.DataFrame(df[df['purpose'] == 'debt_consolidation'])

fico_and_debt_consolidation = len(df[(df['fico'] > 700) & (df['purpose'] == 'debt_consolidation')])
p_a_b = fico_and_debt_consolidation/fico
print(p_a_b)

result = p_a_b == p_a
print(result)



# code ends here


# --------------
# code starts here
total = len(df)
paid_back_loan = len(df[df['paid.back.loan'] == 'Yes'])
prob_lp = paid_back_loan/total
print(prob_lp)

credit_policy = len(df[df['credit.policy'] == 'Yes'])
prob_cs = credit_policy/total
print(prob_cs)

new_df = pd.DataFrame(df[df['paid.back.loan'] == 'Yes'])
new_credit_policy = len(new_df[new_df['credit.policy'] == 'Yes'])
prob_pd_cs = new_credit_policy/paid_back_loan
print(prob_pd_cs)

bayes = (prob_pd_cs*prob_lp)/prob_cs
print(bayes)




# code ends here


# --------------
# code starts here
df['purpose'].value_counts().plot(kind = 'bar')
plt.show()
df1 = pd.DataFrame(df[df['paid.back.loan'] == 'No'])
df1['purpose'].value_counts().plot(kind = 'bar')
plt.show()



# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
print(inst_median)
inst_mean = df['installment'].mean()
print(inst_mean)
plt.hist(df['installment'], bins = 20)
plt.show()
plt.hist(df['log.annual.inc'], bins = 20)
plt.show()



# code ends here


