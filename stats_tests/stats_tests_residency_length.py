import pandas as pd
import scipy.stats as stats



import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# Load the PAS dataset
data = pd.read_csv('../data/PAS_data.csv', low_memory=False)


# Dropping missing values and converting to integer
residence_length = data['Q1'].dropna().astype(int)
trust_levels = data['NQ135BD'].dropna().astype(int)

# Cross-tabulate residence length with trust levels
cross_tab = pd.crosstab(residence_length, trust_levels)

# Perform Chi-Square test
chi2, p, dof, ex = stats.chi2_contingency(cross_tab)
print(f"Chi-Square Test for Residence Length vs Trust in Police: chi2 = {chi2}, p = {p}")

# Perform Kruskal-Wallis H Test
kruskal_stat, kruskal_p = stats.kruskal(*[data[data['Q1'] == i]['NQ135BD'] for i in range(1, 9)])
print(f"Kruskal-Wallis H Test for Residence Length vs Trust in Police: H = {kruskal_stat}, p = {kruskal_p}")

# Plotting the results (if necessary)
import matplotlib.pyplot as plt

# Plotting as a grouped bar chart
plt.figure(figsize=(12, 8))
cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
cross_tab_norm.plot(kind='bar', colormap='viridis', ax=plt.gca())
plt.title('Trust in Police vs Length of Residence')
plt.xlabel('Length of Residence')
plt.ylabel('Percentage of Respondents')
plt.legend(title='Trust in Police', labels=[
    'Strongly agree',
    'Tend to agree',
    'Neither agree nor disagree',
    'Tend to disagree',
    'Strongly disagree'
])
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7], labels=[
    'Less than 12 months',
    '12 months but less than 2 years',
    '2 years but less than 3 years',
    '3 years but less than 5 years',
    '5 years but less than 10 years',
    '10 years but less than 20 years',
    '20 years but less than 30 years',
    '30 years or more'
], rotation=45)
plt.show()


# Ordinal Logistic Regression
def ordinal_logistic_regression(data):
    model = smf.mnlogit('NQ135BD ~ Q1', data=data).fit()
    print(model.summary())
    return model

# Correlation Analysis
def correlation_analysis(data):
    corr, p_value = spearmanr(data['Q1'].dropna(), data['NQ135BD'].dropna())
    print(f"Spearman Correlation: {corr}, p-value: {p_value}")

# Cross-tabulation with Percentages
def cross_tabulation_percentages(data):
    cross_tab = pd.crosstab(data['Q1'], data['NQ135BD'], normalize='index') * 100
    print(cross_tab)
    cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Trust in Police vs Length of Residence')
    plt.xlabel('Length of Residence')
    plt.ylabel('Percentage of Respondents')
    plt.legend(title='Trust in Police')
    plt.xticks(rotation=45)
    plt.show()

# Descriptive Statistics
def descriptive_statistics(data):
    desc_stats = data.groupby('Q1')['NQ135BD'].describe()
    print(desc_stats)


print("\nCorrelation Analysis:")
correlation_analysis(data)

print("\nCross-tabulation with Percentages:")
cross_tabulation_percentages(data)

print("\nDescriptive Statistics:")
descriptive_statistics(data)
