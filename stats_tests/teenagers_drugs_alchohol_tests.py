import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal
import numpy as np

# Load the datasets
pas_data = pd.read_csv('../data/PAS_data.csv', low_memory=False)

# Replace -1 with NaN to handle missing values appropriately
pas_data.replace(-1, np.nan, inplace=True)

# Filter the dataset for female respondents
female_pas_data = pas_data[pas_data['XQ135r_Male'] == False]

# Select the relevant columns for analysis
relevant_columns = ['Q3C', 'Q60', 'Q61', 'ZQ10F']
female_relevant_data = female_pas_data[relevant_columns]

# Compute the cross-tabulation matrices for each question
cross_tab_q3c = pd.crosstab(female_pas_data['Q3C'].dropna(), female_pas_data['ZQ10F'].dropna())
cross_tab_q60 = pd.crosstab(female_pas_data['Q60'].dropna(), female_pas_data['ZQ10F'].dropna())
cross_tab_q61 = pd.crosstab(female_pas_data['Q61'].dropna(), female_pas_data['ZQ10F'].dropna())

# Perform Chi-Square tests
chi2_q3c, p_q3c, dof_q3c, expected_q3c = chi2_contingency(cross_tab_q3c)
chi2_q60, p_q60, dof_q60, expected_q60 = chi2_contingency(cross_tab_q60)
chi2_q61, p_q61, dof_q61, expected_q61 = chi2_contingency(cross_tab_q61)

print(f"Chi-Square Test for Q3C with ZQ10F: chi2 = {chi2_q3c}, p = {p_q3c}")
print(f"Chi-Square Test for Q60 with ZQ10F: chi2 = {chi2_q60}, p = {p_q60}")
print(f"Chi-Square Test for Q61 with ZQ10F: chi2 = {chi2_q61}, p = {p_q61}")

# Kruskal-Wallis H Test to see if there's a significant difference in trust (Q3C) based on ZQ10F
kruskal_stat_q3c, kruskal_p_q3c = kruskal(
    female_pas_data[female_pas_data['ZQ10F'] == 1]['Q3C'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 2]['Q3C'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 3]['Q3C'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 4]['Q3C'].dropna()
)
print(f"Kruskal-Wallis H Test for Q3C based on ZQ10F: stat = {kruskal_stat_q3c}, p = {kruskal_p_q3c}")

# Kruskal-Wallis H Test to see if there's a significant difference in Q60 based on ZQ10F
kruskal_stat_q60, kruskal_p_q60 = kruskal(
    female_pas_data[female_pas_data['ZQ10F'] == 1]['Q60'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 2]['Q60'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 3]['Q60'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 4]['Q60'].dropna()
)
print(f"Kruskal-Wallis H Test for Q60 based on ZQ10F: stat = {kruskal_stat_q60}, p = {kruskal_p_q60}")

# Kruskal-Wallis H Test to see if there's a significant difference in Q61 based on ZQ10F
kruskal_stat_q61, kruskal_p_q61 = kruskal(
    female_pas_data[female_pas_data['ZQ10F'] == 1]['Q61'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 2]['Q61'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 3]['Q61'].dropna(),
    female_pas_data[female_pas_data['ZQ10F'] == 4]['Q61'].dropna()
)
print(f"Kruskal-Wallis H Test for Q61 based on ZQ10F: stat = {kruskal_stat_q61}, p = {kruskal_p_q61}")

# Plotting the distribution of Q3C, Q60, and Q61 based on ZQ10F
plt.figure(figsize=(12, 6))
sns.countplot(x='Q3C', hue='ZQ10F', data=female_pas_data)
plt.title('Distribution of Trust (Q3C) based on Public Drug/Alcohol Usage (ZQ10F)')
plt.xlabel('Q3C: People in this neighborhood can be trusted')
plt.ylabel('Count')
plt.legend(title='ZQ10F: Public Drug/Alcohol Usage')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Q60', hue='ZQ10F', data=female_pas_data)
plt.title('Distribution of Q60 based on Public Drug/Alcohol Usage (ZQ10F)')
plt.xlabel('Q60: How good a job do you think the police in this area are doing?')
plt.ylabel('Count')
plt.legend(title='ZQ10F: Public Drug/Alcohol Usage')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Q61', hue='ZQ10F', data=female_pas_data)
plt.title('Distribution of Q61 based on Public Drug/Alcohol Usage (ZQ10F)')
plt.xlabel('Q61: How good a job do you think the police in London as a whole are doing?')
plt.ylabel('Count')
plt.legend(title='ZQ10F: Public Drug/Alcohol Usage')
plt.show()
