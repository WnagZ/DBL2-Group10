import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols, logit
import statsmodels.api as sm
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

# Load the PAS dataset
pas_data = pd.read_csv('../data/PAS_data.csv', low_memory=False)

# Verify column names
print("PAS Data Columns:", pas_data.columns)

# Replace -1 with NaN to handle missing values appropriately
pas_data.replace(-1, np.nan, inplace=True)

# Filter female respondents
female_pas_data = pas_data[pas_data['XQ135r_Male'] == False].copy()
print("Female PAS Data:\n", female_pas_data.head())

# Analysis of Q60 responses for female respondents
female_q60_responses = female_pas_data['Q60'].dropna()

# Count the number of valid responses
valid_responses_count_q60 = female_q60_responses.count()
print(f"Number of valid responses in Q60 for female respondents: {valid_responses_count_q60}")

# Print the summary of Q60 responses for female respondents
print("Summary of Q60 responses for female respondents:\n", female_q60_responses.describe())

# Plot the distribution of Q60 responses for female respondents
plt.figure(figsize=(10, 6))
sns.countplot(x=female_q60_responses.astype(int), order=[1, 2, 3, 4, 5])
plt.title('Distribution of Q60 Responses for Female Respondents')
plt.xlabel('Q60: How good a job do you think the police in this area are doing?')
plt.ylabel('Count')
plt.show()

# Analysis of Q61 responses for female respondents
female_q61_responses = female_pas_data['Q61'].dropna()

# Count the number of valid responses
valid_responses_count_q61 = female_q61_responses.count()
print(f"Number of valid responses in Q61 for female respondents: {valid_responses_count_q61}")

# Print the summary of Q61 responses for female respondents
print("Summary of Q61 responses for female respondents:\n", female_q61_responses.describe())

# Plot the distribution of Q61 responses for female respondents
plt.figure(figsize=(10, 6))
sns.countplot(x=female_q61_responses.astype(int), order=[1, 2, 3, 4, 5])
plt.title('Distribution of Q61 Responses for Female Respondents')
plt.xlabel('Q61: How good a job do you think the police in London as a whole are doing?')
plt.ylabel('Count')
plt.show()

# Analysis of Q79J responses for female respondents
female_q79j_responses = female_pas_data['Q79J'].dropna()

# Count the number of valid responses
valid_responses_count_q79j = female_q79j_responses.count()
print(f"Number of valid responses in Q79J for female respondents: {valid_responses_count_q79j}")

# Print the summary of Q79J responses for female respondents
print("Summary of Q79J responses for female respondents:\n", female_q79j_responses.describe())

# Plot the distribution of Q79J responses for female respondents
plt.figure(figsize=(10, 6))
sns.countplot(x=female_q79j_responses.astype(int), order=[1, 2, 3, 4, 5, 6, 7])
plt.title('Distribution of Q79J Responses for Female Respondents')
plt.xlabel('Q79J: How well do you think the police respond to violence against women and girls?')
plt.ylabel('Count')
plt.show()

# Chi-Square Test for Independence between Gender and Q60, Q61, Q79J
contingency_table_q60 = pd.crosstab(pas_data['Q60'].dropna(), pas_data['XQ135r_Male'].dropna())
chi2_q60, p_q60, dof_q60, expected_q60 = chi2_contingency(contingency_table_q60)
print(f"Chi-Square Test for Q60: chi2 = {chi2_q60}, p = {p_q60}")

contingency_table_q61 = pd.crosstab(pas_data['Q61'].dropna(), pas_data['XQ135r_Male'].dropna())
chi2_q61, p_q61, dof_q61, expected_q61 = chi2_contingency(contingency_table_q61)
print(f"Chi-Square Test for Q61: chi2 = {chi2_q61}, p = {p_q61}")

contingency_table_q79j = pd.crosstab(pas_data['Q79J'].dropna(), pas_data['XQ135r_Male'].dropna())
chi2_q79j, p_q79j, dof_q79j, expected_q79j = chi2_contingency(contingency_table_q79j)
print(f"Chi-Square Test for Q79J: chi2 = {chi2_q79j}, p = {p_q79j}")

# Chi-Square Test for Independence between Gender and Q3H
contingency_table_q3h = pd.crosstab(pas_data['Q3H'].dropna(), pas_data['XQ135r_Male'].dropna())
chi2_q3h, p_q3h, dof_q3h, expected_q3h = chi2_contingency(contingency_table_q3h)
print(f"Chi-Square Test for Q3H: chi2 = {chi2_q3h}, p = {p_q3h}")

# ANOVA for Trust in Police (Q3C) across different Boroughs (C2)
anova_model = ols('Q3C ~ C2', data=female_pas_data).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# Correlation Analysis for selected variables
correlation_matrix = female_pas_data[['Q3C', 'Q13', 'Q15', 'Q39A_2', 'Q60', 'Q61', 'Q79J']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Box Plot for Q15 (Worry about Anti-Social Behavior)
plt.figure(figsize=(10, 6))
sns.boxplot(x=female_pas_data['Q15'].dropna().astype(int))
plt.title('Box Plot for Q15 (Worry about Anti-Social Behavior) Among Female Respondents')
plt.xlabel('Q15: Worry about Anti-Social Behavior')
plt.show()

# Bar Plot for Q13 (Worry about Crime)
plt.figure(figsize=(10, 6))
sns.countplot(x=female_pas_data['Q13'].dropna().astype(int), order=[1, 2, 3, 4])
plt.title('Distribution of Q13 (Worry about Crime) for Female Respondents')
plt.xlabel('Q13: Worry about Crime')
plt.ylabel('Count')
plt.show()

# Box Plot for Trust in Police (Q3C) Across Different Boroughs
plt.figure(figsize=(12, 8))
sns.boxplot(x='C2', y='Q3C', data=female_pas_data)
plt.title('Trust in Police (Q3C) Across Different Boroughs')
plt.xlabel('Borough')
plt.ylabel('Trust in Police (Q3C)')
plt.xticks(rotation=90)
plt.show()

# Line Chart for Monthly Average Trust in Police (Q3C) for Female Respondents
female_pas_data.loc[:, 'datetime'] = pd.to_datetime(female_pas_data['datetime'])
monthly_trust = female_pas_data.set_index('datetime').resample('MS')['Q3C'].mean()
plt.figure(figsize=(14, 7))
monthly_trust.plot()
plt.title('Monthly Average Trust in Police (Q3C) for Female Respondents')
plt.xlabel('Date')
plt.ylabel('Average Trust in Police (Q3C)')
plt.show()

# Analysis of Feeling Unsafe and Trust in Police (Q61)
# Assuming feeling unsafe is indicated by Q14A (Your personal experience with crime)
unsafe_responses = female_pas_data[female_pas_data['Q14A'] == True]['Q61'].dropna()
safe_responses = female_pas_data[female_pas_data['Q14A'] == False]['Q61'].dropna()

if not unsafe_responses.empty and not safe_responses.empty:
    # T-test for feeling unsafe vs. safe on trust in police (Q61)
    t_stat_unsafe, p_val_unsafe = stats.ttest_ind(unsafe_responses, safe_responses, nan_policy='omit')
    print(f"T-test for Feeling Unsafe (Q14A) vs. Safe on Trust in Police (Q61): t_stat = {t_stat_unsafe}, p_val = {p_val_unsafe}")
else:
    print("No data available for one of the groups (unsafe or safe).")

# Plot the distribution of Q61 responses for unsafe vs. safe female respondents
if not unsafe_responses.empty:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=unsafe_responses.astype(int), order=[1, 2, 3, 4, 5])
    plt.title('Distribution of Q61 Responses for Unsafe Female Respondents')
    plt.xlabel('Q61: How good a job do you think the police in London as a whole are doing?')
    plt.ylabel('Count')
    plt.show()

if not safe_responses.empty:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=safe_responses.astype(int), order=[1, 2, 3, 4, 5])
    plt.title('Distribution of Q61 Responses for Safe Female Respondents')
    plt.xlabel('Q61: How good a job do you think the police in London as a whole are doing?')
    plt.ylabel('Count')
    plt.show()

# Kruskal-Wallis H Test: Compare trust in police (Q61) across different boroughs
groups = [group['Q61'].dropna().values for name, group in female_pas_data.groupby('C2')]
stat, p = kruskal(*groups)
print(f'Kruskal-Wallis H Test: stat = {stat}, p = {p}')

# Mann-Whitney U Test: Compare trust in police (Q61) between female and male respondents
male_pas_data = pas_data[pas_data['XQ135r_Male'] == True]
stat, p = mannwhitneyu(female_pas_data['Q61'].dropna(), male_pas_data['Q61'].dropna())
print(f'Mann-Whitney U Test: stat = {stat}, p = {p}')

# Analysis of NQ135BD responses for female and male respondents
female_nq135bd_responses = female_pas_data['NQ135BD'].dropna()
male_nq135bd_responses = male_pas_data['NQ135BD'].dropna()

# Count the number of valid responses
valid_responses_count_nq135bd_female = female_nq135bd_responses.count()
valid_responses_count_nq135bd_male = male_nq135bd_responses.count()
print(f"Number of valid responses in NQ135BD for female respondents: {valid_responses_count_nq135bd_female}")
print(f"Number of valid responses in NQ135BD for male respondents: {valid_responses_count_nq135bd_male}")

# Plot the distribution of NQ135BD responses for female and male respondents
plt.figure(figsize=(12, 6))
sns.countplot(x=female_nq135bd_responses.astype(int), order=[1, 2, 3, 4, 5], label='Female', color='blue', alpha=0.6)
sns.countplot(x=male_nq135bd_responses.astype(int), order=[1, 2, 3, 4, 5], label='Male', color='orange', alpha=0.6)
plt.title('Distribution of NQ135BD (Trust in Police) Responses by Gender')
plt.xlabel('NQ135BD: Trust in Police (1=Strongly agree, 5=Strongly disagree)')
plt.ylabel('Count')
plt.legend()
plt.show()

# Calculate and print the percentage distribution
female_nq135bd_percent = female_nq135bd_responses.value_counts(normalize=True) * 100
male_nq135bd_percent = male_nq135bd_responses.value_counts(normalize=True) * 100
print(f"Percentage distribution of NQ135BD responses for female respondents:\n{female_nq135bd_percent}")
print(f"Percentage distribution of NQ135BD responses for male respondents:\n{male_nq135bd_percent}")
# Select the first 20 relevant columns (questions only)
question_columns = [col for col in female_pas_data.columns if col.startswith('Q') or col.startswith('NQ')][:20]
female_question_data = female_pas_data[question_columns]

# Compute the correlation matrix
correlation_matrix = female_question_data.corr()

# Print the correlation matrix
print("Correlation Matrix:\n", correlation_matrix)

# Plot the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for First 20 Survey Questions (Female Respondents)')
plt.show()

# Continue with next set of questions
question_columns = [col for col in female_pas_data.columns if col.startswith('Q') or col.startswith('NQ')][20:40]
female_question_data = female_pas_data[question_columns]

# Compute the correlation matrix
correlation_matrix = female_question_data.corr()

# Print the correlation matrix
print("Correlation Matrix:\n", correlation_matrix)

# Plot the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for Next 20 Survey Questions (Female Respondents)')
plt.show()

# Filter male respondents
male_pas_data = pas_data[pas_data['XQ135r_Male'] == True]

# Break down Q61 responses by education level for both genders
education_levels = ['NQ146_O-levels/CSE/GCSEs', 'NQ146_ONC, OND or City and Guilds',
                    'NQ146_Trade apprenticeship', 'NQ146_University Degree (Bachelor degree)',
                    'NQ146_Post-graduate degree or qualification']

for level in education_levels:
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Q61', hue=level, data=female_pas_data)
    plt.title(f'Distribution of Q61 Responses for Female Respondents by Education Level ({level})')
    plt.xlabel('Q61: How good a job do you think the police in London as a whole are doing?')
    plt.ylabel('Count')
    plt.legend(title=level)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Q61', hue=level, data=male_pas_data)
    plt.title(f'Distribution of Q61 Responses for Male Respondents by Education Level ({level})')
    plt.xlabel('Q61: How good a job do you think the police in London as a whole are doing?')
    plt.ylabel('Count')
    plt.legend(title=level)
    plt.show()

# Correlation analysis for Q61 with other questions
questions_of_interest = ['Q3C', 'Q13', 'Q15', 'Q39A_2', 'Q60', 'Q79J']
correlation_matrix = female_pas_data[questions_of_interest + ['Q61']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Selected Questions (Female Respondents)')
plt.show()
