import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal, f_oneway
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm

# Load the PAS dataset
pas_data = pd.read_csv('../data/PAS_data.csv', low_memory=False)

# Verify column names
print("PAS Data Columns:", pas_data.columns)

# Replace -1 with NaN to handle missing values appropriately
pas_data.replace(-1, np.nan, inplace=True)

# Ensure the index is unique
pas_data = pas_data.reset_index(drop=True)

# Analysis of NQ135BD responses for trust across different boroughs
nq135bd_responses = pas_data['NQ135BD'].dropna().astype(int)

# Count the number of valid responses
valid_responses_count_nq135bd = nq135bd_responses.count()
print(f"Number of valid responses in NQ135BD: {valid_responses_count_nq135bd}")

# Chi-Square Test for NQ135BD across different Boroughs (C2)
contingency_table_nq135bd = pd.crosstab(pas_data['NQ135BD'].dropna().astype(int), pas_data['C2'].dropna())
chi2_nq135bd, p_nq135bd, dof_nq135bd, expected_nq135bd = chi2_contingency(contingency_table_nq135bd)
print(f"Chi-Square Test for NQ135BD: chi2 = {chi2_nq135bd}, p = {p_nq135bd}")

# Kruskal-Wallis H Test for NQ135BD based on C2
groups_nq135bd = [group['NQ135BD'].dropna().astype(int).values for name, group in pas_data.groupby('C2')]
stat_nq135bd, p_nq135bd_kruskal = kruskal(*groups_nq135bd)
print(f'Kruskal-Wallis H Test for NQ135BD based on C2: stat = {stat_nq135bd}, p = {p_nq135bd_kruskal}')

# Chi-Square Test for Q60 (Confidence in Police in Local Area) across different Boroughs (C2)
contingency_table_q60 = pd.crosstab(pas_data['Q60'].dropna().astype(int), pas_data['C2'].dropna())
chi2_q60, p_q60, dof_q60, expected_q60 = chi2_contingency(contingency_table_q60)
print(f"Chi-Square Test for Q60: chi2 = {chi2_q60}, p = {p_q60}")

# Kruskal-Wallis H Test for Q60 based on C2
groups_q60 = [group['Q60'].dropna().astype(int).values for name, group in pas_data.groupby('C2')]
stat_q60, p_q60_kruskal = kruskal(*groups_q60)
print(f'Kruskal-Wallis H Test for Q60 based on C2: stat = {stat_q60}, p = {p_q60_kruskal}')

# Chi-Square Test for Q61 (Confidence in Police in London) across different Boroughs (C2)
contingency_table_q61 = pd.crosstab(pas_data['Q61'].dropna().astype(int), pas_data['C2'].dropna())
chi2_q61, p_q61, dof_q61, expected_q61 = chi2_contingency(contingency_table_q61)
print(f"Chi-Square Test for Q61: chi2 = {chi2_q61}, p = {p_q61}")

# Kruskal-Wallis H Test for Q61 based on C2
groups_q61 = [group['Q61'].dropna().astype(int).values for name, group in pas_data.groupby('C2')]
stat_q61, p_q61_kruskal = kruskal(*groups_q61)
print(f'Kruskal-Wallis H Test for Q61 based on C2: stat = {stat_q61}, p = {p_q61_kruskal}')

# Detailed Analysis of Top and Bottom Boroughs
# Identify boroughs with highest and lowest trust (NQ135BD)
mean_trust_boroughs = pas_data.groupby('C2')['NQ135BD'].mean()
top_trust_boroughs = mean_trust_boroughs.nlargest(5)
bottom_trust_boroughs = mean_trust_boroughs.nsmallest(5)

print("Top 5 Boroughs with Highest Trust (NQ135BD):")
print(top_trust_boroughs)
print("\nBottom 5 Boroughs with Lowest Trust (NQ135BD):")
print(bottom_trust_boroughs)

# Identify boroughs with highest and lowest confidence in local police (Q60)
mean_confidence_local_boroughs = pas_data.groupby('C2')['Q60'].mean()
top_confidence_local_boroughs = mean_confidence_local_boroughs.nlargest(5)
bottom_confidence_local_boroughs = mean_confidence_local_boroughs.nsmallest(5)

print("Top 5 Boroughs with Highest Confidence in Local Police (Q60):")
print(top_confidence_local_boroughs)
print("\nBottom 5 Boroughs with Lowest Confidence in Local Police (Q60):")
print(bottom_confidence_local_boroughs)

# Identify boroughs with highest and lowest confidence in London police (Q61)
mean_confidence_london_boroughs = pas_data.groupby('C2')['Q61'].mean()
top_confidence_london_boroughs = mean_confidence_london_boroughs.nlargest(5)
bottom_confidence_london_boroughs = mean_confidence_london_boroughs.nsmallest(5)

print("Top 5 Boroughs with Highest Confidence in London Police (Q61):")
print(top_confidence_london_boroughs)
print("\nBottom 5 Boroughs with Lowest Confidence in London Police (Q61):")
print(bottom_confidence_london_boroughs)

# Convert datetime to pandas datetime object
pas_data['datetime'] = pd.to_datetime(pas_data['datetime'])

# Set datetime as index
pas_data.set_index('datetime', inplace=True)

# Resample to monthly frequency and calculate mean trust and confidence
monthly_trust_confidence = pas_data.resample('MS')[['NQ135BD', 'Q60', 'Q61']].mean()

# Plot the trends
plt.figure(figsize=(12, 6))
monthly_trust_confidence.plot(title="Monthly Average Trust and Confidence")
plt.xlabel("Date")
plt.ylabel("Average Score")
plt.show()

# Mapping for education levels
education_mapping = {
    'NQ146_Trade apprenticeship': 1,
    'NQ146_NVQ/GNVQ': 2,
    'NQ146_BTEC Level 1': 3,
    'NQ146_O-levels/CSE/GCSEs': 4,
    'NQ146_BTEC level 2': 5,
    'NQ146_A-levels': 6,
    'NQ146_BTEC level 3': 7,
    'NQ146_ONC, OND or City and Guilds': 8,
    'NQ146_HNC or HND/BTEC level 4': 9,
    'NQ146_University Degree (Bachelor degree)': 10,
    'NQ146_Post-graduate degree or qualification': 11,
    'NQ146_Other': 12,
    'NQ146_No qualifications': 13,
    "NQ146_Don't know": np.nan,
    'NQ146_Refused': np.nan,
    'NQ146_Not asked': np.nan
}

# Ensure columns are present in the DataFrame before applying mapping
existing_columns = [col for col in education_mapping.keys() if col in pas_data.columns]

# Create EducationLevel column based on the highest education level in each row
pas_data['EducationLevel'] = pas_data[existing_columns].apply(lambda row: row.idxmax(), axis=1).map(education_mapping)

# Ensure no duplicate indices in the DataFrame
pas_data = pas_data.reset_index(drop=True)

# Bar plot for Confidence in London Police (Q61) vs Education Level
plt.figure(figsize=(12, 6))
sns.barplot(x=pas_data['EducationLevel'], y=pas_data['Q61'], hue=pas_data['C2'], errorbar=None)
plt.title("Confidence in London Police (Q61) vs. Education Level")
plt.xlabel("Education Level")
plt.ylabel("Confidence in London Police (Q61)")
plt.legend(title="Boroughs", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Additional analyses and tests

# Descriptive statistics for NQ135BD, Q60, and Q61 by borough
desc_stats = pas_data.groupby('C2')[['NQ135BD', 'Q60', 'Q61']].agg(['mean', 'median', 'std'])
print(desc_stats)

# ANOVA for NQ135BD (Trust) across boroughs
f_stat, p_value = f_oneway(*[group['NQ135BD'].dropna().astype(int) for name, group in pas_data.groupby('C2')])
print(f"ANOVA for NQ135BD: F-statistic = {f_stat}, p-value = {p_value}")

# ANOVA for Q60 (Confidence in Local Police) across boroughs
f_stat, p_value = f_oneway(*[group['Q60'].dropna().astype(int) for name, group in pas_data.groupby('C2')])
print(f"ANOVA for Q60: F-statistic = {f_stat}, p-value = {p_value}")

# ANOVA for Q61 (Confidence in London Police) across boroughs
f_stat, p_value = f_oneway(*[group['Q61'].dropna().astype(int) for name, group in pas_data.groupby('C2')])
print(f"ANOVA for Q61: F-statistic = {f_stat}, p-value = {p_value}")

# Align indices for post-hoc tests
aligned_nq135bd = pas_data[['NQ135BD', 'C2']].dropna()
aligned_q60 = pas_data[['Q60', 'C2']].dropna()
aligned_q61 = pas_data[['Q61', 'C2']].dropna()

# Post-hoc test for NQ135BD (Trust)
trust_mc = mc.MultiComparison(aligned_nq135bd['NQ135BD'], aligned_nq135bd['C2'])
trust_result = trust_mc.tukeyhsd()
print(trust_result)

# Post-hoc test for Q60 (Confidence in Local Police)
local_mc = mc.MultiComparison(aligned_q60['Q60'], aligned_q60['C2'])
local_result = local_mc.tukeyhsd()
print(local_result)

# Post-hoc test for Q61 (Confidence in London Police)
london_mc = mc.MultiComparison(aligned_q61['Q61'], aligned_q61['C2'])
london_result = london_mc.tukeyhsd()
print(london_result)

# Correlation matrix
corr_matrix = pas_data[['NQ135BD', 'Q60', 'Q61', 'EducationLevel', 'Q1', 'Q3C', 'Q3H', 'Q3J', 'Q13', 'NQ21']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Key Variables')
plt.show()

