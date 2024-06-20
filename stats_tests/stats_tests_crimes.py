import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import levene

# Load the datasets
pas_data = pd.read_csv('../data/PAS_data.csv', low_memory=False)
crimes_data = pd.read_csv('../data/crimes.csv', low_memory=False)

# Display the first few rows of each DataFrame
print("PAS Data Columns:\n", pas_data.head())
print("\nCrimes Data Columns:\n", crimes_data.head())

# Convert datetime in pas_data to pandas datetime type
pas_data['datetime'] = pd.to_datetime(pas_data['datetime'])
pas_data['Year'] = pas_data['datetime'].dt.year

# Aggregate crime counts per borough and year
crimes_data['Crime ID'] = crimes_data['Crime ID'].fillna(method='ffill')  # Ensure Crime ID is filled
crimes_agg = crimes_data.groupby(['boroughs', 'Year']).size().reset_index(name='Crime_counts')

# Merge the datasets on boroughs and year
merged_data = pd.merge(pas_data, crimes_agg, left_on=['C2', 'Year'], right_on=['boroughs', 'Year'])

# Summarize the merged data
print("Merged Data Summary:\n", merged_data.describe())

# Check for missing values
print("\nMissing values in merged data:\n", merged_data.isnull().sum())

# Drop non-numeric columns before correlation analysis
numeric_merged_data = merged_data.select_dtypes(include=[np.number])

# Correlation Matrix
corr_matrix = numeric_merged_data.corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Regression Analysis
regression_model = ols('Q3C ~ Crime_counts', data=merged_data).fit()
print("\nRegression Analysis Summary:")
print(regression_model.summary())

# T-Test: Compare Trust (Q3C) between two groups (e.g., different lengths of residence)
group1 = merged_data[merged_data['Q1'] <= 4]['Q3C']
group2 = merged_data[merged_data['Q1'] > 4]['Q3C']
t_stat, p_val = stats.ttest_ind(group1.dropna(), group2.dropna())
print(f"\nT-test results for comparing Trust (Q3C) between two groups: t_stat = {t_stat}, p_val = {p_val}")

# T-Test for Q3C (Trust in Police) and Q13 (Worry about Crime)
t_stat, p_val = stats.ttest_ind(merged_data[merged_data['Q3C'] == 1]['Q13'],
                                merged_data[merged_data['Q3C'] == 0]['Q13'])
print(f"\nT-test results for Q3C (Trust in Police) and Q13 (Worry about Crime): t_stat = {t_stat}, p_val = {p_val}")

# Check for missing values
print(merged_data[['Q3C', 'C2']].isnull().sum())

# Drop rows with missing values
merged_data_clean = merged_data.dropna(subset=['Q3C', 'C2'])

# Levene's test for homogeneity of variances (across boroughs)
groups = [merged_data_clean[merged_data_clean['C2'] == borough]['Q3C'].dropna() for borough in merged_data_clean['C2'].unique()]
levene_test = levene(*groups)
print(f"\nLevene's test for homogeneity of variances across boroughs: {levene_test}")

# Define the model formula for ANOVA
formula = 'Q3C ~ C2'

# Fit the model
model = ols(formula, data=merged_data_clean).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()  # Calculate eta-squared for effect size
print("ANOVA results for Q3C across different boroughs (C2) with effect size:\n", anova_table)

# Visualization: Boxplot for Q3C across different boroughs
plt.figure(figsize=(12, 8))
sns.boxplot(x='C2', y='Q3C', data=merged_data_clean)
plt.title('Trust in the Police (Q3C) Across Different Boroughs (C2)')
plt.xticks(rotation=90)
plt.xlabel('Boroughs')
plt.ylabel('Trust in Police (Q3C)')
plt.show()

# Perform Tukey's HSD post-hoc test
tukey = pairwise_tukeyhsd(endog=merged_data_clean['Q3C'],
                          groups=merged_data_clean['C2'],
                          alpha=0.05)
print(tukey)

# Residual plot
residuals = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(10, 6))
sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# QQ plot
sm.qqplot(residuals, line='45')
plt.title('QQ Plot')
plt.show()

# Trust Change over time
# Resample data by month
monthly_trust = merged_data.set_index('datetime')['Q3C'].resample('M').mean()

# Plot the time series
monthly_trust.plot(title='Monthly Average Trust in Police (Q3C)')
plt.xlabel('Date')
plt.ylabel('Average Trust')
plt.show()

print("Process finished with exit code 0")
