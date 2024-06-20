import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the PAS dataset
data = pd.read_csv('../data/PAS_data.csv', low_memory=False)

# Dropping missing values and converting to integer
data = data[['Q136r', 'NQ135BD']].dropna()
data['Q136r'] = data['Q136r'].astype(int)
data['NQ135BD'] = data['NQ135BD'].astype(int)

# Define labels for trust levels including -1
trust_labels = {
    1: 'Strongly Agree',
    2: 'Tend to Agree',
    3: 'Neither Agree Nor Disagree',
    4: 'Tend to Disagree',
    5: 'Strongly Disagree'
}


# Compute percentages for each trust level
def compute_trust_response_percentages(data):
    results = []
    trust_levels = sorted(data['NQ135BD'].unique())

    for trust_level in trust_levels:
        trust_data = data[data['NQ135BD'] == trust_level]
        total_count = len(trust_data)

        age_groups = sorted(trust_data['Q136r'].unique())

        for age_group in age_groups:
            group_data = trust_data[trust_data['Q136r'] == age_group]
            age_count = len(group_data)

            age_percentage = (age_count / total_count) * 100
            results.append((trust_level, age_group, age_percentage))

    return results


# Get the results
trust_response_percentages = compute_trust_response_percentages(data)

# Print the results
print(f"{'Trust Level':<20} {'Age Group':<10} {'Percentage (%)':<15}")
for trust_level, age_group, age_percentage in trust_response_percentages:
    trust_label = trust_labels.get(trust_level, f'Unknown ({trust_level})')
    print(f"{trust_label:<20} {age_group:<10} {age_percentage:<15.2f}")

# Map age group numbers to labels
age_labels_dict = {
    -1: 'Unknown',
    1: '16-17',
    2: '18-21',
    3: '22-24',
    4: '25-34',
    5: '35-44',
    6: '45-54',
    7: '55-64',
    8: '65-74',
    9: '75-84',
    10: '85+'
}

# Optional: Plotting the results
trust_levels, age_groups, percentages = zip(*trust_response_percentages)
unique_trust_levels = sorted(set(trust_levels))
unique_age_groups = sorted(set(age_groups))

# Create a DataFrame for easier plotting
plot_data = pd.DataFrame({
    'Trust Level': trust_levels,
    'Age Group': age_groups,
    'Percentage': percentages
})

fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.1
for i, trust_level in enumerate(unique_trust_levels):
    level_data = plot_data[plot_data['Trust Level'] == trust_level]
    ax.bar(level_data['Age Group'] + i * bar_width, level_data['Percentage'], width=bar_width,
           label=trust_labels.get(trust_level, f'Unknown ({trust_level})'))

age_labels = [age_labels_dict[age_group] for age_group in unique_age_groups]
ax.set_xticks(unique_age_groups)
ax.set_xticklabels(age_labels, rotation=45)
ax.set_xlabel('Age Groups')
ax.set_ylabel('Percentage')
ax.set_title('Percentage Distribution of Trust Levels by Age Group')
ax.legend(title='Trust Level')

plt.tight_layout()
plt.show()
