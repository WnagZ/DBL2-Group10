import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the PAS dataset
pas_data = pd.read_csv('../data/PAS_data.csv', low_memory=False)

# Verify column names
print("PAS Data Columns:", pas_data.columns)

# Replace -1 with NaN to handle missing values appropriately
pas_data.replace(-1, np.nan, inplace=True)

# Ensure the index is unique
pas_data = pas_data.reset_index(drop=True)

# Define the correct ethnic group columns based on the dataset
ethnic_columns = {
    'NQ147r_Black': 'Black',
    'NQ147r_White British': 'White',
    'NQ147r_White Other': 'White Other',
    'NQ147r_Mixed': 'Mixed',
    'NQ147r_Other': 'Other'
}

# Ensure the columns exist
available_ethnic_columns = [col for col in ethnic_columns.keys() if col in pas_data.columns]
if available_ethnic_columns:
    pas_data['Ethnicity'] = pas_data[available_ethnic_columns].idxmax(axis=1).map(ethnic_columns)
else:
    print("Required ethnic columns are not available in the dataset.")
    available_ethnic_columns = []

# Mediation Analysis
def mediation_analysis(data):
    # Encode categorical variables
    data['Ethnicity'] = data['Ethnicity'].astype('category').cat.codes

    # Mediator model
    mediator_model = smf.ols('A121 ~ Ethnicity', data=data).fit()

    # Outcome model
    outcome_model = smf.ols('NQ135BD ~ A121 + Ethnicity', data=data).fit()

    print(mediator_model.summary())
    print(outcome_model.summary())

def trust_vs_racial_profiling(data):
    # Dropping missing values and converting to integer
    trust_levels = data['NQ135BD'].dropna().astype(int)
    profiling_confidence = data['A121'].dropna().astype(int)

    # Cross-tabulate trust levels with profiling confidence and normalize by columns
    cross_tab = pd.crosstab(trust_levels, profiling_confidence, normalize='columns') * 100

    # Perform Chi-Square test
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(trust_levels, profiling_confidence))
    print(f"Chi-Square Test for Trust vs Racial Profiling: chi2 = {chi2}, p = {p}")

    # Plotting as a grouped bar chart
    cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Trust in Police vs Confidence in Fairness of Stop and Search')
    plt.xlabel('Confidence in Stop and Search Fairness')
    plt.ylabel('Percentage of Respondents')
    plt.legend(title='Trust in Police', labels=[
        'Strongly disagree',
        'Tend to disagree',
        'Neither agree nor disagree',
        'Tend to agree',
        'Strongly agree'
    ])
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[
        'Strongly disagree',
        'Disagree',
        'Neutral',
        'Agree',
        'Strongly agree'
    ], rotation=45)
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def opinion_on_stop_search(data):
    opinion_levels = data['A120'].dropna().astype(int)
    if 'Ethnicity' in data.columns:
        cross_tab = pd.crosstab(opinion_levels, data['Ethnicity'].dropna(), normalize='columns') * 100

        chi2, p, dof, ex = chi2_contingency(pd.crosstab(opinion_levels, data['Ethnicity'].dropna()))

        print(f"Chi-Square Test for Opinion on Stop and Search: chi2 = {chi2}, p = {p}")

        # Plotting as a grouped bar chart
        cross_tab.plot(kind='bar', colormap='viridis')
        plt.title('Opinion on Stop and Search vs Ethnicity')
        plt.xlabel('Opinion on Stop and Search')
        plt.ylabel('Percentage of Respondents')
        plt.legend(title='Ethnicity')
        plt.xticks(ticks=[0, 1, 2, 3], labels=[
            'Very confident',
            'Fairly confident',
            'Not very confident',
            'Not at all confident'
        ], rotation=45)
        plt.show()
    else:
        print("Ethnicity data is not available for analysis.")

def perception_of_fairness(data):
    # Dropping missing values and converting to integer
    fairness_levels = data['Q62C'].dropna().astype(int)
    ethnicity = data['Ethnicity'].dropna()

    if not ethnicity.empty:
        # Cross-tabulate fairness levels with ethnicity and normalize by columns
        cross_tab = pd.crosstab(fairness_levels, ethnicity, normalize='columns') * 100

        # Perform Chi-Square test
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(fairness_levels, ethnicity))
        print(f"Chi-Square Test for Perception of Police Fairness: chi2 = {chi2}, p = {p}")

        # Plotting as a grouped bar chart
        plt.figure(figsize=(20, 12))
        cross_tab.plot(kind='bar', colormap='viridis')
        plt.title('Perception of Police Fairness vs Ethnicity')
        plt.xlabel('Perception of Police Fairness')
        plt.ylabel('Percentage of Respondents')
        plt.legend(title='Ethnicity')
        plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[
            'Strongly agree',
            'Tend to agree',
            'Neutral',
            'Tend to disagree',
            'Strongly disagree'
        ], rotation=15)
        plt.show()
    else:
        print("Ethnicity data is not available for analysis.")



def calculate_trust_index(row):
    score_mapping = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}
    return score_mapping.get(row, 0)  # Return 0 if the value is not in score_mapping

def longitudinal_analysis(data):
    # Convert datetime to Year
    data['Year'] = pd.to_datetime(data['datetime']).dt.year

    # Fill NaN values in NQ135BD with a neutral score (e.g., 3 for "Neither agree nor disagree")
    #data['NQ135BD'] = data['NQ135BD'].fillna(3)

    # Map scores to NQ135BD responses
    data['Trust_Score'] = data['NQ135BD'].map(calculate_trust_index)

    # Calculate yearly trust index for each ethnicity
    yearly_trust = data.groupby(['Year', 'Ethnicity'])['Trust_Score'].sum().unstack()
    response_counts = data.groupby(['Year', 'Ethnicity'])['Trust_Score'].count().unstack()

    trust_index = yearly_trust / response_counts

    # Plotting
    trust_index.plot(kind='line')
    plt.title('Yearly Trust in Police by Ethnicity (Trust Index)')
    plt.xlabel('Year')
    plt.ylabel('Trust Index')
    plt.legend(title='Ethnicity')
    plt.ylim(-2, 2)  # Set limits based on score range
    plt.show()

# Sentiment Towards Police Presence by Ethnicity
def sentiment_towards_police_presence(data):
    police_presence = data['NQ21'].dropna().astype(int)
    if available_ethnic_columns:
        cross_tab = pd.crosstab(police_presence, data['Ethnicity'].dropna())

        chi2, p, dof, ex = chi2_contingency(cross_tab)

        print(f"Chi-Square Test for Sentiment Towards Police Presence: chi2 = {chi2}, p = {p}")

        cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Sentiment Towards Police Presence vs Ethnicity')
        plt.xlabel('Sentiment Towards Police Presence')
        plt.ylabel('Number of Respondents')
        plt.legend(title='Ethnicity')
        plt.show()
    else:
        print("Ethnicity data is not available for analysis.")


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the data
# Assuming data is loaded into a pandas DataFrame named `data`
# data = pd.read_csv('path_to_your_data.csv')

# Trust in Police by Frequency of Patrols (White respondents)
def trust_by_patrol_frequency_white(data):
    if 'White' in data['Ethnicity'].unique():
        white_data = data[data['Ethnicity'] == 'White']
        white_patrol_frequency = white_data['Q65'].dropna().astype(int)
        white_trust_levels = white_data['NQ135BD'].dropna().astype(int)

        cross_tab_white = pd.crosstab(white_patrol_frequency, white_trust_levels, normalize='columns') * 100

        chi2_white, p_white, dof_white, ex_white = chi2_contingency(pd.crosstab(white_patrol_frequency, white_trust_levels))

        print(f"Chi-Square Test for Trust by Patrol Frequency (White respondents): chi2 = {chi2_white}, p = {p_white}")

        cross_tab_white.plot(kind='bar', colormap='viridis')
        plt.title('Trust in Police vs Frequency of Patrols (White respondents)')
        plt.xlabel('Frequency of Patrols')
        plt.ylabel('Percentage of Respondents')
        plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=[
            'At least daily',
            'At least weekly',
            'At least fortnightly',
            'At least monthly',
            'Less often',
            'Never'
        ], rotation=45)
        plt.legend(title='Trust in Police', labels=[
            'Strongly disagree',
            'Tend to disagree',
            'Neither agree nor disagree',
            'Tend to agree',
            'Strongly agree'
        ])
        plt.show()
    else:
        print("White respondents data is not available for analysis.")

# Trust in Police by Frequency of Patrols (Black respondents)
def trust_by_patrol_frequency_black(data):
    if 'Black' in data['Ethnicity'].unique():
        black_data = data[data['Ethnicity'] == 'Black']
        black_patrol_frequency = black_data['Q65'].dropna().astype(int)
        black_trust_levels = black_data['NQ135BD'].dropna().astype(int)

        cross_tab_black = pd.crosstab(black_patrol_frequency, black_trust_levels, normalize='columns') * 100

        chi2_black, p_black, dof_black, ex_black = chi2_contingency(pd.crosstab(black_patrol_frequency, black_trust_levels))

        print(f"Chi-Square Test for Trust by Patrol Frequency (Black respondents): chi2 = {chi2_black}, p = {p_black}")

        cross_tab_black.plot(kind='bar', colormap='viridis')
        plt.title('Trust in Police vs Frequency of Patrols (Black respondents)')
        plt.xlabel('Frequency of Patrols')
        plt.ylabel('Percentage of Respondents')
        plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=[
            'At least daily',
            'At least weekly',
            'At least fortnightly',
            'At least monthly',
            'Less often',
            'Never'
        ], rotation=45)
        plt.legend(title='Trust in Police', labels=[
            'Strongly disagree',
            'Tend to disagree',
            'Neither agree nor disagree',
            'Tend to agree',
            'Strongly agree'
        ])
        plt.show()
    else:
        print("Black respondents data is not available for analysis.")


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def analyze_trust_neighbors_vs_patrols(data):
    # Define high trust levels for Q3C
    high_trust_values = [1, 2]  # Assuming 1 is 'Strongly agree' and 2 is 'Tend to agree'
    data['High_Trust_Neighbors'] = data['Q3C'].apply(lambda x: 'High Trust' if x in high_trust_values else 'Low Trust')

    # Dropna for relevant columns
    clean_data = data.dropna(subset=['High_Trust_Neighbors', 'Q65'])

    # Cross-tabulation
    cross_tab = pd.crosstab(clean_data['High_Trust_Neighbors'], clean_data['Q65'])

    # Chi-Square test
    chi2, p, dof, ex = chi2_contingency(cross_tab)

    print(f"Chi-Square Test for High Trust in Neighbors vs Perception of Patrols: chi2 = {chi2}, p = {p}")

    # Plotting the results as a grouped bar chart
    cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
    cross_tab_normalized.plot(kind='bar', colormap='viridis')
    plt.title('Perception of Patrols vs Trust in Neighbors')
    plt.xlabel('Trust in Neighbors')
    plt.ylabel('Percentage of Respondents')
    plt.xticks(rotation=0)
    plt.legend(title='Frequency of Patrols', labels=[
        'At least daily',
        'At least weekly',
        'At least fortnightly',
        'At least monthly',
        'Less often',
        'Never'
    ])
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# Example function to perform the analysis
def analyze_ethnicity_vs_patrols(data):
    # Assuming the column 'Ethnicity' contains the ethnicity data and 'Q65' contains the patrol perception data
    # Dropping missing values for relevant columns
    clean_data = data.dropna(subset=['Ethnicity', 'Q65'])

    # Define the mapping for negative perception (for simplicity, let's consider "Less often" and "Never" as negative)
    negative_perception_values = [5, 6]  # Assuming 5 is 'Less often' and 6 is 'Never'
    clean_data['Negative_Patrol_Perception'] = clean_data['Q65'].apply(
        lambda x: 'Negative' if x in negative_perception_values else 'Positive')

    # Cross-tabulation
    cross_tab = pd.crosstab(clean_data['Ethnicity'], clean_data['Negative_Patrol_Perception'])

    # Chi-Square test
    chi2, p, dof, ex = chi2_contingency(cross_tab)

    print(f"Chi-Square Test for Ethnicity vs Perception of Patrols: chi2 = {chi2}, p = {p}")

    # Plotting the results
    cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
    cross_tab_normalized.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Perception of Patrols vs Ethnicity')
    plt.xlabel('Ethnicity')
    plt.ylabel('Percentage of Respondents')
    plt.xticks(rotation=45)
    plt.legend(title='Perception of Patrols')
    plt.show()




# Execute the analysis functions
analyze_ethnicity_vs_patrols(pas_data)
trust_vs_racial_profiling(pas_data)
opinion_on_stop_search(pas_data)
perception_of_fairness(pas_data)
longitudinal_analysis(pas_data)
sentiment_towards_police_presence(pas_data)
trust_by_patrol_frequency_white(pas_data)
trust_by_patrol_frequency_black(pas_data)
mediation_analysis(pas_data)
analyze_trust_neighbors_vs_patrols(pas_data)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


def opinion_on_stop_search(data):
    # Dropping missing values and converting to integer
    opinion_levels = data['A120'].dropna().astype(int)
    ethnicity = data['Ethnicity'].dropna()

    if not ethnicity.empty:
        # Cross-tabulate opinion levels with ethnicity
        cross_tab = pd.crosstab(opinion_levels, ethnicity)

        # Perform Chi-Square test
        chi2, p, dof, ex = chi2_contingency(cross_tab)
        print(f"Chi-Square Test for Opinion on Stop and Search: chi2 = {chi2}, p = {p}")

        # Calculate standardized residuals
        residuals = (cross_tab - ex) / np.sqrt(ex)

        # Print contingency table and standardized residuals
        print("\nContingency Table (Observed Frequencies):")
        print(cross_tab)

        print("\nExpected Frequencies:")
        print(pd.DataFrame(ex, index=cross_tab.index, columns=cross_tab.columns))

        print("\nStandardized Residuals:")
        print(residuals)

        # Normalize the cross-tabulation by columns for plotting
        cross_tab_norm = cross_tab.div(cross_tab.sum(axis=0), axis=1) * 100

        # Plotting as a grouped bar chart
        cross_tab_norm.plot(kind='bar', colormap='viridis')
        plt.title('Opinion on Stop and Search vs Ethnicity')
        plt.xlabel('Opinion on Stop and Search')
        plt.ylabel('Percentage of Respondents')
        plt.legend(title='Ethnicity')
        plt.xticks(ticks=[1, 2, 3, 4], labels=[
            'Very Confident',
            'Fairly Confident',
            'Not Very Confident',
            'Not At All Confident'
        ], rotation=45)
        plt.show()
    else:
        print("Ethnicity data is not available for analysis.")
"""
# Example usage with your data
opinion_on_stop_search(pas_data)

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame for observed frequencies
observed_data = {
    'Ethnicity': ['White', 'Asian', 'Black', 'Other', 'Mixed'],
    'Very Confident': [904, 114, 85, 645, 491],
    'Fairly Confident': [1479, 171, 152, 1461, 754],
    'Not Very Confident': [2569, 309, 260, 2824, 1267],
    'Not At All Confident': [7821, 646, 968, 12637, 4998],
    'No Response': [5963, 389, 911, 11130, 4290]
}

df_observed = pd.DataFrame(observed_data)

# Calculate total number of respondents for each ethnic group
df_observed['Total'] = df_observed.sum(axis=1)

# Calculate the percentages
df_percentages = df_observed.copy()
for column in df_percentages.columns[1:-1]:
    df_percentages[column] = (df_percentages[column] / df_percentages['Total']) * 100

# Drop the 'Total' column as it's no longer needed
df_percentages.drop(columns=['Total'], inplace=True)

# Transpose the DataFrame to match the required format for plotting
df_percentages_transposed = df_percentages.set_index('Ethnicity').transpose()

# Plotting grouped bar chart
ax = df_percentages_transposed.plot(kind='bar', colormap='viridis', width=0.8)
plt.title('Opinions on Stop and Search by Ethnicity (Percentages)')
plt.xlabel('Opinion on Stop and Search')
plt.ylabel('Percentage of Respondents')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Observed frequencies
observed_data = {
    'Ethnicity': ['White', 'Asian', 'Black', 'Other', 'Mixed'],
    'Very Confident': [904, 114, 85, 645, 491],
    'Fairly Confident': [1479, 171, 152, 1461, 754],
    'Not Very Confident': [2569, 309, 260, 2824, 1267],
    'Not At All Confident': [7821, 646, 968, 12637, 4998],
    'No Response': [5963, 389, 911, 11130, 4290]
}"""
""""
df_observed = pd.DataFrame(observed_data)

# Convert all columns except 'Ethnicity' to numeric types
df_observed.iloc[:, 1:] = df_observed.iloc[:, 1:].apply(pd.to_numeric)

# Calculate total number of respondents for each ethnic group
df_observed['Total'] = df_observed.sum(axis=1)

# Calculate the percentages
df_percentages = df_observed.copy()
for column in df_percentages.columns[1:-1]:
    df_percentages[column] = (df_percentages[column] / df_percentages['Total']) * 100

# Drop the 'Total' column as it's no longer needed
df_percentages.drop(columns=['Total'], inplace=True)

# Transpose the DataFrame to match the required format for plotting
df_percentages_transposed = df_percentages.set_index('Ethnicity').transpose()

# Plotting grouped bar chart
ax = df_percentages_transposed.plot(kind='bar', colormap='viridis', width=0.8)
plt.title('Opinions on Stop and Search by Ethnicity (Percentages)')
plt.xlabel('Opinion on Stop and Search')
plt.ylabel('Percentage of Respondents')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()"""
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def perception_of_fairness(data):
    # Dropping missing values and converting to integer
    fairness_levels = data['Q62C'].dropna().astype(int)
    ethnicity = data['Ethnicity'].dropna()

    if not ethnicity.empty:
        # Cross-tabulate fairness levels with ethnicity and normalize by columns
        cross_tab = pd.crosstab(fairness_levels, ethnicity, normalize='columns') * 100

        # Perform Chi-Square test
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(fairness_levels, ethnicity))
        print(f"Chi-Square Test for Perception of Police Fairness: chi2 = {chi2}, p = {p}")

        # Plotting as a grouped bar chart
        plt.figure(figsize=(20, 12))
        cross_tab.plot(kind='bar', colormap='viridis')
        plt.title('Perception of Police Fairness vs Ethnicity')
        plt.xlabel('Perception of Police Fairness')
        plt.ylabel('Percentage of Respondents')
        plt.legend(title='Ethnicity')
        plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[
            'Strongly agree',
            'Tend to agree',
            'Neutral',
            'Tend to disagree',
            'Strongly disagree'
        ], rotation=15)
        plt.show()

        # Additional Analysis: Post-Hoc Pairwise Comparisons
        # Create a dataframe for pairwise comparisons
        comparison_data = data[['Q62C', 'Ethnicity']].dropna()
        comparison_data['Q62C'] = comparison_data['Q62C'].astype(int)

        # Perform pairwise comparisons using Tukey's HSD test
        tukey_result = pairwise_tukeyhsd(comparison_data['Q62C'], comparison_data['Ethnicity'])
        print(tukey_result)

        # Interpret the Tukey's HSD results
        print("Pairwise Comparisons for Perception of Police Fairness by Ethnicity:")
        print("Groups with significant differences (p < 0.05) are indicated by *")
        print(tukey_result.summary())

    else:
        print("Ethnicity data is not available for analysis.")


perception_of_fairness(pas_data)


