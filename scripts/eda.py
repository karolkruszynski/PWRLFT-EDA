import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/sample_data.csv", low_memory=False)
sns.histplot(df['Age'].dropna(), bins=50, kde=True)
plt.title("Age distribution of lifters")
plt.xlabel("Age")
plt.ylabel("Number of lifters")
plt.show()

## Analysis of Gender Differences

# Points Score Female vs Male

df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
scores = ['Dots', 'Wilks', 'Glossbrenner']
for score in scores:
    grouped = df.groupby(['Year', 'Sex'])[score].mean().reset_index()
    sns.lineplot(data=grouped, x='Year', y=score, hue='Sex')
    plt.show()

grouped = df.groupby(['Year', 'Sex'])[['Dots', 'Wilks', 'Glossbrenner']].mean().reset_index()
melted = grouped.melt(id_vars=['Year', 'Sex'],
                      value_vars=['Wilks', 'Dots', 'Glossbrenner'],
                      var_name='ScoreType',
                      value_name='AvgScore')
g = sns.FacetGrid(data=melted, col='Sex', height=5, aspect=1.2)
g.map_dataframe(sns.lineplot, x="Year", y="AvgScore", hue="ScoreType", marker="o")
g.add_legend()
g.set_axis_labels("Year", "Average adjustment value")
g.set_titles("{col_name}")
plt.suptitle("Comparison of Wilks / Dots / Glossbrenner indices - Separate for gender.", y=1.05)
plt.tight_layout()
plt.show()

# Visualization of strength distributions for men and women by weight category
#print(df['WeightClassKg'].unique())
df_raw = df[df['Equipment'] == 'Raw']

def parse_weight_class(val):
    try:
        return float(val.replace('+', '').replace('-','-'))
    except:
        return np.nan

df_raw['WeightClassKg_clean'] = df_raw['WeightClassKg'].astype(str).apply(parse_weight_class)
print(df_raw['WeightClassKg_clean'].unique())
bins = [0, 59, 69, 79, 89, 99, 109, 120, 200]
labels = ['≤59', '60–69', '70–79', '80–89', '90–99', '100–109', '110–120', '120+']

df_raw['WeightCategory'] = pd.cut(df_raw['WeightClassKg_clean'], bins=bins, labels=labels, right=False)
plt.figure(figsize=(14, 6))
sns.barplot(data=df_raw,
            x='WeightCategory',
            y='TotalKg',
            hue='Sex',
            palette='pastel')
plt.title("Distribution of TotalKg results by weight category and gender")
plt.xlabel("Weight category (kg)")
plt.ylabel("TotalKg score")
plt.legend(title="Sex")
plt.tight_layout()
plt.show()

# Diff between female and male lifters in Squat attempts
mean_squat = df_raw[['Sex', 'Squat1Kg', 'Squat2Kg', 'Squat3Kg']].groupby('Sex').mean().reset_index()
melted = mean_squat.melt(id_vars='Sex',
                         value_vars=['Squat1Kg', 'Squat2Kg', 'Squat3Kg'],
                         var_name='Squat Attempts',
                         value_name='Average Weight')
plt.figure(figsize=(8,5))
sns.lineplot(data=melted, x='Squat Attempts', y='Average Weight', hue='Sex', marker='o')
plt.title('Distribution of squat1-3 results by gender')
plt.xlabel('Attempts')
plt.ylabel('Result [kg]')
plt.show()

# Diff between female and male lifters in Squat attempts in %
mean_squat = df_raw.groupby('Sex')[['Squat1Kg', 'Squat2Kg', 'Squat3Kg']].mean()
pct_diff = (mean_squat.loc['M'] - mean_squat.loc['F'])  / mean_squat.loc['F'] * 100
plt.figure(figsize=(10,5))
sns.heatmap(pct_diff.to_frame().T, annot=True, cmap='coolwarm', fmt='.1f', cbar_kws={'label': '% difference'})
plt.title('Percentage difference in mean Squat1-3 scores: Men vs Women')
plt.yticks([])
plt.tight_layout()
plt.show()

# Std for Gender by attempt
std_df = df.groupby('Sex')[['Squat1Kg', 'Squat2Kg', 'Squat3Kg']].std()
std_melted = std_df.reset_index().melt(id_vars='Sex', var_name='Attempt', value_name='StdDev')

plt.figure(figsize=(8,5))
sns.barplot(data=std_melted, x='Attempt', y='StdDev', hue='Sex')
plt.title('Standard deviation of Squat scores by gender and attempt')
plt.ylabel('Std [kg]')
plt.show()
