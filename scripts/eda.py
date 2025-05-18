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