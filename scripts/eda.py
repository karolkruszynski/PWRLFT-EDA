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