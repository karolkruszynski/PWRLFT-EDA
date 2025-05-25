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
bins = [0, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 200]
labels = ['<45', '45-50' , '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '86-90', '91-95', '96-100', '101-105', '106-110', '111-115', '116-120', '120+']

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


# Diff between female and male lifters in every exercise attempts in %
exercises = [['Squat1Kg', 'Squat2Kg', 'Squat3Kg'],['Bench1Kg', 'Bench2Kg', 'Bench3Kg'],['Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg']]
for exercise in exercises:
    mean_exercise = df_raw.groupby('Sex')[exercise].mean()
    pct_diff = (mean_exercise.loc['M'] - mean_exercise.loc['F']) / mean_exercise.loc['F'] * 100
    plt.figure(figsize=(10,5))
    sns.heatmap(pct_diff.to_frame().T, annot=True, cmap='coolwarm', fmt='.1f', cbar_kws={'label': '% difference'})
    plt.title(f'Percentage difference in mean {exercise} scores: Men vs Women')
    plt.yticks([])
    plt.tight_layout()
    plt.show()

# Std for Gender by attempt
for exercise in exercises:
    std_df = df.groupby('Sex')[exercise].std()
    std_melted = std_df.reset_index().melt(id_vars='Sex', var_name='Attempt', value_name='StdDev')
    plt.figure(figsize=(8,5))
    sns.barplot(data=std_melted, x='Attempt', y='StdDev', hue='Sex')
    plt.title(f'Standard deviation of {exercise} scores by gender and attempt')
    plt.ylabel('Std [kg]')
    plt.show()

# Prcnt distribution good attempts
for exercise in exercises:
    success_rate = df_raw.groupby('Sex')[exercise].apply(lambda x: x.notna().mean() * 100)
    sns.heatmap(success_rate, annot=True, fmt='.1f', cmap='Greens', cbar_kws={'label': '% good attempts'})
    plt.tight_layout()
    plt.show()

# Performance indicator TotalKg / Bodyweight
df_raw['TotalKg_per_BodyweightKg'] = df_raw['TotalKg'] / df_raw['BodyweightKg']
mean_performance_by_bodyweight = df_raw.groupby(['Sex', 'WeightCategory'])['TotalKg_per_BodyweightKg'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(data=mean_performance_by_bodyweight.reset_index(),
             x='WeightCategory',
             y='TotalKg_per_BodyweightKg',
             hue='Sex',
             marker='o')
plt.title('Average Total / Body weight by gender')
plt.xlabel('WeightCategory')
plt.ylabel('Total / Bodyweight')
plt.grid(True)
plt.tight_layout()
plt.show()