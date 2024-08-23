import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Directory and file name

dir = "20240715033306-6-4-0.88-0.95"
file = "resultbest_20240715033306.csv"

# dir = "20240715021813-4-3-0.8-0.9"
# file = "resultbest_20240715021813.csv"

# Extract n_problems and n_reps from directory name
n_problems, n_reps, myF, myCR = [float(x) if '.' in x else int(x) for x in dir.split('-')[1:5]]

# Output directory
out_dir = f"results/{dir}/best"
parent_out = os.path.join("/Users/mlnick/Documents/University/Git/Tsukuba-Master/ExpDesign/report-2/", out_dir)

# Load the CSV data
df = pd.read_csv(f"{parent_out}/{file}")

# Filter the runtime information - the runtime information is included in the last line
runtime_info = df.iloc[-1]
if "Runtime" in runtime_info.values[0]:
    runtime_line = df.iloc[-1, 0]
    df = df.iloc[:-1]  # Remove the last line

# Convert columns to appropriate data types
df['Selection Policy'] = df['Selection Policy'].astype(int)
df['Problem'] = df['Problem'].astype(int)
df['Repetition'] = df['Repetition'].astype(int)
df['best'] = df['best'].astype(float)

# Compute statistics
statistics = df.groupby(['Selection Policy', 'Problem'])['best'].agg(['mean', 'median', 'std'])
print(statistics)

# Save statistics and runtime to a file
stats_file = os.path.join(parent_out, f"{dir}_statistics.txt")
with open(stats_file, 'w') as f:
    f.write(statistics.to_string())
    f.write("\n\n")
    f.write(runtime_line)
    print(f"Statistics and runtime saved to {stats_file}")

# Visualization
plt.figure(figsize=(14, 7))

# Boxplot for 'best' values
plt.subplot(1, 2, 1)
sns.boxplot(x='Selection Policy', y='best', data=df)
plt.title('Boxplot of Best Values by Selection Policy')
plt.xlabel('Selection Policy')
plt.ylabel('Best Value')

boxplot_file = os.path.join(parent_out, f"{dir}_boxplot.png")
plt.savefig(boxplot_file)
print(f"Boxplot saved to {boxplot_file}")

# Violin plot for 'best' values
plt.subplot(1, 2, 2)
sns.violinplot(x='Selection Policy', y='best', data=df)
plt.title('Violin Plot of Best Values by Selection Policy')
plt.xlabel('Selection Policy')
plt.ylabel('Best Value')

violinplot_file = os.path.join(parent_out, f"{dir}_violinplot.png")
plt.tight_layout()
plt.savefig(violinplot_file)
print(f"Violin plot saved to {violinplot_file}")

# Additional plot: Mean and Median for each Policy and Problem
mean_median_df = statistics.reset_index().melt(id_vars=['Selection Policy', 'Problem'], value_vars=['mean', 'median'], var_name='Statistic', value_name='Value')

plt.figure(figsize=(14, 7))
sns.barplot(x='Problem', y='Value', hue='Statistic', data=mean_median_df, palette='muted', errorbar=None)
plt.title('Mean and Median of Best Values by Problem and Selection Policy')
plt.xlabel('Problem')
plt.ylabel('Value')
plt.legend(title='Statistic')

barplot_file = os.path.join(parent_out, f"{dir}_mean_median.png")
plt.savefig(barplot_file)
print(f"Mean and Median plot saved to {barplot_file}")
plt.show()
