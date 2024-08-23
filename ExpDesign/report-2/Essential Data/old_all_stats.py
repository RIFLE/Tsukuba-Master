import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Directory and file name

dir = "20240715033306-6-4-0.88-0.95"
file = "resultall_20240715033306.csv"

# dir = "20240715021813-4-3-0.8-0.9"
# file = "resultall_20240715021813.csv"

# Extract n_problems and n_reps from directory name
n_problems, n_reps, myF, myCR = [float(x) if '.' in x else int(x) for x in dir.split('-')[1:5]]

# Output directory
out_dir = f"results/{dir}/all"
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
df['neval'] = df['neval'].astype(int)
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

# Histogram of 'best' values
plt.subplot(2, 2, 1)
sns.histplot(df['best'], kde=True, bins=30)
plt.title('Histogram of Best Values')
plt.xlabel('Best Value')
plt.ylabel('Frequency')

histogram_file = os.path.join(parent_out, f"{dir}_histogram.png")
plt.savefig(histogram_file)
print(f"Histogram saved to {histogram_file}")

# Boxplot for 'best' values
plt.subplot(2, 2, 2)
sns.boxplot(x='Selection Policy', y='best', data=df)
plt.title('Boxplot of Best Values by Selection Policy')
plt.xlabel('Selection Policy')
plt.ylabel('Best Value')

boxplot_file = os.path.join(parent_out, f"{dir}_boxplot.png")
plt.savefig(boxplot_file)
print(f"Boxplot saved to {boxplot_file}")

# Violin plot for 'best' values
plt.subplot(2, 2, 3)
sns.violinplot(x='Selection Policy', y='best', data=df)
plt.title('Violin Plot of Best Values by Selection Policy')
plt.xlabel('Selection Policy')
plt.ylabel('Best Value')

violinplot_file = os.path.join(parent_out, f"{dir}_violinplot.png")
plt.savefig(violinplot_file)
print(f"Violin plot saved to {violinplot_file}")

# Q-Q Plot for 'best' values to check normality
plt.subplot(2, 2, 4)
stats.probplot(df['best'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Best Values')

qqplot_file = os.path.join(parent_out, f"{dir}_qqplot.png")
plt.savefig(qqplot_file)
print(f"Q-Q plot saved to {qqplot_file}")

plt.tight_layout()
plt.show()

# Additional plots: Raw data for each Selection Policy
selection_policies = df['Selection Policy'].unique()

for sp in selection_policies:
    plt.figure(figsize=(14, 7))
    subset_df = df[df['Selection Policy'] == sp]

    plt.subplot(1, 1, 1)
    sns.histplot(subset_df['best'], kde=True, bins=30)
    plt.title(f'Histogram of Best Values for Selection Policy {sp}')
    plt.xlabel('Best Value')
    plt.ylabel('Frequency')

    hist_subset_file = os.path.join(parent_out, f"{dir}_histogram_policy_{sp}.png")
    plt.savefig(hist_subset_file)
    print(f"Histogram for Selection Policy {sp} saved to {hist_subset_file}")

    plt.show()
