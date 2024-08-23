import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower

from itertools import combinations

import numpy as np

import os

from scipy.stats import kruskal
import scikit_posthocs as sp

import warnings
warnings.filterwarnings("ignore") # Let your pc rip

########################### SAMPLE SETS - DATA   ######################################
# Construct output directory

# dir = "20240715033306-6-4-0.88-0.95"
# file = "resultall_20240715033306.csv"

# dir = "20240715134924-4-3-0.8-0.9"
# file = "resultall_20240715134924.csv"

# dir = "20240715144745-8-24-0.8-0.9"
# file = "resultall_20240715144745.csv"

######################################

# dir = "20240715180434-8-24-0.7-0.75"
# file = "resultall_20240715180434.csv"

# dir = "20240715181704-8-24-0.8-0.95"
# file = "resultall_20240715181704.csv"

# dir = "20240715222446-8-24-0.8-0.98"
# file = "resultall_20240715222446.csv"

# dir = "20240715223103-8-24-0.8-0.99"
# file = "resultall_20240715223103.csv"

# dir = "20240715223804-8-24-0.95-0.9"
# file = "resultall_20240715223804.csv"

# dir = "20240715224219-8-24-0.3-0.9"
# file = "resultall_20240715224219.csv"

# dir = "20240715224701-8-24-0.98-0.99"
# file = "resultall_20240715224701.csv"

# dir = "20240715225304-8-24-1-0.75"
# file = "resultall_20240715225304.csv"

# dir = "20240715232740-8-96-0.8-0.9"
# file = "resultall_20240715232740.csv"

# dir = "20240715233404-8-96-0.7-0.75"
# file = "resultall_20240715233404.csv"

# dir = "20240715234319-8-384-0.7-0.75"
# file = "resultall_20240715234319.csv"

# dir = "20240716035725-8-194-0.7-0.75"
# file = "resultall_20240716035725.csv"

dir = "20240716041251-8-194-0.5-1"
file = "resultall_20240716041251.csv"

# dir = "20240716041617-8-194-1-0.5"
# file = "resultall_20240716041617.csv"

# dir = "20240716042411-8-194-1-0.2"
# file = "resultall_20240716042411.csv"

# dir = "20240716044244-8-194-0.8-0.2"
# file = "resultall_20240716044244.csv"

# dir = "20240716045020-8-194-0.65-0.35"
# file = "resultall_20240716045020.csv"

dir = "20240716062920-8-194-0.7-0.5"
file = "resultall_20240716062920.csv"

####################################

dir = "20240716145045-8-194-0.05-1"
file = "resultall_20240716145045.csv"

dir = "20240716143339-8-194-0.1-1"
file = "resultall_20240716143339.csv"

dir = "20240716143827-8-194-0.3-1"
file = "resultall_20240716143827.csv"

dir = "20240716144045-8-194-0.5-1"
file = "resultall_20240716144045.csv"

dir = "20240716144207-8-194-0.7-1"
file = "resultall_20240716144207.csv"

dir = "20240716144534-8-194-0.9-1"
file = "resultall_20240716144534.csv"

dir = "20240716144847-8-194-1-1"
file = "resultall_20240716144847.csv"

##

dir = "20240716145310-8-194-0.05-0.9"
file = "resultall_20240716145310.csv"

dir = "20240716145612-8-194-0.1-0.9"
file = "resultall_20240716145612.csv"

dir = "20240716145906-8-194-0.3-0.9"
file = "resultall_20240716145906.csv"

dir = "20240716150244-8-194-0.5-0.9"
file = "resultall_20240716150244.csv"

dir = "20240716150658-8-194-0.7-0.9"
file = "resultall_20240716150658.csv"

dir = "20240716151149-8-194-0.9-0.9"
file = "resultall_20240716151149.csv"

dir = "20240716151700-8-194-1-0.9"
file = "resultall_20240716151700.csv"

## 
dir = "20240716155602-8-194-0.05-0.5"
file = "resultall_20240716155602.csv"

dir = "20240716155052-8-194-0.1-0.5"
file = "resultall_20240716155052.csv"

dir = "20240716154521-8-194-0.3-0.5"
file = "resultall_20240716154521.csv"

dir = "20240716154006-8-194-0.5-0.5"
file = "resultall_20240716154006.csv"

dir = "20240716153246-8-194-0.7-0.5"
file = "resultall_20240716153246.csv"

dir = "20240716152736-8-194-0.9-0.5"
file = "resultall_20240716152736.csv"

dir = "20240716152245-8-194-1-0.5"
file = "resultall_20240716152245.csv"

##

dir = "20240716160103-8-194-1-0.1"
file = "resultall_20240716160103.csv"

dir = "20240716160604-8-194-0.9-0.1"
file = "resultall_20240716160604.csv"

dir = "20240716161151-8-194-0.7-0.1"
file = "resultall_20240716161151.csv"

dir = "20240716161808-8-194-0.5-0.1"
file = "resultall_20240716161808.csv"

dir = "20240716162356-8-194-0.3-0.1"
file = "resultall_20240716162356.csv"

dir = "20240716162912-8-194-0.1-0.1"
file = "resultall_20240716162912.csv"

dir = "20240716163616-8-194-0.05-0.1"
file = "resultall_20240716163616.csv"

##

dir = "20240716164204-8-194-1-0"
file = "resultall_20240716164204.csv"

dir = "20240716164826-8-194-0.9-0"
file = "resultall_20240716164826.csv"

dir = "20240716165344-8-194-0.7-0"
file = "resultall_20240716165344.csv"

dir = "20240716165822-8-194-0.5-0"
file = "resultall_20240716165822.csv"

dir = "20240716170310-8-194-0.3-0"
file = "resultall_20240716170310.csv"

dir = "20240716170844-8-194-0.1-0"
file = "resultall_20240716170844.csv"

dir = "20240716171551-8-194-0.05-0"
file = "resultall_20240716171551.csv"

########################### DIRECTORY CONCAT  #########################################
out_dir = f"results/{dir}/all"
parent_out = os.path.join("/Users/mlnick/Documents/University/Git/Tsukuba-Master/ExpDesign/report-2/", out_dir)

########################### LOAD EXP DATA     #########################################
# Load the CSV data
data = pd.read_csv(f"{parent_out}/{file}")

runtime_info = data.iloc[-1]
if "Runtime" in runtime_info.values[0]:
    runtime_line = data.iloc[-1, 0]
    data = data.iloc[:-1]  # Remove the last line

# Identify unique policies and problems
policies = data['Selection Policy'].unique()
problems = data['Problem'].unique()

stats_file = os.path.join(parent_out, f"{dir}_all_statistics.txt")
buffer = [""]

########################### STAT COMPUT.                    ###########################
# Compute statistics
statistics = data.groupby(['Selection Policy', 'Problem'])['best'].agg(['mean', 'median', 'std', 'var'])
print(f"Essential statistics for {dir}: ")
print(statistics, "\n")

# Save statistics and runtime to a file as .txt
with open(stats_file, 'w') as f:
    f.write(str(runtime_line))
    f.write(statistics.to_string())
    f.write("\n\n")
    print(f"Statistics and experiment runtime saved to {stats_file}")

''' # THIS IS NOT NECESSARY DUE TO THE DATA NOT FOLLOWING NORMAL DISTRIBUTION! WHAT A WASTE!
### [YET NOT READY FOR ANALYSIS] -> MIGRATE TO THE BOTTOM
########################### ANOVA FOR PROBLEM & SELECTION   ########################### 
# For an ANOVA test across different selection policies for a single problem
for prob in data['Problem'].unique():
    samples = [data[(data['Problem'] == prob) & (data['Selection Policy'] == policy)]['best'] for policy in data['Selection Policy'].unique()]
    f_value, p_value = stats.f_oneway(*samples)
    buffer += f"ANOVA results for Problem {prob}: F={f_value}, p={p_value}"
    print(f"ANOVA results for Problem {prob}: F={f_value}, p={p_value}")

# Save statistics, ANOVA, and runtime to a file as .txt
with open(stats_file, 'w') as f:
    f.write(str(runtime_line))
    f.write('\n'.join(buffer))
    f.write(statistics.to_string())
    f.write("\n\n")
    print(f"Statistics and experiment runtime saved to {stats_file}")

### [YET NOT READY FOR ANALYSIS] -> MIGRATE TO THE BOTTOM
# Perform Tukey HSD post-hoc test for a particular problems
problem_data = data[data['Problem'] == 0]  # example for Problem 0
tukey_results = pairwise_tukeyhsd(endog=problem_data['best'], groups=problem_data['Selection Policy'], alpha=0.05)
print(tukey_results)
'''

### FOR FUTURE CONVENIENCE
# Recalculate and save the statistics for later use as .csv
stats_df = data.groupby(['Problem', 'Selection Policy'])['best'].agg(['mean', 'median', 'std', 'var']).reset_index()
stats_file_csv = os.path.join(parent_out, f"{dir}_all_statistics.csv")
# print(stats_df)
stats_df.to_csv(stats_file_csv, index=False)
print(f"Statistics saved to {stats_file_csv}\n")

# Estimate variance
grouped_data = data.groupby(['Problem', 'Selection Policy'])
variance_df = grouped_data['best'].var()
print(f"Estimated variance for {dir}: ")
print(variance_df, "\n")

### UNCOMMENT FROM HERE


########################### VISUALIZATION OF PROGRESSION    ###########################
# Prepare a large figure to hold all subplots
fig, axes = plt.subplots(len(problems), len(policies), figsize=(15, 10), constrained_layout=True)

for i, prob in enumerate(problems):
    for j, policy in enumerate(policies):
        ax = axes[i, j] if len(problems) > 1 and len(policies) > 1 else axes[max(i, j)]
        
        # Filter data for the current problem and policy
        subset = data[(data['Problem'] == prob) & (data['Selection Policy'] == policy)]
        
        # Plot the progression of 'best' values over 'neval'
        sns.lineplot(x='neval', y='best', data=subset, ax=ax, marker='o')
        ax.set_title(f'Problem {prob}, Policy {policy}')
        ax.set_yscale('log')  # Use logarithmic scale if values vary widely

# Add overall figure title
fig.suptitle('Progression of Best Values by Policy and Problem', fontsize=16)
fig_save = plt.gcf()
# plt.show()

progression_dir = os.path.join(parent_out, f"{dir}_progression.png")
fig_save.savefig(progression_dir)
print(f"Progression plot saved to {progression_dir}.\n")

########################### HISTO BOXES SHOWCASE     ##################################
# Plot boxplots in a separate figure to analyze final best values distribution
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(x='Problem', y='best', hue='Selection Policy', data=data, ax=ax)
ax.set_yscale('log')
ax.set_title('Distribution of Final Best Values by Policy and Problem')
plt.legend(title='Selection Policy')
fig_save = plt.gcf()
# plt.show()

boxplot_dir = os.path.join(parent_out, f"{dir}_boxplot.png")
fig_save.savefig(boxplot_dir)

print(f"Boxplot saved to {boxplot_dir}.\n")

########################### QQ PLOTS FOR NORMALITY   ##################################
# Q-Q plots - done for each group to check normality
qqplot_file = ""
qq_plot_dir = os.path.join(parent_out, "qq-plots/")
if not os.path.exists(qq_plot_dir): 
    os.mkdir(qq_plot_dir)

for prob in problems:
    for policy in policies:
        subset = data[(data['Problem'] == prob) & (data['Selection Policy'] == policy)]
        fig, ax = plt.subplots()
        stats.probplot(subset['best'], dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: Problem {int(prob)}, Policy {int(policy)}')
        # plt.show() # Optionally show the figure

        qqplot_file = os.path.join(qq_plot_dir, f"{dir}_qqplot_A{policy}-P{prob}.png")
        plt.savefig(qqplot_file)
        plt.close()

print(f"Q-Q plot saved to \"{qq_plot_dir}/{dir}_qqplot_A<X>-P<Y>.png\"\n")


########################### SHAPIRO WILK TEST      ####################################

# Load the data
# data = pd.read_csv('stats_file_csv.csv')  # replace with actual file path

# Initialize a DataFrame to store the Shapiro-Wilk test results
print(f"Shapiro-Wilk test results for {dir}: ")
results = pd.DataFrame(columns=['Problem', 'Selection Policy', 'SW Statistic', 'p-value'])

# Perform the Shapiro-Wilk test for each combination of problem and selection policy

# parameters = ['Problem', 'Selection Policy', 'SW Statistic', 'p-value']
# df = pd.DataFrame(columns=parameters)

for problem in data['Problem'].unique():
    for policy in data['Selection Policy'].unique():
        subset = data[(data['Problem'] == problem) & (data['Selection Policy'] == policy)]['best']
        sw_statistic, p_value = stats.shapiro(subset)
        results = results._append({'Problem': problem, 'Selection Policy': policy,
                                  'SW Statistic': sw_statistic, 'p-value': p_value}, ignore_index=True)
        # results = pd.concat([df, pd.DataFrame.from_records([{'Problem': problem, 'Selection Policy': policy,
                                 #  'SW Statistic': sw_statistic, 'p-value': p_value}])], ignore_index=True)

# Print the results
print(results, "\n")

# Saving the results to a CSV file
sw_file_csv = os.path.join(parent_out, f"{dir}_all_sw.csv")
results.to_csv(sw_file_csv, index=False)
print(f"Shapiro-Wilk results saved to {sw_file_csv}\n")



###########################  KRUSKAL-WALLIS TEST      ####################################

# Group data by 'Selection Policy'
grouped_data = data.groupby('Selection Policy')

# Extract data for each policy
data_groups = [group['best'].values for name, group in grouped_data]

# Perform the Kruskal-Wallis test
stat, p_value = kruskal(*data_groups)

print("Kruskal-Wallis H-test test-statistic:", stat)
print("P-value:", p_value)

# Interpret the result
if p_value < 0.05:
    print("There is a statistically significant difference between the groups.")
else:
    print("No significant difference found between the groups.")

###########################  VISUALIZATION  ####################################

# Plotting raw
plt.figure(figsize=(12, 6))
sns.boxplot(x='Selection Policy', y='best', data=data)
plt.title('Raw Results by Selection Policy')
plt.ylabel('Best Values')
plt.xlabel('Selection Policy')
fig_save = plt.gcf()
# plt.show()

comparison_dir = os.path.join(parent_out, f"{dir}_comparison.png")
fig_save.savefig(comparison_dir)
print(f"Policy compatison plot in raw results saved to {comparison_dir}.\n")

# Log transformation for visualization
data['log_best'] = np.log(data['best'] + 1)  # log transformation

# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(x='Selection Policy', y='log_best', data=data)
plt.title('Normalized Results by Selection Policy')
plt.ylabel('Log of Best Values')
plt.xlabel('Selection Policy')
fig_save = plt.gcf()
plt.show()

log_comparison_dir = os.path.join(parent_out, f"{dir}_log_comparison.png")
fig_save.savefig(log_comparison_dir)
print(f"Policy compatison plot in log scale saved to {log_comparison_dir}.\n")


########################### POST HOC  ####################################
#'data' is loaded and contains the 'Selection Policy' and 'best' columns
data['log_best'] = np.log(data['best'])

# Post-hoc analysis after Kruskal-Wallis - Group by Selection Policy
print("Post-hoc analysis after Kruskal-Wallis - Group by Selection Policy")
posthoc_results = sp.posthoc_dunn(data, val_col='log_best', group_col='Selection Policy', p_adjust='bonferroni')
print(posthoc_results, "\n")

# Post-hoc analysis advanced - Group by Selection Policy and Problem (Benchmark)
# Here, 'data' is loaded and contains 'Selection Policy', 'problem', and 'log_best'
print("Post-hoc analysis advanced - Group by Selection Policy and Problem (Benchmark)")
unique_problems = data['Problem'].unique()
results = pd.DataFrame()

for problem in unique_problems:
    subset = data[data['Problem'] == problem]
    if subset.shape[0] < 20:  # Ensure there's enough data to perform the test
        continue
    
    # Perform Kruskal-Wallis test for each problem
    stat, p_value = kruskal(*[group["log_best"].values for name, group in subset.groupby('Selection Policy')])
    
    # If significant, perform Dunn's post-hoc test
    if p_value < 0.05:
        posthoc = sp.posthoc_dunn(subset, val_col='log_best', group_col='Selection Policy', p_adjust='bonferroni')
        posthoc_df = pd.DataFrame(posthoc)
        posthoc_df.reset_index(inplace=True)  # Reset index to turn policy labels into a column
        posthoc_df.rename(columns={'index': 'Policy N'}, inplace=True)
        posthoc_df['Problem'] = problem  # Add problem identifier
        results = pd.concat([results, posthoc_df])
        

# Output results for each benchmark
print(results, "\n")

############################## FINAL BOXPLOT VISUALIZATION #########################

# Assuming log transformation is already applied to 'best' values
data['log_best'] = np.log(data['best'] + 1)  # to handle zero or negative values if any

plt.figure(figsize=(12, 8))
sns.boxplot(x='Problem', y='log_best', hue='Selection Policy', data=data)
plt.title('Distribution of Log Best Values by Selection Policy Across Problems')
plt.xlabel('Problem')
plt.ylabel('Log of Best Values')
plt.legend(title='Selection Policy')
plt.xticks(rotation=45)  # Rotates the labels on the x-axis for better readability
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap

fig_save = plt.gcf()
plt.show()

final_boxplot_dir = os.path.join(parent_out, f"{dir}_final_boxplot_comparison.png")
fig_save.savefig(final_boxplot_dir)
print(f"Boxplot comparison in log scale saved to {final_boxplot_dir}.\n")

############################## POST HOC VISUALIZATION #########################
posthoc_heatmaps_file = ""
posthoc_heatmaps_dir = os.path.join(parent_out, "posthoc-heatmaps/")
if not os.path.exists(posthoc_heatmaps_dir): 
    os.mkdir(posthoc_heatmaps_dir)

# For Selection Policy
plt.figure(figsize=(10, 8))
sns.heatmap(posthoc_results, annot=True, fmt=".2e", cmap='coolwarm', cbar=True)
plt.title('P-values from Dunn’s Post-hoc Test Across Selection Policies')
plt.xlabel('Selection Policy')
plt.ylabel('Selection Policy')
fig_save = plt.gcf()
plt.show()

posthoc_heatmaps_file = os.path.join(posthoc_heatmaps_dir, f"{dir}_main_hosthoc_heatmap.png")
plt.savefig(posthoc_heatmaps_file)
plt.close()

posthoc_dir = os.path.join(parent_out, f"{dir}_hosthoc_heatmap.png")
fig_save.savefig(posthoc_dir)
print(f"HostHoc comparison in heatmap saved to {posthoc_dir}.\n")

# For each Problem
# Visualization of results for each problem
for problem in results['Problem'].unique():
    problem_data = results[results['Problem'] == problem]
    if not problem_data.empty:
        pivot = problem_data.pivot(columns = 'Policy N')  # Adjust
        plt.figure(figsize=(8, 6))
        # heatmap_data = problem_data.pivot_table(index='Selection Policy', columns='Problem', values='p-value')
        sns.heatmap(pivot, annot=True, fmt=".2e", cmap='coolwarm', cbar=True)
        plt.title(f'P-values from Dunn’s Post-hoc Test for Problem {problem}')
        plt.xlabel('Selection Policy')
        plt.ylabel('Selection Policy')
        fig_save = plt.gcf()
        # plt.show()

        posthoc_heatmaps_file = os.path.join(posthoc_heatmaps_dir, f"{dir}_P{problem}_hosthoc_heatmap.png")
        fig_save.savefig(posthoc_heatmaps_file)
print(f"HostHoc comparison in heatmap saved to {posthoc_heatmaps_file}, x{problems} files.\n")


###################### SOME OPTIONAL CHECKS ON CONFIDENCE ######################
# Doublecheck on confidence intervals

mean = data['log_best'].mean()
std_dev = data['log_best'].std()
n = len(data['log_best'])  # Total sample size

# Compute the 95% confidence interval
confidence_interval = stats.norm.interval(0.95, loc=mean, scale=std_dev / np.sqrt(n))

print(f"95% confidence interval for the mean: {confidence_interval}")

###################### COHEN'S CHECK ON EFFECT SIZE #####################

# Cohen's d for independent samples
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

# Calculate Cohen's d for each pair of selection policies
cohen_d_values = {}
groups = data.groupby('Selection Policy')['log_best']
for (label1, group1), (label2, group2) in combinations(groups, 2):
    d = cohen_d(group1, group2)
    cohen_d_values[(label1, label2)] = d

print("Cohen's d values between policies:", cohen_d_values, "\n")

########################## CHECK ON INDEPENDENCE ########################
'''
# Plotting best results across repetitions for a given policy and problem
sns.lineplot(data=data[data['Selection Policy'] == '0'][data['Problem'] == '0'], x='Repetition', y='best')
plt.title('Performance Over Repetitions for Policy 0 on Problem 0')
plt.xlabel('Repetition Number')
plt.ylabel('Best Result')
plt.show()
'''

# Chi-square independence test - DATA IS DEPENDENT
contingency_table = pd.crosstab(data['Selection Policy'], data['Problem'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("Chi-square test p-value:", p)