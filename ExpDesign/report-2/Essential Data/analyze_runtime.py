import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load runtime data
data = pd.read_csv('runtime_stats.csv')

# Compute average runtime for each algorithm and configuration
average_runtimes = data.groupby(['F', 'CR']).agg({
    'overall': 'mean',
    'alg1': 'mean',
    'alg2': 'mean',
    'alg3': 'mean'
}).reset_index()

# Save the computed averages to a CSV file
average_runtimes.to_csv('average_runtimes.csv', index=False)
print("Average runtimes saved to 'average_runtimes.csv'")

# Plotting the runtime data for varying F values
plt.figure(figsize=(12, 8))
sns.lineplot(data=average_runtimes, x='F', y='alg1', marker='o', label='Algorithm 1')
sns.lineplot(data=average_runtimes, x='F', y='alg2', marker='o', label='Algorithm 2')
sns.lineplot(data=average_runtimes, x='F', y='alg3', marker='o', label='Algorithm 3')
plt.title('Average Runtime of Algorithms Across Different F values')
plt.xlabel('Differential Weight (F)')
plt.ylabel('Average Runtime (seconds)')
plt.legend(title='Algorithm')
plt.grid(True)
plt.show()

# Plotting the runtime data for varying CR values
plt.figure(figsize=(12, 8))
sns.lineplot(data=average_runtimes, x='CR', y='alg1', marker='o', label='Algorithm 1')
sns.lineplot(data=average_runtimes, x='CR', y='alg2', marker='o', label='Algorithm 2')
sns.lineplot(data=average_runtimes, x='CR', y='alg3', marker='o', label='Algorithm 3')
plt.title('Average Runtime of Algorithms Across Different CR values')
plt.xlabel('Crossover Probability (CR)')
plt.ylabel('Average Runtime (seconds)')
plt.legend(title='Algorithm')
plt.grid(True)
plt.show()
