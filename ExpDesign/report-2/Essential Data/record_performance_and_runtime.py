import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_full_performance(directory, file_name):
    file_path = os.path.join('results', directory, 'best', file_name)
    data = pd.read_csv(file_path)
    data = data.dropna()  # Assuming the last row might not be part of the data
    return data

# List of all directories and files
configurations = [
    {"dir": "20240716145045-8-194-0.05-1", "file": "resultbest_20240716145045.csv"},
    {"dir": "20240716143339-8-194-0.1-1", "file": "resultbest_20240716143339.csv"},
    {"dir": "20240716143827-8-194-0.3-1", "file": "resultbest_20240716143827.csv"},
    {"dir": "20240716144045-8-194-0.5-1", "file": "resultbest_20240716144045.csv"},
    {"dir": "20240716144207-8-194-0.7-1", "file": "resultbest_20240716144207.csv"},
    {"dir": "20240716144534-8-194-0.9-1", "file": "resultbest_20240716144534.csv"},
    {"dir": "20240716144847-8-194-1-1", "file": "resultbest_20240716144847.csv"},
    {"dir": "20240716145310-8-194-0.05-0.9", "file": "resultbest_20240716145310.csv"},
    {"dir": "20240716145612-8-194-0.1-0.9", "file": "resultbest_20240716145612.csv"},
    {"dir": "20240716145906-8-194-0.3-0.9", "file": "resultbest_20240716145906.csv"},
    {"dir": "20240716150244-8-194-0.5-0.9", "file": "resultbest_20240716150244.csv"},
    {"dir": "20240716150658-8-194-0.7-0.9", "file": "resultbest_20240716150658.csv"},
    {"dir": "20240716151149-8-194-0.9-0.9", "file": "resultbest_20240716151149.csv"},
    {"dir": "20240716151700-8-194-1-0.9", "file": "resultbest_20240716151700.csv"},
    {"dir": "20240716155602-8-194-0.05-0.5", "file": "resultbest_20240716155602.csv"},
    {"dir": "20240716155052-8-194-0.1-0.5", "file": "resultbest_20240716155052.csv"},
    {"dir": "20240716154521-8-194-0.3-0.5", "file": "resultbest_20240716154521.csv"},
    {"dir": "20240716154006-8-194-0.5-0.5", "file": "resultbest_20240716154006.csv"},
    {"dir": "20240716153246-8-194-0.7-0.5", "file": "resultbest_20240716153246.csv"},
    {"dir": "20240716152736-8-194-0.9-0.5", "file": "resultbest_20240716152736.csv"},
    {"dir": "20240716152245-8-194-1-0.5", "file": "resultbest_20240716152245.csv"},
    {"dir": "20240716160103-8-194-1-0.1", "file": "resultbest_20240716160103.csv"},
    {"dir": "20240716160604-8-194-0.9-0.1", "file": "resultbest_20240716160604.csv"},
    {"dir": "20240716161151-8-194-0.7-0.1", "file": "resultbest_20240716161151.csv"},
    {"dir": "20240716161808-8-194-0.5-0.1", "file": "resultbest_20240716161808.csv"},
    {"dir": "20240716162356-8-194-0.3-0.1", "file": "resultbest_20240716162356.csv"},
    {"dir": "20240716162912-8-194-0.1-0.1", "file": "resultbest_20240716162912.csv"},
    {"dir": "20240716163616-8-194-0.05-0.1", "file": "resultbest_20240716163616.csv"},
    {"dir": "20240716164204-8-194-1-0", "file": "resultbest_20240716164204.csv"},
    {"dir": "20240716164826-8-194-0.9-0", "file": "resultbest_20240716164826.csv"},
    {"dir": "20240716165344-8-194-0.7-0", "file": "resultbest_20240716165344.csv"},
    {"dir": "20240716165822-8-194-0.5-0", "file": "resultbest_20240716165822.csv"},
    {"dir": "20240716170310-8-194-0.3-0", "file": "resultbest_20240716170310.csv"},
    {"dir": "20240716170844-8-194-0.1-0", "file": "resultbest_20240716170844.csv"},
    {"dir": "20240716171551-8-194-0.05-0", "file": "resultbest_20240716171551.csv"},
]
# Prepare a DataFrame to store all performance data
all_performance_data = []

for config in configurations:
    data = analyze_full_performance(config['dir'], config['file'])
    F = config['dir'].split('-')[3]
    CR = config['dir'].split('-')[4]
    data['F'] = float(F)
    data['CR'] = float(CR)
    all_performance_data.append(data)

# Concatenate all data into a single DataFrame
full_performance_df = pd.concat(all_performance_data)

# Calculate average performance for each F, CR, and Policy
average_performance = full_performance_df.groupby(['Selection Policy', 'F', 'CR'])['best'].mean().reset_index()

# Plotting with regression lines
sns.set(style="whitegrid")
plt.figure(figsize=(18, 6))
plt.suptitle('Average Algorithm Performance across CR values with Regression Lines')

# Create a separate plot for each selection policy
for policy in average_performance['Selection Policy'].unique():
    subset = average_performance[average_performance['Selection Policy'] == policy]
    plt.subplot(1, 3, int(policy) + 1)
    sns.regplot(x='CR', y='best', data=subset)
    plt.title(f'Selection Policy {policy}')
    plt.xlabel('Crossover Rate (CR)')
    plt.ylabel('Average Best Value')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
plt.show()

# Optionally, save the DataFrame to a CSV file for further analysis or reporting
average_performance.to_csv('full_algorithm_performance_across_CR.csv', index=False)
print("Full data on algorithm performance saved to 'full_algorithm_performance_across_CR.csv'")
'''
import os
import pandas as pd

def analyze_performance(directory, file_name):
    file_path = os.path.join('results', directory, 'best', file_name)
    data = pd.read_csv(file_path)
    runtime_info = data.iloc[-1, 0]  # Extract runtime information
    data = data.iloc[:-1]  # Remove the last info row
    min_best = data.groupby('Selection Policy')['best'].min().reset_index()
    return min_best, runtime_info

def extract_runtime_info(runtime_str):
    parts = runtime_str.split(',')
    runtime = {
        'overall': float(parts[3].split(':')[1].strip().split(' ')[0]),
        'alg1': float(parts[4].split('=')[1].strip('s')),
        'alg2': float(parts[5].split('=')[1].strip('s')),
        'alg3': float(parts[6].split('=')[1].strip('s'))
    }
    return runtime

# List of all directories and files
configurations = [
    {"dir": "20240716145045-8-194-0.05-1", "file": "resultbest_20240716145045.csv"},
    {"dir": "20240716143339-8-194-0.1-1", "file": "resultbest_20240716143339.csv"},
    {"dir": "20240716143827-8-194-0.3-1", "file": "resultbest_20240716143827.csv"},
    {"dir": "20240716144045-8-194-0.5-1", "file": "resultbest_20240716144045.csv"},
    {"dir": "20240716144207-8-194-0.7-1", "file": "resultbest_20240716144207.csv"},
    {"dir": "20240716144534-8-194-0.9-1", "file": "resultbest_20240716144534.csv"},
    {"dir": "20240716144847-8-194-1-1", "file": "resultbest_20240716144847.csv"},
    {"dir": "20240716145310-8-194-0.05-0.9", "file": "resultbest_20240716145310.csv"},
    {"dir": "20240716145612-8-194-0.1-0.9", "file": "resultbest_20240716145612.csv"},
    {"dir": "20240716145906-8-194-0.3-0.9", "file": "resultbest_20240716145906.csv"},
    {"dir": "20240716150244-8-194-0.5-0.9", "file": "resultbest_20240716150244.csv"},
    {"dir": "20240716150658-8-194-0.7-0.9", "file": "resultbest_20240716150658.csv"},
    {"dir": "20240716151149-8-194-0.9-0.9", "file": "resultbest_20240716151149.csv"},
    {"dir": "20240716151700-8-194-1-0.9", "file": "resultbest_20240716151700.csv"},
    {"dir": "20240716155602-8-194-0.05-0.5", "file": "resultbest_20240716155602.csv"},
    {"dir": "20240716155052-8-194-0.1-0.5", "file": "resultbest_20240716155052.csv"},
    {"dir": "20240716154521-8-194-0.3-0.5", "file": "resultbest_20240716154521.csv"},
    {"dir": "20240716154006-8-194-0.5-0.5", "file": "resultbest_20240716154006.csv"},
    {"dir": "20240716153246-8-194-0.7-0.5", "file": "resultbest_20240716153246.csv"},
    {"dir": "20240716152736-8-194-0.9-0.5", "file": "resultbest_20240716152736.csv"},
    {"dir": "20240716152245-8-194-1-0.5", "file": "resultbest_20240716152245.csv"},
    {"dir": "20240716160103-8-194-1-0.1", "file": "resultbest_20240716160103.csv"},
    {"dir": "20240716160604-8-194-0.9-0.1", "file": "resultbest_20240716160604.csv"},
    {"dir": "20240716161151-8-194-0.7-0.1", "file": "resultbest_20240716161151.csv"},
    {"dir": "20240716161808-8-194-0.5-0.1", "file": "resultbest_20240716161808.csv"},
    {"dir": "20240716162356-8-194-0.3-0.1", "file": "resultbest_20240716162356.csv"},
    {"dir": "20240716162912-8-194-0.1-0.1", "file": "resultbest_20240716162912.csv"},
    {"dir": "20240716163616-8-194-0.05-0.1", "file": "resultbest_20240716163616.csv"},
    {"dir": "20240716164204-8-194-1-0", "file": "resultbest_20240716164204.csv"},
    {"dir": "20240716164826-8-194-0.9-0", "file": "resultbest_20240716164826.csv"},
    {"dir": "20240716165344-8-194-0.7-0", "file": "resultbest_20240716165344.csv"},
    {"dir": "20240716165822-8-194-0.5-0", "file": "resultbest_20240716165822.csv"},
    {"dir": "20240716170310-8-194-0.3-0", "file": "resultbest_20240716170310.csv"},
    {"dir": "20240716170844-8-194-0.1-0", "file": "resultbest_20240716170844.csv"},
    {"dir": "20240716171551-8-194-0.05-0", "file": "resultbest_20240716171551.csv"},
]

best_performances = {}
runtime_data = []

for config in configurations:
    best_data, runtime_info = analyze_performance(config['dir'], config['file'])
    runtime = extract_runtime_info(runtime_info)
    runtime['F'] = config['dir'].split('-')[3]  # Extract F value
    runtime['CR'] = config['dir'].split('-')[4]  # Extract CR value
    runtime_data.append(runtime)
    
    for index, row in best_data.iterrows():
        policy = int(row['Selection Policy'])
        best_score = row['best']
        if policy not in best_performances or best_score < best_performances[policy]['best_score']:
            best_performances[policy] = {
                'best_score': best_score,
                'F': config['dir'].split('-')[3],
                'CR': config['dir'].split('-')[4]
            }

# Create dataframes from the performance and runtime data
performance_df = pd.DataFrame.from_dict(best_performances, orient='index')
runtime_df = pd.DataFrame(runtime_data)

# Save the performance data to a CSV file
performance_df.to_csv('performance_stats.csv', index_label='Selection Policy')
print("Performance statistics saved to 'performance_stats.csv'")

# Save the runtime data to another CSV file
runtime_df.to_csv('runtime_stats.csv', index=False)
print("Runtime statistics saved to 'runtime_stats.csv'")
'''