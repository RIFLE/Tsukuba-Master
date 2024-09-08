import numpy as np
import math

# Loading the results from file
loaded_arrays = np.load('code_py/Results.npy', allow_pickle=True).item()

# Calculating standard deviation
def compute_std_dev(data):
    if len(data) < 2:
        raise ValueError("Standard deviation requires at least two data points.")
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    std_dev = math.sqrt(variance)
    return std_dev

# Defining confidence levels for each metric, i.e. 90 and 95%
z_scores = {
    "HUGO-10-ACCURACY": 1.96,
    "HUGO-10-IOU": 1.645,
    "HUGO-25-ACCURACY": 1.96,
    "HUGO-25-IOU": 1.645,
    "WOW-10-ACCURACY": 1.96,
    "WOW-10-IOU": 1.645,
    "WOW-25-ACCURACY": 1.96,
    "WOW-25-IOU": 1.645
}

# Computing standard deviations and confidence intervals
results = {}
for key, data in loaded_arrays.items():
    std_dev = compute_std_dev(data)
    mean = sum(data) / len(data)
    z_score = z_scores[key]
    # Calculating the margin of error
    margin_of_error = z_score * (std_dev / math.sqrt(len(data)))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    results[key] = (mean, std_dev, lower_bound, upper_bound, margin_of_error)

# Saving the results to a text file
output_filename = 'Deviations_and_intervals.txt'
with open(output_filename, 'w') as file:
    for name, (mean, std, lower, upper, margin) in results.items():
        file.write(f"{name}: Mean = {mean:.4f}, Std Dev = {std:.4f}, MoE = ~{margin:.5f}, CI = [{lower:.4f}, {upper:.4f}]\n")
print(f"Results written to {output_filename}")
