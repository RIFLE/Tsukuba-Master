import numpy as np
import scipy.stats as stats

def simulate_kruskal(n, num_groups, effect_size, iterations):
    power_count = 0
    for _ in range(iterations):
        # Create data for each group with some effect
        groups = [np.random.normal(loc=i*effect_size, scale=1, size=n) for i in range(num_groups)]
        _, p_value = stats.kruskal(*groups)
        if p_value < 0.05:
            power_count += 1
    return power_count / iterations

# Set parameters
num_groups = 3  # As per three selection policies
desired_power = 0.8
alpha = 0.05
effect_size = 0.05  # Hypothetical effect size, adjust based on expected differences
iterations = 1000  # More iterations, more accurate power estimation

# Estimate required sample size
sample_size = 10  # Start with a low number
while True:
    power = simulate_kruskal(sample_size, num_groups, effect_size, iterations)
    print(f"Sample size: {sample_size}, Estimated Power: {power}")
    if power >= desired_power:
        break
    sample_size += 5

print(f"Required sample size to achieve {desired_power*100}% power is approximately {sample_size}")
