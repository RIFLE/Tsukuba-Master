from statsmodels.stats.power import FTestAnovaPower

# Settings for power analysis
effect_size = 0.05  # Cohen's f
alpha = 0.05  # Significance level
power = 0.8  # Desired power

# Create a power analysis object
analysis = FTestAnovaPower()

# Calculate sample size
sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, k_groups=3)
print(f"Required sample size: {sample_size}")