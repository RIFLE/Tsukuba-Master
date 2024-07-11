import numpy as np
import matplotlib.pyplot as plt

# Parameters
v = 1  # Wave speed
dx = 1  # Spatial resolution (Delta x)
dt = 1  # Time step (Delta t)
k = np.linspace(-np.pi, np.pi, 500)  # Wave number range from -pi to pi

# Analytical Dispersion Relation
omega_analytical = v * k

# Numerical Dispersion Relations for different v*dt/dx ratios
ratios = [0.5, 0.25, 0.1]
colors = ['r', 'g', 'b']  # Colors for different plots

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k, omega_analytical, 'k-', label='Analytical (v*k)', linewidth=2)

# Calculate and plot numerical dispersion relation for each ratio
for ratio, color in zip(ratios, colors):
    omega_numerical = 2 * np.arcsin(ratio * np.sin(k / 2)) / dt
    plt.plot(k, omega_numerical, color, label=f'Numerical: v*dt/dx = {ratio}', linewidth=2)

# Formatting the plot
plt.xlabel('Wave number k')
plt.ylabel('Angular frequency ω')
plt.title('Dispersion Relation Comparison')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
