import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 200  # Number of grid points
dx = 1.0  # Space step, normalized to 1
dt = dx / 2  # Time step, satisfies CFL condition for stability
T = 1  # Number of time steps to simulate
Ey = np.zeros(L)  # Electric field
Bz = np.zeros(L)  # Magnetic field
Ey[L//2] = 1  # Initial impulse in Electric field

# Simulation
for n in range(T):
    Bz[:-1] = Bz[:-1] - (dt/dx) * (Ey[1:] - Ey[:-1])
    Bz[-1] = Bz[-1] - (dt/dx) * (Ey[0] - Ey[-1])  # Periodic boundary condition
    Ey[1:] = Ey[1:] - (dt/dx) * (Bz[1:] - Bz[:-1])
    Ey[0] = Ey[0] - (dt/dx) * (Bz[0] - Bz[-1])  # Periodic boundary condition

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, 1, L), Ey, label='Electric Field (Ey)')
plt.title('Electric Field at Final Time Step')
plt.xlabel('Normalized Grid Position')
plt.ylabel('Field Amplitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, 1, L), Bz, color='red', label='Magnetic Field (Bz)')
plt.title('Magnetic Field at Final Time Step')
plt.xlabel('Normalized Grid Position')
plt.grid(True)

plt.tight_layout()
plt.show()
