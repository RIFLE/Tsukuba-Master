import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 200  # Number of grid points
dx = 1.0  # Space step
dt = dx / 2  # Time step, satisfying CFL condition
T = 50  # Total time steps for simulation
Ey = np.zeros(L)
Bz = np.zeros(L)
J = np.zeros(L)  # Current density array

# Initial conditions for fields and particles
Ey[L//2] = 1  # Initial impulse in the electric field
J[L//2] = 1  # Approximate current due to particle movement

for t in range(T):
    # Update magnetic field
    Bz[:-1] -= (dt/dx) * (Ey[1:] - Ey[:-1])
    Bz[-1] -= (dt/dx) * (Ey[0] - Ey[-1])  # Periodic boundary for Bz

    # Update electric field
    Ey[1:] -= (dt/dx) * (Bz[1:] - Bz[:-1]) + J[1:] * dt
    Ey[0] -= (dt/dx) * (Bz[0] - Bz[-1]) + J[0] * dt  # Periodic boundary for Ey

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(Ey, label='Electric Field (Ey)')
plt.plot(Bz, label='Magnetic Field (Bz)')
plt.title('FDTD Simulation with Current')
plt.xlabel('Grid Index')
plt.ylabel('Field Amplitude')
plt.legend()
plt.show()
