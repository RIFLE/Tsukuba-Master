import numpy as np
import matplotlib.pyplot as plt

# Initialization
dt = 0.5e-3  # Time step
qm = 100  # Charge-to-mass ratio
n_iter = 1000  # Number of iterations
initial_v = np.array([2., 3., 4.])  # Initial velocity
initial_x = np.array([10., 12., 1.])  # Initial position
b_field = np.array([5., 7., 8.])  # Magnetic field
e_field = np.array([1., 2., 1.])  # Electric field
x_mem = np.zeros((n_iter, 3))  # Memory for positions

# Calculate s and t vectors
t_vec = qm * b_field * dt / 2  # Magnetic field term for velocity rotation
s_vec = 2 * t_vec / (1 + np.dot(t_vec, t_vec))  # Scaling factor for the rotation

# Boris' algorithm
v = initial_v.copy()
x = initial_x.copy()
for idx in range(n_iter):
    # Step 3: Half-step velocity update for electric field
    v_minus = v + (e_field * qm * dt / 2)

    # Step 4: Rotation due to magnetic field
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Step 5: Complete the velocity step with the second half-step for electric field
    v = v_plus + (e_field * qm * dt / 2)

    # Step 6: Update position, using the average velocity over the timestep
    x += (v_plus + v_minus) / 2 * dt

    # Store the results in x_mem for plotting
    x_mem[idx, :] = x

# Plotting
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 6))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot(x_mem[:, 0], x_mem[:, 1], x_mem[:, 2], label='Particle Trajectory')
ax.legend()
plt.show()

