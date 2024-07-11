import numpy as np
import matplotlib.pyplot as plt

# Initialization
dt = 0.5e-3
qm = 100
n_iter = 1000
initial_v = np.array([2., 3., 4.])
initial_x = np.array([10., 12., 1.])
b_field = np.array([5., 7., 8.])
e_field = np.array([1., 2., 1.])
x_mem = np.zeros((n_iter, 3))

# Calculate s and t vectors
t_vec = qm * b_field * 0.5 * dt  # Magnetic field term for velocity rotation
s_vec = 2. * t_vec / (1. + np.dot(t_vec, t_vec))  # Scaling factor for the rotation

# Boris' algorithm
v = initial_v.copy()
x = initial_x.copy()
for idx in range(n_iter):
    # Step 3: Half-step velocity update for electric field
    v_minus = v + qm * e_field * 0.5 * dt

    # Step 4: Rotation due to magnetic field
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Step 5: Complete the velocity step with the second half-step for electric field
    v = v_plus + qm * e_field * 0.5 * dt

    # Step 6: Update position
    x += v * dt

    # Store the results in x_mem for plotting
    x_mem[idx, :] = x

# Plotting
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 6))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot(x_mem[:, 0], x_mem[:, 1], x_mem[:, 2])
plt.show()

