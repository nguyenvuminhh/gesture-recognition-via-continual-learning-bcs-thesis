import numpy as np
import matplotlib.pyplot as plt

# Define example gradients
g_replay = np.array([-3, 1])
g_current = np.array([4, 4])

# A-GEM projection
dot = np.dot(g_current, g_replay)
norm_sq = np.dot(g_replay, g_replay)
scalar = dot / (norm_sq + 1e-8)
g_total = g_current - scalar * g_replay

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("A-GEM Gradient Projection")

# Plot each vector
ax.quiver(0, 0, g_replay[0], g_replay[1], color='red', angles='xy', scale_units='xy', scale=1, label='g_replay')
ax.quiver(0, 0, g_current[0], g_current[1], color='blue', angles='xy', scale_units='xy', scale=1, label='g_current')
ax.quiver(0, 0, g_total[0], g_total[1], color='green', angles='xy', scale_units='xy', scale=1, label='g_total (projected)')

ax.legend()

plt.savefig('plotting_scripts/plots/agem_gradient.png')

