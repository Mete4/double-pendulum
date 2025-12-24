
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import physics_kernel as pk
import os

from generate_visuals import VISUAL_ICS

os.makedirs("visuals", exist_ok=True)

# Render Angle Space Periodicity Map with labels
print("Rendering Angle Space Periodicity Map with labels...")
angle_grid, angle_extent = pk.run_fractal_gen(mode=0, res=1024, t_max=20.0, dt=0.01, metric="poincare")
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(angle_grid, extent=angle_extent, origin='lower', cmap=plt.cm.twilight_shifted)
ax.set_title("Double Pendulum Periodicity Map (Angle Space)", fontsize=16)
ax.set_xlabel(r"$\theta_1$ (rad)")
ax.set_ylabel(r"$\theta_2$ (rad)")

# Overlay points of interest
for name, coords in VISUAL_ICS.items():
	ax.plot(coords[0], coords[1], 'o', color='white', markersize=7, markeredgecolor='black', zorder=10)
	ax.text(coords[0], coords[1] + 0.05, f" {name}", color='white',
			fontsize=10, fontweight='bold', verticalalignment='bottom',
			bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2), zorder=11)

plt.savefig("visuals/periodicity_angle_label.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved to visuals/periodicity_angle_label.png")

# Render Momentum Space Periodicity Map 
print("Rendering Momentum Space Periodicity Map...")
mom_grid, mom_extent = pk.run_fractal_gen(mode=1, res=1024, t_max=20.0, dt=0.01, metric="poincare")
plt.figure(figsize=(12, 12))
plt.imshow(mom_grid, extent=mom_extent, origin='lower', cmap=plt.cm.viridis)
plt.title("Double Pendulum Periodicity Map (Momentum Space)", fontsize=16)
plt.xlabel(r"$\omega_1$ (rad/s)")
plt.ylabel(r"$\omega_2$ (rad/s)")
plt.savefig("visuals/periodicity_momentum.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved to visuals/periodicity_momentum.png")
