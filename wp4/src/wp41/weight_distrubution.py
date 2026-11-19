import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Script is at wp4/src/wp41/<this_file>.py -> parents[2] is wp4/
out_dir = Path(__file__).resolve().parents[2] / "out" / "wp41"

weight_distribution = np.array([])
points = np.array([])

#Wing planform
c_r = 0.5  # root chord length
c_t = 0.2  # tip chord length
b = 10.0  # wing span

W_FT = 1000.0 #Total fuel weight carried in the wings (both halfwings)

a = -3*W_FT/(b*(c_r**2 + c_r*c_t + c_t**2))


for y in np.arange(0, b/2, 0.01):
    w = a*(-(c_r-c_t)/(b/2)*y+c_r)**2
    points = np.append(points, y)
    weight_distribution = np.append(weight_distribution, w)

"""For plotting the weight distribution:

fig, ax = plt.subplots(figsize=(10, 4))

# Beam
ax.plot([0, b/2], [0, 0], color="black", linewidth=4)

# Root support
ax.plot([0, 0], [-0.4, 0.4], color="black", linewidth=8)

# Scale for plotting
scale = max(np.abs(weight_distribution)) / 3

offset = 0

# Distributed load arrows
for y, w in zip(points[::20], weight_distribution[::20]):
    ax.arrow(
        y,
        offset + abs(w)/scale,
        0,
        -abs(w)/scale,
        head_width=0.06,
        head_length=0.08,
        length_includes_head=True,
        color="tab:blue"
    )

# Envelope through arrow tails
ax.plot(
    points,
    offset + np.abs(weight_distribution)/scale,
    color="tab:blue",
    linewidth=2
)

ax.fill_between(
    points,
    offset + np.abs(weight_distribution)/scale,
    0,
    color="tab:blue",
    alpha=0.2
)

ax.set_xlabel("Spanwise location y [m]")
ax.set_title("Fuel Weight Distribution")

plt.tight_layout()
plt.show() """

np.savetxt(out_dir / "weight.csv", np.column_stack((points, weight_distribution)))