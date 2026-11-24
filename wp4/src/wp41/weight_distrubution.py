import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path

from configs.aircraft import WING, MASSES # noqa: E402

# Script is at wp4/src/wp41/<this_file>.py -> parents[2] is wp4/
out_dir = Path(__file__).resolve().parents[2] / "out" / "wp41"

weight_distribution = np.array([])
points = np.array([])

#Wing planform from configs
b = math.sqrt(WING.aspect_ratio * WING.area_m2)  # wing span   
c_r = 2.0 * WING.area_m2 / (b * (1.0 + WING.taper))  # root chord length
c_t = WING.taper * c_r  # tip chord length
y_f = WING.fuselage_diameter_m / 2.0  # spanwise location of the fuselage edge

c_f = c_r - (WING.fuselage_diameter_m / 2.0) * (c_r - c_t) / (b / 2.0)  # root chord length outside fuselage

W_FT = MASSES.fuel_kg * 9.81 #Total fuel weight carried in the wings (both halfwings)

a = -3 * W_FT / (b - 2*y_f) / (c_f**2 + c_f*c_t + c_t**2)

for y in np.arange(0, y_f, 0.01):
    weight_distribution = np.append(weight_distribution, 0)

for y in np.arange(y_f, b/2, 0.01):
    w = a*(-(c_r-c_t)/(b/2)*y+c_r)**2
    points = np.append(points, y)
    weight_distribution = np.append(weight_distribution, w)

#For plotting the weight distribution:
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot([0, b / 2], [0, 0], color="black", linewidth=4)
ax.plot([0, 0], [-0.4, 0.4], color="black", linewidth=8)

scale = np.max(np.abs(weight_distribution)) / 3
offset = 0

for y, w in zip(points[::20], weight_distribution[::20]):
    if w != 0.0:  # no arrows inside the fuselage
        ax.arrow(y, offset + abs(w) / scale, 0, -abs(w) / scale,
                 head_width=0.06, head_length=0.08,
                 length_includes_head=True, color="tab:blue")

ax.plot(points, offset + np.abs(weight_distribution) / scale,
        color="tab:blue", linewidth=2)
ax.fill_between(points, offset + np.abs(weight_distribution) / scale, 0,
                color="tab:blue", alpha=0.2)

ax.set_xlabel("Spanwise location y [m]")
ax.set_title("Fuel Weight Distribution")

plt.tight_layout()
plt.show()

print(weight_distribution)

np.savetxt(
    out_dir / "weight.csv",
    np.column_stack((points, weight_distribution)),
    delimiter=",",
    header="y_m,w_N_per_m",
    comments="",
)