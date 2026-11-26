import sys
from pathlib import Path
_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

import numpy as np
import matplotlib.pyplot as plt
from configs.aircraft import WING, MASSES, ENGINE
from wp41.weight import fuel_per_span


g = 9.81
n = 2.5                       
mach, alpha = 0.30, 10.0          
lift_dist = Path(__file__).resolve().parents[1] / "out" / "wp41" / "lift_distribution.txt"  


def load_cl(path, mach, alpha):
    machs = alphas = None
    with open(path) as f:
        for line in f:
            if line.startswith("# mach:"):
                machs = np.array(line.split(":")[1].split(), float)
            elif line.startswith("# alpha_deg:"):
                alphas = np.array(line.split(":")[1].split(), float)
    data = np.loadtxt(path, comments="#")
    i = 0
    for i in range(len(machs)):
        if machs[i] == mach and alphas[i] == alpha:
            idx = i
    return data[:, 0], data[:, 1], data[:, 2 + idx]


def tip_integral(f, y):
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(y)
    out = np.zeros_like(f)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


y, chord, cl = load_cl(lift_dist, mach, alpha)
w_fuel = fuel_per_span(y, MASSES.fuel_kg*g, tolerance=.01)


shape = chord * cl
lift_half = n * MASSES.mtom_kg * g / 2.0
lift = shape * lift_half / np.trapezoid(shape, y)
q = lift-n*w_fuel

P_eng = n * MASSES.engine_installed_kg * g
a = 4.4


shear_dist = tip_integral(q, y)
moment = tip_integral(shear_dist, y)
for i in range(len(moment)):    
    if y[i] > a:
        moment[i] -= P_eng * (y[i] - a)
        shear_dist[i] -= P_eng


fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(y, lift);  ax[0].set_ylabel("lift [N/m]")
ax[1].plot(y, shear_dist); ax[1].set_ylabel("shear [N]")
ax[2].plot(y, moment); ax[2].set_ylabel("M [Nm]")
ax[2].set_xlabel("y [m]")
plt.show()