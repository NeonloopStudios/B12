import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path

from configs.aircraft import WING, MASSES
from wp41.weight_distrubution import W_FT # noqa: E402

# Script is at wp4/src/wp41/<this_file>.py -> parents[2] is wp4/
out_dir_case1 = Path(__file__).resolve().parents[2] / "out" / "wp41" / "case1"
out_dir_case2 = Path(__file__).resolve().parents[2] / "out" / "wp41" / "case2"
out_dir_case3 = Path(__file__).resolve().parents[2] / "out" / "wp41" / "case3"

#Wing planform from configs
b = math.sqrt(WING.aspect_ratio * WING.area_m2)  # wing span   
c_r = 2.0 * WING.area_m2 / (b * (1.0 + WING.taper))  # root chord length
c_t = WING.taper * c_r  # tip chord length
y_f = WING.fuselage_diameter_m / 2.0  # spanwise location of the fuselage edge

c_f = c_r - (WING.fuselage_diameter_m / 2.0) * (c_r - c_t) / (b / 2.0)  # root chord length outside fuselage

def weight_distribution(W_fuel):
    weight_distribution = np.array([])
    points = np.array([])
    a = 3 * W_FT / (b - 2*y_f) / (c_f**2 + c_f*c_t + c_t**2)

    for y in np.arange(0, y_f, 0.01):
        points = np.append(points, y)
        weight_distribution = np.append(weight_distribution, 0)

    for y in np.arange(y_f, b/2, 0.01):
        w = a*(-(c_r-c_t)/(b/2)*y+c_r)**2
        points = np.append(points, y)
        weight_distribution = np.append(weight_distribution, w)
    return(points, weight_distribution)


points_c1, weight_distribution_c1 = weight_distribution(W_FT_c1)

np.savetxt(
    out_dir_case1 / "weight.csv",
    np.column_stack((points_c1, weight_distribution)),
    delimiter=",",
    header="y_m,w_N_per_m",
    comments="",
)