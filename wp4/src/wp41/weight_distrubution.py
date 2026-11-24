import math

import numpy as np
import numpy.typing as npt

from configs.aircraft import WING

#Wing planform from configs
b = math.sqrt(WING.aspect_ratio * WING.area_m2)  # wing span   
c_r = 2.0 * WING.area_m2 / (b * (1.0 + WING.taper))  # root chord length
c_t = WING.taper * c_r  # tip chord length
y_f = WING.fuselage_diameter_m / 2.0  # spanwise location of the fuselage edge

c_f = c_r - (WING.fuselage_diameter_m / 2.0) * (c_r - c_t) / (b / 2.0)  # root chord length outside fuselage

def weight_distribution(
    W_fuel: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    weight_distribution = np.array([])
    points = np.array([])
    a = 3 * W_fuel / (b - 2*y_f) / (c_f**2 + c_f*c_t + c_t**2)

    for y in np.arange(0, y_f, 0.01):
        points = np.append(points, y)
        weight_distribution = np.append(weight_distribution, 0)

    for y in np.arange(y_f, b/2, 0.01):
        w = a*(-(c_r-c_t)/(b/2)*y+c_r)**2
        points = np.append(points, y)
        weight_distribution = np.append(weight_distribution, w)
    return(points, weight_distribution)
