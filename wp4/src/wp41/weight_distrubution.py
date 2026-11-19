import numpy as np

weight_distribution = np.array([])
points = np.array([])

#Wing planform
c_r = 0.5  # root chord length
c_t = 0.2  # tip chord length
b = 10.0  # wing span

W_FT = 1000.0 #Total fuel weight carried in the wings (both halfwings)

a = 3*W_FT/(b*(c_r**2 + c_r*c_t + c_t**2))


for y in np.arange(0, b/2, 0.01):
    w = a*(-(c_r-c_t)/(b/2)*y+c_r)**2
    points = np.append(points, y)
    weight_distribution = np.append(weight_distribution, w)

np.savetxt("weight_distribution.txt", np.column_stack((points, weight_distribution)))