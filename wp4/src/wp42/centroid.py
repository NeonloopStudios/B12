# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-19
"""WP4.2: centroid calculations from wingbox edge coordinates."""

import math as m
import numpy as np 

#xf and xr are the x-pos of the spars
def rectangle(xf, xr, zup, zdn):
    centroid = (xf + (xr-xf)/2, zdn + (zup-zdn)/2)
    area = abs((xr - xf) * (zup - zdn))
    return (centroid, area) #in m and m^2 for a 1m chord

#front_tip is a pseudo-bool; 1 if true
def triangle(xf, xr, ztip, zbase, front_tip: bool):
    x_loc = ((1+front_tip)*xf + (2-front_tip)*xr) / 3
    centroid = (x_loc, (2*zbase + ztip)/3)
    area = abs((xr - xf) * (zbase - ztip)) / 2
    return (centroid, area)

#tmax_ul is the max unitless (dat-format) thickness
#tmax_c is the max thickness as a frac of the chord
#function returns in this order: centroid x, centroid z, area; last two as fractions of chord
def centroid_wingbox(fspar_loc: float, fspar_up: float, fspar_dn: float, rspar_loc: float, rspar_up: float, rspar_dn: float, tmax_ul: float, tmax_c: float):
    fspar_up = -fspar_up
    fspar_dn = -fspar_dn
    rspar_up = -rspar_up
    rspar_dn = -rspar_dn
    
    ctr_up = min(fspar_up, rspar_up)
    ctr_dn = max(fspar_dn, rspar_dn)

    P1 = rectangle(fspar_loc, rspar_loc, ctr_up, ctr_dn)

    upper_triangle_fspar = True #1 indicates the tip of the upper triangle is above the front spar
    if fspar_up < rspar_up: upper_triangle_fspar = False
    lower_triangle_fspar = True
    if fspar_dn > rspar_dn: lower_triangle_fspar = False

    P2 = triangle(fspar_loc, rspar_loc, max(fspar_up, rspar_up), ctr_up, upper_triangle_fspar)
    P3 = triangle(fspar_loc, rspar_loc, min(fspar_dn, rspar_dn), ctr_dn, lower_triangle_fspar)
    
    area = P1[1] + P2[1] + P3[1]
    centroid_x = (P1[0][0] * P1[1] + P2[0][0] * P2[1] + P3[0][0] * P3[1]) / area
    centroid_z = (P1[0][1] * P1[1] + P2[0][1] * P2[1] + P3[0][1] * P3[1]) / area

    return ((centroid_x, -centroid_z * tmax_c / tmax_ul), area)
    #return(f"{P1} \n {P2} \n {P3} \n {(centroid_x, centroid_z)}") #for testing