"""
This is a Class II weight estimator for a CS25 transport aircraft, based on Raymer.
"""

import numpy as np
import math as m

def Knot(speed): #from m/s
    return speed * 1.94384

def Unknot(speed): #to m/s
    return speed / 1.94384

def Lbs(mass):
    return mass * 2.20462

def Kg(mass):
    return mass / 2.20462

def Feet(l, exp): #exp is 1 for lengths, 2 for areas, 3 for volumes
    return l * pow(3.28084, exp)

def Meter(l, exp): #exp is 1 for lengths, 2 for areas, 3 for volumes
    return l / pow(3.28084, exp)

def W_wing(W_dg, N_z, S_w, A, tc_r, taper, sweep, S_csw):
    return Kg(0.0051 * pow(Lbs(W_dg) * 1.5 * N_z, 0.557) * pow(Feet(S_w, 2), 0.649) * pow(A, 0.5) * pow(tc_r, -0.4) * pow(1 + taper, 0.1) * pow(m.cos(m.radians(sweep)), -1.0) * pow(Feet(S_csw, 2), 0.1))
    #N_z limit load factor
    #S_w trapezoidal wing area
    #S_csw control surface area
    #tc_r t/c ratio root

def W_htail(K_uht, F_w, B_h, W_dg, N_z, S_ht, L_t, sweep_ht, A_h, S_e):
    return Kg(0.0379 * K_uht * pow(1 + Feet(F_w, 1)/Feet(B_h, 1), -0.25) * pow(Lbs(W_dg), 0.639) * pow(1.5 * N_z, 0.1) * pow(Feet(S_ht, 2), 0.75) * pow(Feet(L_t, 1), -1) * pow(0.3 * Feet(L_t, 1), 0.704) * pow(m.cos(m.radians(sweep_ht)), -1.0) * pow(A_h, 0.166) * pow(1 + S_e/S_ht, 0.1))
    #K_uht 1.143 for all-moving htail or 1.0 otherwise
    #F_w fuselage width at htail intersection
    #B_h htail span
    #S_ht horizontal tail area
    #S_e elevator area
    #L_t tail length, wing quarter MAC to tail quater MAC

def W_vtail(Ht_Hv, W_dg, N_z, L_t, S_vt, sweep_vt, A_v, tc_r):
    return Kg(0.0026 * pow(1 + Ht_Hv, 0.225) * pow(Lbs(W_dg), 0.556) * pow(1.5 * N_z, 0.536) * pow(Feet(L_t, 1), -0.5) * pow(Feet(S_vt, 2), 0.5) * pow(L_t, 0.875) * pow(m.cos(m.radians(sweep_vt)), -1) * pow(A_v, 0.35) * pow(tc_r, -0.5))
    #Ht_Hv interpolate between 0.0 for conventional tail and 1.0 for T tail
    #S_vt vertical tail area

def W_fuselage(K_door, K_lg, W_dg, N_z, L, S_f, K_ws, LD, taper, sweep, B_w):
    return Kg(0.328 * K_door * K_lg * pow(Lbs(W_dg) * 1.5 * N_z, 0.5) * pow(Feet(L, 1), 0.25) * pow(Feet(S_f, 2), 0.302) * pow(1 + K_ws(taper, B_w, sweep, L), 0.04) * pow(LD, 0.1))
    #K_door 1.0 no cargo door, 1.06 one side cargo door, 1.12 two side cargo door or aft clamshell door, 1.25 if both two side and aft
    #K_lg 1.12 if fuselage-mounted main landing gear, 1.0 otherwise
    #L fuselage structural length (excludes radome and tail cap)    
    #S_f fuselage wetted area
    #B_w wingspan

def K_ws(taper, B_w, sweep, L):
    return 0.75 * ((1 + 2*taper) / (1 + taper)) * Feet(B_w, 1) * m.tan(m.radians(sweep)) / Feet(L, 1)

def W_mainldg(kneeling_ldg, W_l, N_l, L_m, N_mw, N_mss, V_stall):
    K_mp = 1.0
    if kneeling_ldg: K_mp = 1.126
    return Kg(0.0106 * K_mp * pow(Lbs(W_l), 0.888) * pow(1.5 * N_l, 0.25) * pow(Feet(L_m, 1) * 12, 0.4) * pow(N_mw, 0.321) * pow(N_mss, -0.5) * pow(Knot(V_stall), 0.1))
    #W_l landing design gross weight
    #N_l limit landing load factor
    #L_m length of main landing gear
    #N_mw nr of main wheels
    #N_mss nr of main gear shock struts

def W_noseldg(kneeling_ldg, W_l, N_l, L_n, N_nw):
    K_np = 1.0
    if kneeling_ldg: K_np = 1.15
    return Kg(0.032 * K_np * pow(Lbs(W_l), 0.646) * pow(1.5 * N_l, 0.2) * pow(Feet(L_n, 1) * 12, 0.5) * pow(N_nw, 0.45))
    #L_n nose gear length
    #N_nw nr of nose wheels

#def W_nacelle():

#def W_engctrl():

#def W_starter():

#def W_fuelsyst():

#def W_flightctrl():

#def W_apu():

#def W_instruments();

#def W_hydraulics():

#def W_electronics():

#def W_avionics():

#def W_furnishings():

#def W_ac():

#def W_anti-ice():

#print(f" {Lbs(10)} \n {Kg(Lbs(10))} \n {Feet(100, 1)} \n {Feet(100, 2)} \n {Meter(Feet(100, 3), 3)}")