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

def Gal(l):
    return l/0.00378541

def CubicMeter(l):
    return l*0.00378541

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

def K_ws(taper, B_w, sweep, L):
    return 0.75 * ((1 + 2*taper) / (1 + taper)) * Feet(B_w, 1) * m.tan(m.radians(sweep)) / Feet(L, 1)

def W_fuselage(K_door, K_lg, W_dg, N_z, L, S_f, LD, taper, sweep, B_w):
    return Kg(0.328 * K_door * K_lg * pow(Lbs(W_dg) * 1.5 * N_z, 0.5) * pow(Feet(L, 1), 0.25) * pow(Feet(S_f, 2), 0.302) * pow(1 + K_ws(taper, B_w, sweep, L), 0.04) * pow(LD, 0.1))
    #K_door 1.0 no cargo door, 1.06 one side cargo door, 1.12 two side cargo door or aft clamshell door, 1.25 if both two side and aft
    #K_lg 1.12 if fuselage-mounted main landing gear, 1.0 otherwise
    #L fuselage structural length (excludes radome and tail cap)    
    #S_f fuselage wetted area
    #B_w wingspan

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

def W_engine_and_contents(engine_with_prop, thrust_reverser, W_en):
    K_p = 1.0
    K_tr = 1.0
    if engine_with_prop: K_p = 1.4
    if thrust_reverser: K_tr = 1.18
    return 2.331*Lbs(W_en)**0.901*K_p*K_tr
    #K_p 1.4 for engine with propeller 1.0 otherwise
    #K_tr 1.18 for thrust reverser or 1.0 otherwise

def W_nacelle(N_lt, N_w, N_z, N_en, S_n, W_en, engine_with_prop, thrust_reverser, pylon_mounted):
    K_ng = 1.0
    if pylon_mounted: K_ng = 1.017
    return Kg(0.6724*K_ng*Feet(N_lt,1)**0.1*Feet(N_w,1)**0.294*1.5*N_z**0.119*W_engine_and_contents(engine_with_prop,thrust_reverser,W_en)**0.611*N_en**0.984*S_n**0.224)
    #N_lt Nacelle length
    #N_w Nacelle width
    #N_z limit load factor 
    #W_ec = Weight of engine and contents 

def W_engctrl(N_en, L_ec):
    return Kg(5*N_en+0.80*Feet(L_ec,1))
    #L_ec Length from engine front to cockpit
    #N_en Number of engines

def W_starter(N_en, W_en):
    return Kg(49.19*(N_en*Lbs(W_en)/1000)**0.541)
    #N_en number of engines
    #W_en engine weight

def W_fuelsyst(V_t, V_i, V_p, N_t):
    return Kg(2.405*Gal(V_t)**0.606*(1+V_i/V_t)**-1*(1+V_p/V_t)*N_t**0.5)
    #V_t total fuel volume, gal
    #V_p self sealing "protected" tanks volume ,gal
    #V_i integral tanks volume, gal          
    #N_t number of fuel tanks     
  
def W_flightctrl(N_f, N_m, S_cs, I_y):
    return Kg(145.9*N_f**0.554*(1+N_m/N_f)**-1*Feet(S_cs,2)**0.20*(Feet(Lbs(I_y),2)*10**-6)**0.07)
    #N_f number of functions performed by controls (4-7)
    #N_m number of mechanical functions (0-2)
    #S_cs total area of control surfaces
    #I_y yawing moment of inertia

def W_instruments(reciprocating_engine,turboprop_engine,N_c,N_en,L_f,B_w):
    K_tp = 1.0
    K_r = 1.0
    if turboprop_engine: K_tp = 0.792
    if reciprocating_engine: K_r = 1.133
    return Kg(4.509*K_r*K_tp*N_c**0.541*N_en*(Feet(L_f,1)+(Feet(B_w,1)))**0.5)
    #N_c number of crew
    #L_f total fuseage length
    #B_w wing span

def W_hydraulics(N_f,L_f,B_w):
    return Kg(0.2673*N_f*(Feet(L_f,1)+Feet(B_w,1))**0.937)
    #N_f number of functions performed by controls (4-7)
    #L_f total fuselage length, 

def W_electronics(R_kva,L_a,N_gen):
    return Kg(7.291*(R_kva**0.782)*((Feet(L_a,1))**0.346)*N_gen**0.1)
    #R_kva system electrical rating
    #L_a electrical routing distance
    #N_gen number of generators (typically engine numb)

def W_avionics(W_uav):
    return Kg(1.73*Lbs(W_uav)**0.983)
    #W_uav uninstalled avionics weight

def W_furnishings(N_c,W_c,S_f):
    return Kg(0.0577*N_c**0.1*Lbs(W_c)**0.393*Feet(S_f,2)**0.75)
    #W_c maximum cargo weight
    #S_f fuselage wetted area
    #N_c number of crew

def W_ac(N_p,V_pr, W_uav):
    return Kg(62.36*(N_p**0.25)*(Feet(V_pr,3)/1000)**0.604*Lbs(W_uav)**0.1)
    #N_p number of people on board
    #V_pr volume of pressureized section
    #W_uav uninstalled avionics weight

def W_antiice(W_dg):
    return Kg(0.002*Lbs(W_dg))
    #print(f" {Lbs(10)} \n {Kg(Lbs(10))} \n {Feet(100, 1)} \n {Feet(100, 2)} \n {Meter(Feet(100, 3), 3)}")

def OemEstimator(W_dg, N_z, S_w, A, tc_r, taper, sweep, S_csw):
    return W_wing(W_dg, N_z, S_w, A, tc_r, taper, sweep, S_csw)

print(f" {OemEstimator(1, 1, 1, 1, 1, 1, 1, 1)} ")