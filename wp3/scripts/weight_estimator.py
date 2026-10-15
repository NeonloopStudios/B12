"""
This is a Class II weight estimator for a wing and the associated aircraft, based on Raymer.
"""

import numpy as np
import math as m 
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

def gal_to_met(l):
    return l*0.00378541

#def W_wing():

#def W_htail():

#def W_vtail():

#def W_fuselage():

#def W_mainldg():

#def W_noseldg():

def W_nacelle(N_lt, N_w, N_z, N_en, S_n, W_en, K_p, K_tr, nacelle_mount):
    K_ng = 0
    if nacelle_mount == "pylon-mounted":
        K_ng = 1.017
    else:
        K_ng = 1
    return Kg(0.6724*K_ng*Feet(N_lt,1)**0.1*Feet(N_w,1)**0.294*1.5*N_z**0.119*W_engine_and_contents(W_en,K_p,K_tr)**0.611*N_en**0.984*S_n**0.224)
#K_ng = 1.017 for pylon-mounted nacelle, 1.0 otherwise
#N_lt Nacelle length
#N_w Nacelle width
#N_z limit load factor 
#W_ec = Weight of engine and contents 

def W_engine_and_contents(engine_with_prop, thrust_reverser, W_en):
    K_p = 0
    K_tr = 0
    if engine_with_prop:
        K_p = 1.4
    else:
        K_p = 1
    if thrust_reverser:
        K_tr = 1.18
    else:
        K_tr = 1
    return 2.331*Lbs(W_en)**0.901*K_p*K_tr
#K_p 1.4 for engine with propeller 1.0 otherwise
#K_tr 1.18 for thrust reverser or 1.0 otherwise

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

def W_instruments(reciprocating_engine,engine_type,N_c,N_en,L_f,B_w):
    K_tp = 0
    K_r = 0
    if engine_type == "turboprop":
        K_tp = 0.792
    else:
        K_tp = 1
    if reciprocating_engine:
        K_r = 1.133
    else:
        K_r = 1
    return Kg(4.509*K_r*K_tp*N_c**0.541*N_en*(Feet(L_f,1)+(Feet(B_w,1)))**0.5)
#K_r = 1.133 if reciprocating engine, 1.0 otherwise 
#K_tp = 0.793 if turboprop, 1.0 otherwise
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