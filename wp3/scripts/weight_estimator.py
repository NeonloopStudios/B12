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

def W_htail(allmoving_tail, F_w, B_h, W_dg, N_z, S_ht, L_t, sweep_ht, A_h, S_e):
    K_uht = 1.0
    if allmoving_tail: K_uht = 1.143
    return Kg(0.0379 * K_uht * pow(1 + Feet(F_w, 1)/Feet(B_h, 1), -0.25) * pow(Lbs(W_dg), 0.639) * pow(1.5 * N_z, 0.1) * pow(Feet(S_ht, 2), 0.75) * pow(Feet(L_t, 1), -1) * pow(0.3 * Feet(L_t, 1), 0.704) * pow(m.cos(m.radians(sweep_ht)), -1.0) * pow(A_h, 0.166) * pow(1 + S_e/S_ht, 0.1))
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
    W_eng = W_engine_and_contents(engine_with_prop, thrust_reverser, W_en)
    return 2*Kg(W_eng + 0.6724*K_ng*Feet(N_lt,1)**0.1*Feet(N_w,1)**0.294*1.5*N_z**0.119*W_eng**0.611*N_en**0.984*S_n**0.224)
    #N_lt Nacelle length
    #N_w Nacelle width
    #N_z limit load factor 
    #W_ec weight of engine and contents
    #W_en engine weight, each
    #S_n nacelle wetted area

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
    #L_f total fuselage length

def W_electronics(R_kva,L_a,N_gen):
    return Kg(7.291*(R_kva**0.782)*((Feet(L_a,1))**0.346)*N_gen**0.1)
    #R_kva system electrical rating
    #L_a electrical routing distance
    #N_gen number of generators (typically engine numb)

def W_furnishings(N_c,W_c,S_f):
    return Kg(0.0577*N_c**0.1*Lbs(W_c)**0.393*Feet(S_f,2)**0.75)
    #W_c maximum cargo weight
    #S_f fuselage wetted area
    #N_c number of crew

def W_antiice(W_dg):
    return Kg(0.002*Lbs(W_dg))

def OemEstimator(W_dg, N_z, S_w, A, tc_r, taper, sweep, S_csw, \
                allmoving_tail, F_w, B_h, S_ht, L_t, sweep_ht, A_h, S_e, \
                Ht_Hv, S_vt, sweep_vt, A_v, \
                K_door, K_lg, L, S_f, LD, B_w, \
                kneeling_ldg, W_l, N_l, L_m, N_mw, N_mss, V_stall, L_n, N_nw, \
                N_lt, N_w, N_en, S_n, W_en, engine_with_prop, thrust_reverser, pylon_mounted, \
                L_ec, V_t, V_i, V_p, N_t, \
                N_f, N_m, S_cs, I_y, reciprocating_engine, turboprop_engine, \
                N_c, L_f, R_kva, L_a, N_gen, W_c):
    mass = 0
    
    item_mass = W_wing(W_dg, N_z, S_w, A, tc_r, taper, sweep, S_csw)
    print(f"{round(item_mass, 2)} kg \t mass of wing")
    mass += item_mass

    item_mass = W_htail(allmoving_tail, F_w, B_h, W_dg, N_z, S_ht, L_t, sweep_ht, A_h, S_e)
    print(f"{round(item_mass, 2)} kg \t mass of horizontal tail")
    mass += item_mass

    item_mass = W_vtail(Ht_Hv, W_dg, N_z, L_t, S_vt, sweep_vt, A_v, tc_r)
    print(f"{round(item_mass, 2)} kg \t mass of vertical tail")
    mass += item_mass

    item_mass = W_fuselage(K_door, K_lg, W_dg, N_z, L, S_f, LD, taper, sweep, B_w)
    print(f"{round(item_mass, 2)} kg \t mass of fuselage")
    mass += item_mass

    item_mass = W_mainldg(kneeling_ldg, W_l, N_l, L_m, N_mw, N_mss, V_stall)
    print(f"{round(item_mass, 2)} kg \t mass of main landing gear")
    mass += item_mass

    item_mass = W_noseldg(kneeling_ldg, W_l, N_l, L_n, N_nw)
    print(f"{round(item_mass, 2)} kg \t mass of nose landing gear")
    mass += item_mass

    item_mass = W_nacelle(N_lt, N_w, N_z, N_en, S_n, W_en, engine_with_prop, thrust_reverser, pylon_mounted)
    print(f"{round(item_mass, 2)} kg \t mass of engines and nacelles")
    mass += item_mass

    item_mass = W_engctrl(N_en, L_ec)
    print(f"{round(item_mass, 2)} kg \t mass of engine control")
    mass += item_mass

    item_mass = W_starter(N_en, W_en)
    print(f"{round(item_mass, 2)} kg \t mass of starter")
    mass += item_mass

    item_mass = W_fuelsyst(V_t, V_i, V_p, N_t)
    print(f"{round(item_mass, 2)} kg \t mass of fuel system")
    mass += item_mass

    item_mass = W_flightctrl(N_f, N_m, S_cs, I_y)
    print(f"{round(item_mass, 2)} kg \t mass of flight control")
    mass += item_mass

    item_mass = W_instruments(reciprocating_engine, turboprop_engine, N_c, N_en, L_f, B_w)
    print(f"{round(item_mass, 2)} kg \t mass of instruments")
    mass += item_mass

    item_mass = W_hydraulics(N_f, L_f, B_w)
    print(f"{round(item_mass, 2)} kg \t mass of hydraulics")
    mass += item_mass

    item_mass = W_electronics(R_kva, L_a, N_gen)
    print(f"{round(item_mass, 2)} kg \t mass of electronics")
    mass += item_mass

    item_mass = W_furnishings(N_c,W_c,S_f)
    print(f"{round(item_mass, 2)} kg \t mass of furnishings")
    mass += item_mass

    item_mass = W_antiice(W_dg)
    print(f"{round(item_mass, 2)} kg \t mass of anti-ice system")
    mass += item_mass

    return round(mass, 2)

def Chord(chord_tip, chord_root, halfspan_percent):
    return (chord_tip - chord_root) * halfspan_percent + chord_root

def ControlSurfaceArea(span, root_frac, tip_frac, chord_frac):
    surface_area = (tip_frac - root_frac) * span * 0.5 * Chord(tip_chord, root_chord, root_frac + 0.5*(tip_frac - root_frac)) * chord_frac
    return 2 * surface_area

def body_wetted_area(
    diameter_m: float,
    nose_length_m: float,
    cylinder_length_m: float,
    tail_length_m: float,
) -> float:
    """Wetted area of a fuselage or nacelle (nose, cylinder, tail cone).

    Args:
        diameter_m: Maximum diameter D [m], > 0.
        nose_length_m: Nose length L1 [m], > 0.
        cylinder_length_m: Cylinder length L2 [m], >= 0.
        tail_length_m: Tail-cone length L3 [m], >= 0.

    Returns:
        Wetted area [m^2].

    Raises:
        ValueError: If D or L1 is not positive, or L2 or L3 is negative.
    """
    if diameter_m <= 0.0:
        raise ValueError(f"diameter_m must be > 0, got {diameter_m}")
    if nose_length_m <= 0.0:
        raise ValueError(f"nose_length_m must be > 0, got {nose_length_m}")
    if cylinder_length_m < 0.0:
        raise ValueError(f"cylinder_length_m must be >= 0, got {cylinder_length_m}")
    if tail_length_m < 0.0:
        raise ValueError(f"tail_length_m must be >= 0, got {tail_length_m}")
    d = diameter_m
    l1 = nose_length_m
    nose: float = (1.0 / (3.0 * l1**2)) * (
        (4.0 * l1**2 + d**2 / 4.0) ** 1.5 - d**3 / 8.0
    )
    cylinder = 4.0 * cylinder_length_m
    cone = 2.0 * m.sqrt(tail_length_m**2 + d**2 / 4.0)
    return m.pi * d / 4.0 * (nose - d + cylinder + cone)

#General design
OEM_design = 17930 #kg
design_payload_mass = 7200 #kg
maximum_payload_mass = 9302 #kg
fuel_mass = 7792 #kg
cruise_design_mass = OEM_design + design_payload_mass + fuel_mass
limit_load_factor = 3.727 #g
wing_area = 66.7 #m^2
aspect_ratio = 9.5
wingspan = m.sqrt(wing_area * aspect_ratio)
root_chord = 3.79
tip_chord = 1.51
tc_ratio_root = 0.12
taper_ratio = 0.4
sweep_angle = 30 #deg
glide_ratio = 17.621
V_approach = 65 #m/s
V_stall = V_approach / 1.23

#Control surface
control_surface_area = ControlSurfaceArea(wingspan, 0.65, 0.983, 0.25) + ControlSurfaceArea(wingspan, 0.12, 0.75, 0.3) + ControlSurfaceArea(wingspan, 0.12, 0.95, 0.2)
unitary_tail = False
tail_length = 18.07 #m, from wing quarter MAC to htail quarter MAC
interpolate_tail_type = 0.5 #0.0 for conventional tail and 1.0 for T tail
htail_span = 7.44 #m
htail_area = 13.84 #m^2
htail_sweep = 30 #deg
htail_aspect = 4
elevator_area = 0.2 * htail_area #m^2
vtail_area = 13.01 #m^2
vtail_sweep = 35 #deg
vtail_aspect = 1.8

#Fuselage
cargo_door_type = 1.06 #1.0 no cargo door, 1.06 one side cargo door, 1.12 two side cargo door or aft clamshell door, 1.25 if both two side and aft
landing_gear_mount = 1.12 #1.12 if fuselage-mounted main landing gear, 1.0 otherwise
total_fuselage_length = 28.68 #m
structural_fuselage_length = 0.9 * total_fuselage_length #m (excludes radome and tail cap)
fuselage_width_max = 3.01 #m^2
fuselage_width_at_htail = 0.65 * fuselage_width_max #m
nose_cone_length = 3.5 #m
tail_cone_length = 10.53 #m
fuselage_wetted_area = body_wetted_area(fuselage_width_max, nose_cone_length, total_fuselage_length - nose_cone_length - tail_cone_length, tail_cone_length) #m^2

#Landing gear
kneeling_landing_gear = False
landing_design_mass = OEM_design + design_payload_mass + fuel_mass
limit_landing_load_factor = 3.727
main_landing_gear_length = 2.5 #m
nr_main_ldg_wheels = 4
nr_main_ldg_shock_struts = 2
nose_landing_gear_length = 2.5 #m
nr_nose_ldg_wheels = 2

#Engine
nacelle_width = 1.34 #m
nacelle_length = 4.07 #m
nr_of_engines = 2
nacelle_wetted_area = nacelle_length * m.pi * nacelle_width #m^2
engine_weight = 1088 #kg, each
engine_with_propeller = False
reciprocating_engine = False
turboprop_engine = False
thrust_reverser = True
pylon_mounted_engine = True
length_from_engine_to_cockpit = 9.5 #m

#Fuel
fuel_mass_density = 800 #kg/m^3
total_fuel_volume = fuel_mass / fuel_mass_density #m^3
self_sealing_tank_volume = 0.0 * total_fuel_volume #m^3
integral_tank_volume = 1.0 * total_fuel_volume #m^3
nr_of_fuel_tanks = 2

#Control, Instruments, and Electronics
functions_performed_by_controls = 5 #4-7
nr_of_mechanical_functions = 1 #0-2
radius_of_gyration = 0.25 * (wingspan + total_fuselage_length) * 0.5
yawing_moment_of_inertia = cruise_design_mass * pow(radius_of_gyration, 2) #kg m^2
system_electrical_rating = 60 #kV*A
electric_routing_dist = 15
nr_of_generators = nr_of_engines

#Misc
nr_of_crew = 7

for i in range(10):
    OEM_design = OemEstimator(cruise_design_mass, limit_load_factor, wing_area, aspect_ratio, tc_ratio_root, taper_ratio, sweep_angle, \
                    control_surface_area, unitary_tail, fuselage_width_at_htail, htail_span, htail_area, tail_length, htail_sweep, htail_aspect, elevator_area, \
                    interpolate_tail_type, vtail_area, vtail_sweep, vtail_aspect, \
                    cargo_door_type, landing_gear_mount, structural_fuselage_length, fuselage_wetted_area, glide_ratio, wingspan, \
                    kneeling_landing_gear, landing_design_mass, limit_landing_load_factor, main_landing_gear_length, nr_main_ldg_wheels, nr_main_ldg_shock_struts, V_stall, nose_landing_gear_length, nr_nose_ldg_wheels, \
                    nacelle_length, nacelle_width, nr_of_engines, nacelle_wetted_area, engine_weight, engine_with_propeller, thrust_reverser, pylon_mounted_engine, \
                    length_from_engine_to_cockpit, total_fuel_volume, self_sealing_tank_volume, integral_tank_volume, nr_of_fuel_tanks, \
                    functions_performed_by_controls, nr_of_mechanical_functions, control_surface_area, yawing_moment_of_inertia, reciprocating_engine, turboprop_engine, \
                    nr_of_crew, total_fuselage_length, system_electrical_rating, electric_routing_dist, nr_of_generators, maximum_payload_mass)
    cruise_design_mass = OEM_design + design_payload_mass + fuel_mass
    landing_design_mass = OEM_design + design_payload_mass + fuel_mass
    print(f"The calculated operating empty mass is {OEM_design} kg, iteration nr {i}.")