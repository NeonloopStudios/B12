"""
This is a Class II weight estimator for a wing and the associated aircraft, based on Raymer.
"""

import numpy as np

def Lbs(mass):
    return mass * 2.20462

def Kg(mass):
    return mass / 2.20462

def Feet(l, exp): #exp is 1 for lengths, 2 for areas, 3 for volumes
    return l * pow(3.28084, exp)

def Meter(l, exp): #exp is 1 for lengths, 2 for areas, 3 for volumes
    return l / pow(3.28084, exp)

#def W_wing():

#def W_htail():

#def W_vtail():

#def W_fuselage():

#def W_mainldg():

#def W_noseldg():

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