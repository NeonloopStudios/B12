import scipy

root_chord = 3.66 
tip_chord = 1.47
span = 24.37
taper_ratio = tip_chord/root_chord

def Am_per_chord (Am, y_pos):
    chord_y = (3.66-taper_ratio*y_pos/span/2)
    Am_y = Am*chord_y
    return Am_y

