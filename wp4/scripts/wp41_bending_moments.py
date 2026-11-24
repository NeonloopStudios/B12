import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) 

import math
import matplotlib.pyplot as plt
from configs.aircraft import WING, MASSES

span = math.sqrt(WING.area_m2 * WING.aspect_ratio)
mroot = 0
Wengine = MASSES.engine_dry_kg * 9.81
deltay = span/2/120
y = 0
ylist = [0]
m = 0
mlist = [0]
a = 4.4

while y < span/2:
    if y > a:
        m = mroot + Wengine * (y-a)
    else:
        m = mroot
    ylist.append(y)
    mlist.append(m)
    y += deltay

plt.figure()
plt.plot(ylist,mlist)
plt.show() 