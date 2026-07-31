import numpy as np
import os
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray

DEVNULL = open(os.devnull, 'w')
TH_O = np.radians(90.001)

def measure_deflection(b):
    with redirect_stdout(DEVNULL):
        ray = trace_ray(0.0, TH_O, b, 0.0)
    ph = np.unwrap(ray['ph'])
    dphi = abs(ph[-1] - ph[0])
    # endpoints sit at finite radius, not infinity: each end is missing
    # an azimuth slice of arcsin(b / r_endpoint). add both back.
    corr = np.arcsin(b / ray['r'][0]) + np.arcsin(b / ray['r'][-1])
    return dphi - np.pi + corr

print(f"{'b':>6} {'measured':>12} {'theory':>12} {'4/b alone':>12} {'diff %':>8}")
ok = True
for b in [20.0, 30.0, 50.0, 100.0]:
    meas = measure_deflection(b)
    theory = 4.0/b + (15*np.pi/4)/b**2 + (128.0/3.0)/b**3   # series to 3rd order
    pct = 100.0 * (meas - theory) / theory
    ok = ok and abs(pct) < 1.0
    print(f"{b:>6.1f} {meas:>12.6f} {theory:>12.6f} {4.0/b:>12.6f} {pct:>7.2f}%")
print("PASS" if ok else "FAIL", "(tolerance 1% vs 3rd-order series)")
