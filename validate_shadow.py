import numpy as np
import os
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray

DEVNULL = open(os.devnull, 'w')

def is_captured(a, th_o, alpha, beta=0.0):
    with redirect_stdout(DEVNULL):
        return trace_ray(a, th_o, alpha, beta)['captured']

def find_boundary(a, th_o, alpha_lo, alpha_hi, tol=1e-3):
    # invariant: the two endpoints have different fates; move whichever
    # endpoint matches the midpoint's fate
    cap_lo = is_captured(a, th_o, alpha_lo)
    cap_hi = is_captured(a, th_o, alpha_hi)
    assert cap_lo != cap_hi, "endpoints must straddle the boundary"
    while abs(alpha_hi - alpha_lo) > tol:
        mid = 0.5 * (alpha_lo + alpha_hi)
        if is_captured(a, th_o, mid) == cap_lo:
            alpha_lo = mid
        else:
            alpha_hi = mid
    return 0.5 * (alpha_lo + alpha_hi)

b_theory = np.sqrt(27.0)

b17_pos = find_boundary(0.0, np.radians(17.0),  1.0,  10.0)
b17_neg = find_boundary(0.0, np.radians(17.0), -1.0, -10.0)
b60_pos = find_boundary(0.0, np.radians(60.0),  1.0,  10.0)
b60_neg = find_boundary(0.0, np.radians(60.0), -1.0, -10.0)
b9_pos  = find_boundary(0.9, np.radians(17.0),  1.0,  10.0)
b9_neg  = find_boundary(0.9, np.radians(17.0), -1.0, -10.0)

print(f"a=0.0, th_o=17:  +{b17_pos:.4f}   {b17_neg:.4f}   (theory ±{b_theory:.4f})")
print(f"a=0.0, th_o=60:  +{b60_pos:.4f}   {b60_neg:.4f}   (inclination must not matter)")
print(f"a=0.9, th_o=17:  +{b9_pos:.4f}   {b9_neg:.4f}   (spin: asymmetric)")

p1 = abs(b17_pos - b_theory) < 0.002 and abs(b17_neg + b_theory) < 0.002
p2 = abs(b60_pos - b_theory) < 0.002 and abs(b60_neg + b_theory) < 0.002
p3 = (b9_pos < b_theory) != (abs(b9_neg) < b_theory)   # exactly one side pulled in
print("check 1 (sqrt27 at 17 deg):", "PASS" if p1 else "FAIL")
print("check 2 (same at 60 deg):  ", "PASS" if p2 else "FAIL")
print("check 3 (spin asymmetry):  ", "PASS" if p3 else "FAIL")
print("OVERALL:", "PASS" if (p1 and p2 and p3) else "FAIL")
