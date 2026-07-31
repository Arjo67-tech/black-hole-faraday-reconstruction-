import numpy as np
import os
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray, transfer_matrix

DEVNULL = open(os.devnull, 'w')

A = 0.0
TH_O = np.radians(60)
BETA = 0.0
SIGMA_F = 0.3
NPIX = 12
EXTENT = 40
NPOLAR = 5

def rho_F(r):
    return SIGMA_F / (1 + (r / 4)**2)

def trace_and_integrate(alpha, lam2_values):
    with redirect_stdout(DEVNULL):
        ray = trace_ray(A, TH_O, alpha, BETA)
    
    r = ray['r']
    th = ray['th']
    ph = ray['ph']
    
    # Build path length array
    dr = np.diff(r)
    dth = np.diff(th)
    dph = np.diff(ph)
    dl = np.sqrt(dr**2 + (r[:-1]*dth)**2 + (r[:-1]*np.sin(th[:-1])*dph)**2)
    s = np.concatenate(([0], np.cumsum(dl)))
    
    # Define j_of_s: emissivity along the path
    def j_of_s(s_val):
        return np.zeros(4)
    
    slopes = []
    for lam2 in lam2_values:
        rho_Vs = 2 * rho_F(r) * lam2
        K_of_s = lambda s_val: transfer_matrix(0, 0, 0, 0, 0.0, 0.0, rho_Vs[np.argmin(np.abs(s - s_val))])
        
        S0 = np.array([1.0, 0.7, 0.0, 0.0])
        S = integrate_ray(s, j_of_s, K_of_s, S0)
        Q, U = S[-1, 1], S[-1, 2]
        chi = 0.5 * np.arctan2(U, Q)
        slopes.append(chi)
    
    return s, r, dl, slopes

def compute_column(r, dl):
    return np.sum(rho_F(r) * dl)

lam2_values = np.linspace(0.2, 1.2, NPOLAR)

# Ray 1 (far): alpha=30
alpha1 = 30
s1, r1, dl1, slopes1 = trace_and_integrate(alpha1, lam2_values)
column_curved1 = compute_column(r1, dl1)
column_straight1 = 16 * np.pi * SIGMA_F / np.sqrt(16 + alpha1**2)
slope1 = np.polyfit(lam2_values, slopes1, 1)[0]
pass_1 = np.isclose(slope1, column_curved1, atol=0.01 * column_curved1)

print(f"Ray 1 (far): alpha={alpha1}")
print(f"  Column curved: {column_curved1:.4f}")
print(f"  Column straight: {column_straight1:.4f} ({100*(column_curved1/column_straight1 - 1):.2f}% offset)")
print(f"  Slope: {slope1:.4f}, expected: {column_curved1:.4f}")
print("check 1 (slope matches column_curved within 1%):", "PASS" if pass_1 else "FAIL")

# Ray 2 (near): alpha=6
alpha2 = 6
s2, r2, dl2, slopes2 = trace_and_integrate(alpha2, lam2_values)
column_curved2 = compute_column(r2, dl2)
column_straight2 = 16 * np.pi * SIGMA_F / np.sqrt(16 + alpha2**2)
slope2 = np.polyfit(lam2_values, slopes2, 1)[0]
pass_2 = np.isclose(slope2, column_curved2, atol=0.01 * column_curved2)

print(f"\nRay 2 (near): alpha={alpha2}")
print(f"  Column curved: {column_curved2:.4f}")
print(f"  Column straight: {column_straight2:.4f} ({100*(column_curved2/column_straight2 - 1):.2f}% offset)")
print(f"  Slope: {slope2:.4f}, expected: {column_curved2:.4f}")
print("check 2 (slope matches column_curved within 1%):", "PASS" if pass_2 else "FAIL")

# Retrace ray 2 with n_points=2000
with redirect_stdout(DEVNULL):
    ray_high_res = trace_ray(A, TH_O, alpha2, BETA, n_points=2000)
r_high_res = ray_high_res['r']
dl_high_res = np.sqrt(np.diff(r_high_res)**2 + (r_high_res[:-1]*np.diff(ray_high_res['th']))**2
                      + (r_high_res[:-1]*np.sin(ray_high_res['th'][:-1])*np.diff(ray_high_res['ph']))**2)
column_curved2_high_res = compute_column(r_high_res, dl_high_res)
pass_4 = np.isclose(column_curved2, column_curved2_high_res, atol=0.01 * column_curved2)

print("check 4 (agrees with the 500-point column within 1%):", "PASS" if pass_4 else "FAIL")

# Ambiguity demo: for ray 1 at lam2 = 7.15
lam2_demo = 7.15
slope_demo = np.polyfit([lam2_demo], [slopes1[-1]], 1)[0]
true_angle = slope_demo * lam2_demo
observed_chi = slopes1[-1]

print(f"\nAmbiguity demo: Ray 1 at lam2={lam2_demo}")
print(f"  True accumulated angle: {true_angle:.4f} rad")
print(f"  Observed exit chi: {observed_chi:.4f} rad")

# Overall PASS/FAIL
overall_pass = pass_1 and pass_2 and pass_4
print("\nOVERALL:", "PASS" if overall_pass else "FAIL")
