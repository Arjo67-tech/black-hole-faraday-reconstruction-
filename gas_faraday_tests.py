import numpy as np
import os
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray
from coeffs import transfer_matrix

DEVNULL = open(os.devnull, 'w')
A = 0.0
TH_O = np.radians(60.0)

def rho_F(r):
    return 0.3 / (1.0 + (r/4.0)**2)

def get_path(alpha, n_points=500):
    with redirect_stdout(DEVNULL):
        ray = trace_ray(A, TH_O, alpha, 0.0, n_points=n_points)
    r, th, ph = ray['r'], ray['th'], ray['ph']
    dl = np.sqrt(np.diff(r)**2 + (r[:-1]*np.diff(th))**2
                 + (r[:-1]*np.sin(th[:-1])*np.diff(ph))**2)
    s = np.concatenate(([0.0], np.cumsum(dl)))
    column = np.sum(rho_F(0.5*(r[:-1] + r[1:])) * dl)   # midpoint sampling
    return s, r, column

def exit_chi(s, r, lam2):
    def j_of_s(sv):
        return np.zeros(4)
    def K_of_s(sv):
        i = np.argmin(np.abs(s - sv))
        return transfer_matrix(0,0,0,0, 0.0, 0.0, 2.0*rho_F(r[i])*lam2)
    S = integrate_ray(s, j_of_s, K_of_s, np.array([1.0, 0.7, 0.0, 0.0]))
    return 0.5*np.arctan2(S[-1,2], S[-1,1])

def fit_slope(s, r):
    lam2s = np.linspace(0.2, 1.2, 5)
    chis = np.array([exit_chi(s, r, l2) for l2 in lam2s])
    chis_u = np.unwrap(chis, period=np.pi)   # polarization angle wraps every pi
    return np.polyfit(lam2s, chis_u, 1)[0], chis, chis_u

print("Ray 1 (far, alpha=30)")
s1, r1, col1 = get_path(30.0)
slope1, chis1, chis1u = fit_slope(s1, r1)
straight1 = 16*np.pi*0.3/np.sqrt(16+900)
print(f"  column curved   {col1:.4f}")
print(f"  column straight {straight1:.4f}  ({100*(col1-straight1)/straight1:+.2f}% GR excess)")
print(f"  fitted slope    {slope1:.4f}")
p1 = abs(slope1-col1)/col1 < 0.01
print("  check 1 (slope = own column, 1%):", "PASS" if p1 else "FAIL")

print("Ray 2 (near, alpha=6)")
s2, r2, col2 = get_path(6.0)
slope2, chis2, chis2u = fit_slope(s2, r2)
straight2 = 16*np.pi*0.3/np.sqrt(16+36)
ratio = col2/straight2
print(f"  column curved   {col2:.4f}")
print(f"  column straight {straight2:.4f}   ratio {ratio:.3f}")
print(f"  raw exit angles       {np.round(chis2,3)}   <- the wraps that broke the last run")
print(f"  unwrapped (period pi) {np.round(chis2u,3)}")
print(f"  fitted slope    {slope2:.4f}")
p2 = abs(slope2-col2)/col2 < 0.01
p3 = ratio > 1.05
print("  check 2 (slope = own column, 1%):", "PASS" if p2 else "FAIL")
print("  check 3 (GR ratio > 1.05):       ", "PASS" if p3 else "FAIL")

_, _, col2b = get_path(6.0, n_points=2000)
p4 = abs(col2b-col2)/col2 < 0.01
print(f"  column at 2000 pts {col2b:.4f}")
print("  check 4 (500 vs 2000 pts, 1%):   ", "PASS" if p4 else "FAIL")

print("Ambiguity demo, ray 1 at lam2=7.15 (86 GHz):")
chi_obs = exit_chi(s1, r1, 7.15)
chi_true = col1 * 7.15
print(f"  true accumulated rotation {chi_true:.4f} rad")
print(f"  observed exit chi         {chi_obs:.4f} rad")
print(f"  difference/pi = {(chi_true-chi_obs)/np.pi:.4f}  -> lost half-turns; one frequency alone is ambiguous")

print("OVERALL:", "PASS" if (p1 and p2 and p3 and p4) else "FAIL")
