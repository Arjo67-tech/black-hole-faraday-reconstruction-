import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray

DEVNULL = open(os.devnull, 'w')

A = 0.0
R_ORB = 8.0
SIG = 1.0
OMEGA = 1.0 / (R_ORB**1.5 + A)                 # orbital angular velocity
UT = (R_ORB**1.5 + A) / np.sqrt(R_ORB**3 - 3.0*R_ORB**2 + 2.0*A*R_ORB**1.5)
NPIX, EXTENT = 32, 15.0

def trace_grid(th_o):
    """Trace all pixels once; paths don't change with time."""
    alphas = np.linspace(-EXTENT, EXTENT, NPIX)
    betas  = np.linspace(-EXTENT, EXTENT, NPIX)
    rays = []
    for beta in betas:
        for alpha in alphas:
            with redirect_stdout(DEVNULL):
                ray = trace_ray(A, th_o, alpha, beta)
            r, th, ph = ray['r'], ray['th'], ray['ph']
            dl = np.sqrt(np.diff(r)**2 + (r[:-1]*np.diff(th))**2
                         + (r[:-1]*np.sin(th[:-1])*np.diff(ph))**2)
            s = np.concatenate(([0.0], np.cumsum(dl)))
            lam = -alpha * np.sin(th_o)          # photon angular momentum
            g = 1.0 / (UT * (1.0 - OMEGA*lam))   # redshift factor, exact for
            rays.append((s, r, th, ph, g))       # circular-orbit emitter
    return rays

def compute_polarization(rays):
    xb, yb = R_ORB*np.cos(0), R_ORB*np.sin(0)
    Itot, Qtot, Utot, Vtot = 0.0, 0.0, 0.0, 0.0
    for (s, r, th, ph, g) in rays:
        x = r*np.sin(th)*np.cos(ph)
        y = r*np.sin(th)*np.sin(ph)
        z = r*np.cos(th)
        emis = np.exp(-((x-xb)**2 + (y-yb)**2 + z**2) / (2*SIG**2))
        if emis.max() < 1e-8:
            continue                              # ray never meets the blob now
        def j_of_s(sv):
            i = np.argmin(np.abs(s - sv))
            return np.array([emis[i], 0.7*emis[i], 0.0, 0.0])
        def K_of_s(sv):
            return np.zeros((4, 4))
        S = integrate_ray(s, j_of_s, K_of_s, np.zeros(4))
        S *= g**3
        Itot += S[-1, 0]
        Qtot += S[-1, 1]
        Utot += S[-1, 2]
        Vtot += S[-1, 3]
    p = np.sqrt(Qtot**2 + Utot**2) / Itot
    chi = 0.5 * np.arctan2(Utot, Qtot)
    return Itot, Qtot, Utot, Vtot, p, chi

th_os = [np.radians(20), np.radians(75)]
results = {}

for th_o in th_os:
    print(f"Tracing at inclination {np.degrees(th_o)} degrees...")
    t0 = time.time()
    rays = trace_grid(th_o)
    Itot, Qtot, Utot, Vtot, p, chi = compute_polarization(rays)
    results[np.degrees(th_o)] = (Itot, Qtot, Utot, Vtot, p, chi)
    print(f"  done in {time.time()-t0:.0f} s")
    print(f"  Itot: {Itot:.4e}, Qtot: {Qtot:.4e}, Utot: {Utot:.4e}, Vtot: {Vtot:.4e}")
    print(f"  p: {p:.4f}, chi: {chi:.4f} rad")
    pass_p = np.isclose(p, 0.7000, atol=0.001)
    pass_chi = np.isclose(chi, 0.0, atol=0.001)
    pass_V = np.max(np.abs(Vtot)) < 1e-10
    print("check 1 (p = 0.7000):", "PASS" if pass_p else "FAIL")
    print("check 2 (chi = 0.0 rad):", "PASS" if pass_chi else "FAIL")
    print("check 3 (max|V| < 1e-10):", "PASS" if pass_V else "FAIL")
    print(f"OVERALL: {'PASS' if (pass_p and pass_chi and pass_V) else 'FAIL'}\n")

# Plotting
plt.figure(figsize=(9, 5))
for th_o, (Itot, Qtot, Utot, Vtot, p, chi) in results.items():
    plt.plot(th_o, Itot, 'o-', label=f'Itot {np.degrees(th_o)} deg')
    plt.plot(th_o, Qtot, 's-', label=f'Qtot {np.degrees(th_o)} deg')
    plt.plot(th_o, Utot, '^-', label=f'Utot {np.degrees(th_o)} deg')
    plt.plot(th_o, Vtot, 'v-', label=f'Vtot {np.degrees(th_o)} deg')

plt.xlabel('inclination (radians)'); plt.ylabel('flux')
plt.legend(); plt.grid(True)
plt.savefig('polarized_blob.png', dpi=110, bbox_inches='tight')
print("saved polarized_blob.png")
