import numpy as np
import os, time
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
T_ORB = 2.0 * np.pi * (R_ORB**1.5 + A)         # orbital period, units of M
UT = (R_ORB**1.5 + A) / np.sqrt(R_ORB**3 - 3.0*R_ORB**2 + 2.0*A*R_ORB**1.5)
NPIX, EXTENT, NPHASE = 32, 15.0, 24

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

def flux_at_phase(rays, ph_blob):
    xb, yb = R_ORB*np.cos(ph_blob), R_ORB*np.sin(ph_blob)
    tot = 0.0
    for (s, r, th, ph, g) in rays:
        x = r*np.sin(th)*np.cos(ph)
        y = r*np.sin(th)*np.sin(ph)
        z = r*np.cos(th)
        emis = np.exp(-((x-xb)**2 + (y-yb)**2 + z**2) / (2*SIG**2))
        if emis.max() < 1e-8:
            continue                              # ray never meets the blob now
        def j_of_s(sv):
            i = np.argmin(np.abs(s - sv))
            return np.array([emis[i], 0.0, 0.0, 0.0])
        def K_of_s(sv):
            return np.zeros((4, 4))
        S = integrate_ray(s, j_of_s, K_of_s, np.zeros(4))
        tot += (g**3) * S[-1, 0]                  # observed brightness scales as g^3
    return tot

phases = np.linspace(0.0, 2*np.pi, NPHASE, endpoint=False)
curves, granges = {}, {}
for name, deg in [('face-on 5 deg', 5.0), ('edge-on 80 deg', 80.0)]:
    th_o = np.radians(deg)
    print(f"{name}: tracing {NPIX*NPIX} rays once...")
    t0 = time.time()
    rays = trace_grid(th_o)
    gs = np.array([rr[4] for rr in rays])
    granges[name] = (gs.min(), gs.max())
    print(f"  done in {time.time()-t0:.0f} s; g range {gs.min():.3f} .. {gs.max():.3f}")
    fl = []
    for k, pb in enumerate(phases):
        fl.append(flux_at_phase(rays, pb))
        print(f"  phase {k+1}/{NPHASE}   ", end='\r')
    curves[name] = np.array(fl)
    print()

face, edge = curves['face-on 5 deg'], curves['edge-on 80 deg']
r_face = face.max() / face.min()
r_edge = edge.max() / edge.min()
gmin80, gmax80 = granges['edge-on 80 deg']

print(f"\nface-on  peak/trough = {r_face:.2f}   (expect ~1.2-1.3, weak modulation)")
print(f"edge-on  peak/trough = {r_edge:.2f}   (expect >3, Doppler beaming)")
print(f"edge-on g range: {gmin80:.3f} .. {gmax80:.3f}   (must straddle 1)")
p1 = r_face < 1.5
p2 = r_edge > 3.0
p3 = (gmin80 < 1.0 < gmax80)
print("check 1 (face-on nearly flat):", "PASS" if p1 else "FAIL")
print("check 2 (edge-on strong peak):", "PASS" if p2 else "FAIL")
print("check 3 (blue+redshift both): ", "PASS" if p3 else "FAIL")
print("OVERALL:", "PASS" if (p1 and p2 and p3) else "FAIL")

M_SEC = 20.4   # GM/c^3 for Sgr A*, seconds
print(f"\nOrbital period T = {T_ORB:.1f} M  =  {T_ORB*M_SEC/60:.0f} minutes for Sgr A*")
print("(GRAVITY watched real Sgr A* flares orbit with ~45 min periods)")

plt.figure(figsize=(9, 5))
plt.plot(phases/(2*np.pi), face/face.max(), 'o-', label='face-on 5 deg')
plt.plot(phases/(2*np.pi), edge/edge.max(), 's-', label='edge-on 80 deg')
plt.xlabel('orbital phase'); plt.ylabel('flux (normalized to peak)')
plt.legend(); plt.grid(True)
plt.savefig('hotspot_lightcurves.png', dpi=110, bbox_inches='tight')
print("saved hotspot_lightcurves.png")
