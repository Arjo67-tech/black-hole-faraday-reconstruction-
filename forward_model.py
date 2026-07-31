import numpy as np
import os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray
from coeffs import transfer_matrix

DEVNULL = open(os.devnull, 'w')

A, R_ORB, SIG = 0.0, 8.0, 1.0
OMEGA = 1.0/(R_ORB**1.5 + A)
UT = (R_ORB**1.5 + A)/np.sqrt(R_ORB**3 - 3.0*R_ORB**2 + 2.0*A*R_ORB**1.5)
NPIX, EXTENT, NPHASE = 32, 15.0, 24
TH_O = np.radians(80.0)
FREQS = {'345': (230.0/345.0)**2, '230': 1.0, '213': (230.0/213.0)**2}
DLAM2 = FREQS['213'] - FREQS['230']

def rho_F(r):
    return 0.3/(1.0 + (r/4.0)**2)

def wrap_diff(d):
    return (d + np.pi/2) % np.pi - np.pi/2

print(f"tracing {NPIX*NPIX} rays once...")
t0 = time.time()
rays = []
for beta in np.linspace(-EXTENT, EXTENT, NPIX):
    for alpha in np.linspace(-EXTENT, EXTENT, NPIX):
        with redirect_stdout(DEVNULL):
            ray = trace_ray(A, TH_O, alpha, beta)
        r = ray['r'][::-1]; th = ray['th'][::-1]; ph = ray['ph'][::-1]
        dl = np.sqrt(np.diff(r)**2 + (r[:-1]*np.diff(th))**2
                     + (r[:-1]*np.sin(th[:-1])*np.diff(ph))**2)
        s = np.concatenate(([0.0], np.cumsum(dl)))
        seg = rho_F(0.5*(r[:-1]+r[1:]))*dl
        C = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))
        rays.append(dict(s=s, r=r,
            x=r*np.sin(th)*np.cos(ph), y=r*np.sin(th)*np.sin(ph),
            z=r*np.cos(th), C=C,
            g=1.0/(UT*(1.0 - OMEGA*(-alpha*np.sin(TH_O))))))
print(f"  done in {time.time()-t0:.0f} s")

phases = np.linspace(0.0, 2*np.pi, NPHASE, endpoint=False)
I_ser = {k: [] for k in FREQS}; Q_ser = {k: [] for k in FREQS}; U_ser = {k: [] for k in FREQS}
dchi_img, pred_img = [], []
ray_ok = ray_tot = 0

t0 = time.time()
for kph, phb in enumerate(phases):
    xb, yb = R_ORB*np.cos(phb), R_ORB*np.sin(phb)
    sums = {k: np.zeros(3) for k in FREQS}
    wcol = wsum = 0.0
    for ray in rays:
        emis = np.exp(-((ray['x']-xb)**2 + (ray['y']-yb)**2 + ray['z']**2)/(2*SIG**2))
        if emis.max() < 1e-8:
            continue
        s, r, g3 = ray['s'], ray['r'], ray['g']**3
        Ss = {}
        for k, lam2 in FREQS.items():
            def j_of_s(sv):
                i = np.argmin(np.abs(s - sv)); e = emis[i]
                return np.array([e, 0.7*e, 0.0, 0.0])
            def K_of_s(sv):
                i = np.argmin(np.abs(s - sv))
                return transfer_matrix(0,0,0,0, 0.0, 0.0, 2.0*rho_F(r[i])*lam2)
            S = integrate_ray(s, j_of_s, K_of_s, np.zeros(4))[-1]*g3
            sums[k] += S[:3]; Ss[k] = S
        c230 = 0.5*np.arctan2(Ss['230'][2], Ss['230'][1])
        c213 = 0.5*np.arctan2(Ss['213'][2], Ss['213'][1])
        d_ray = wrap_diff(c213 - c230)
        p_ray = DLAM2 * np.sum(emis*ray['C'])/np.sum(emis)
        if p_ray > 0.02:
            ray_tot += 1
            if abs(d_ray - p_ray) <= 0.10*p_ray:
                ray_ok += 1
        wcol += Ss['230'][0] * p_ray; wsum += Ss['230'][0]
    for k in FREQS:
        I_ser[k].append(sums[k][0]); Q_ser[k].append(sums[k][1]); U_ser[k].append(sums[k][2])
    ci230 = 0.5*np.arctan2(sums['230'][2], sums['230'][1])
    ci213 = 0.5*np.arctan2(sums['213'][2], sums['213'][1])
    dchi_img.append(wrap_diff(ci213 - ci230))
    pred_img.append(wcol/wsum if wsum > 0 else 0.0)
    print(f"  phase {kph+1:2d}/{NPHASE}", end='\r')
print(f"\nphases done in {time.time()-t0:.0f} s")

dchi_img = np.array(dchi_img); pred_img = np.array(pred_img)

print(f"\nGate A (every contributing ray, all phases): {ray_ok}/{ray_tot} within 10%")
pA = ray_tot > 0 and ray_ok/ray_tot >= 0.9
imax = int(np.argmax(dchi_img))
print(f"Gate B: image-integrated dchi peaks at phase index {imax} (behind-half window 8..16)")
pB = 8 <= imax <= 16
print("check A (per-ray physics):   ", "PASS" if pA else "FAIL")
print("check B (peaks blob-behind): ", "PASS" if pB else "FAIL")
print("OVERALL:", "PASS" if (pA and pB) else "FAIL")
print(f"peak image sideband difference: {np.degrees(dchi_img.max()):.1f} deg "
      f"(real Sgr A* ALMA sidebands: ~7.65 deg — same observable)")

ph_frac = phases/(2*np.pi)
plt.figure(figsize=(9,5))
for k in FREQS:
    plt.plot(ph_frac, np.array(I_ser[k])/max(I_ser['230']), 'o-', label=f'{k} GHz')
plt.xlabel('orbital phase'); plt.ylabel('I (rel.)'); plt.legend(); plt.grid(True)
plt.savefig('forward_lightcurves.png', dpi=110, bbox_inches='tight')

plt.figure(figsize=(9,5))
for k in FREQS:
    chi = np.unwrap(0.5*np.arctan2(np.array(U_ser[k]), np.array(Q_ser[k])), period=np.pi)
    plt.plot(ph_frac, chi, 'o-', label=f'chi {k} GHz')
plt.plot(ph_frac, dchi_img, 'k^-', label='image dchi 213-230 (measured)')
plt.plot(ph_frac, pred_img, 'r--', label='flux-weighted column (guide)')
plt.xlabel('orbital phase'); plt.ylabel('angle (rad)'); plt.legend(); plt.grid(True)
plt.savefig('forward_chi.png', dpi=110, bbox_inches='tight')

plt.figure(figsize=(6,6))
for k in FREQS:
    plt.plot(Q_ser[k], U_ser[k], 'o-', label=f'{k} GHz')
plt.xlabel('Q'); plt.ylabel('U'); plt.legend(); plt.grid(True); plt.axis('equal')
plt.savefig('forward_QU.png', dpi=110, bbox_inches='tight')
print("saved forward_lightcurves.png, forward_chi.png, forward_QU.png")
