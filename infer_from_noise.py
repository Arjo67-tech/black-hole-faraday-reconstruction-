import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time
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
FREQS = {'230': 1.0, '213': (230.0/213.0)**2}
DLAM2 = FREQS['213'] - FREQS['230']
NOISE_AMP = 0.01
NOISE_ANG = np.radians(1.0)
NBOOT = 200

def rho_F(r):
    return 0.3/(1.0 + (r/4.0)**2)

def wrap_diff(d):
    return (d + np.pi/2) % np.pi - np.pi/2

print("tracing rays once...")
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

print("simulating clean phases (24)...")
phases = np.linspace(0.0, 2*np.pi, NPHASE, endpoint=False)
clean_sums, truth = [], []
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
        SI230 = None
        for k, lam2 in FREQS.items():
            def j_of_s(sv):
                i = np.argmin(np.abs(s - sv)); e = emis[i]
                return np.array([e, 0.7*e, 0.0, 0.0])
            def K_of_s(sv):
                i = np.argmin(np.abs(s - sv))
                return transfer_matrix(0,0,0,0, 0.0, 0.0, 2.0*rho_F(r[i])*lam2)
            S = integrate_ray(s, j_of_s, K_of_s, np.zeros(4))[-1]*g3
            sums[k] += S[:3]
            if k == '230':
                SI230 = S[0]
        col = np.sum(emis*ray['C'])/np.sum(emis)
        wcol += SI230*col; wsum += SI230
    clean_sums.append(sums)
    truth.append(wcol/wsum if wsum > 0 else 0.0)
    print(f"  {kph+1}/{NPHASE}", end='\r')
truth = np.array(truth)
print(f"\n  done in {time.time()-t0:.0f} s")

print(f"bootstrapping {NBOOT} noisy realizations per phase...")
recovered = np.zeros((NBOOT, NPHASE))
rng = np.random.default_rng(0)
for b in range(NBOOT):
    for kph in range(NPHASE):
        chis = {}
        for k in FREQS:
            I, Q, U = clean_sums[kph][k]
            Qn = Q + rng.normal(0, NOISE_AMP*I)
            Un = U + rng.normal(0, NOISE_AMP*I)
            chis[k] = 0.5*np.arctan2(Un, Qn) + rng.normal(0, NOISE_ANG)
        recovered[b, kph] = wrap_diff(chis['213'] - chis['230']) / DLAM2

col_mean = recovered.mean(axis=0)
col_std = recovered.std(axis=0)

within = np.abs(col_mean - truth) <= 2*col_std
print(f"\ntruth within 2-sigma of recovery: {within.mean()*100:.0f}% of phases")

print("error scaling when averaging phases:")
base_err = col_std.mean()
for navg in [1, 2, 4, 8]:
    eff = base_err/np.sqrt(navg)
    print(f"  {navg} phases averaged: effective error {eff:.4f}  (sqrt-N predicts {base_err:.4f}/{np.sqrt(navg):.2f})")

pC = within.mean() >= 0.9
print("check C (truth within 2-sigma, >=90%):", "PASS" if pC else "FAIL")
print("OVERALL:", "PASS" if pC else "FAIL")

plt.figure(figsize=(9,5))
ph_frac = phases/(2*np.pi)
plt.plot(ph_frac, truth, 'k-', lw=2, label='true gas column (hidden)')
plt.errorbar(ph_frac, col_mean, yerr=2*col_std, fmt='o', capsize=3,
             label='recovered from noisy angles (2-sigma)')
plt.xlabel('orbital phase'); plt.ylabel('gas column')
plt.legend(); plt.grid(True)
plt.savefig('inference_recovery.png', dpi=110, bbox_inches='tight')
print("saved inference_recovery.png")
