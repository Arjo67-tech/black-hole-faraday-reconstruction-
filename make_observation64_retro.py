import numpy as np
import os, time
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray
from coeffs import transfer_matrix

DEVNULL = open(os.devnull, 'w')
A, R_ORB, SIG = 0.0, 8.0, 1.0
OMEGA = 1.0/(R_ORB**1.5 + A)
UT = (R_ORB**1.5 + A)/np.sqrt(R_ORB**3 - 3.0*R_ORB**2 + 2.0*A*R_ORB**1.5)
NPIX, EXTENT, NPHASE = 64, 15.0, 24
TH_O = np.radians(20.0)
FREQS = {'230': 1.0, '213': (230.0/213.0)**2}
DLAM2 = FREQS['213'] - FREQS['230']
NOISE_AMP = 0.01
NOISE_ANG = np.radians(1.0)

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

S_TRUE, PHI0_TRUE, CHI0_TRUE = -1, 0.7, 0.3
NPHASE = 24
T_ORB = 2*np.pi/OMEGA
times = np.arange(NPHASE) * T_ORB / NPHASE

print("rendering clean frames (24)...")
I_clean, Q230_clean, U230_clean, Q213_clean, U213_clean = [], [], [], [], []
t0 = time.time()
for kt, t in enumerate(times):
    phb = S_TRUE*OMEGA*t + PHI0_TRUE
    xb, yb = R_ORB*np.cos(phb), R_ORB*np.sin(phb)
    sums = {k: np.zeros(3) for k in FREQS}
    for ray in rays:
        emis = np.exp(-((ray['x']-xb)**2 + (ray['y']-yb)**2 + ray['z']**2)/(2*SIG**2))
        if emis.max() < 1e-8:
            continue
        s, r, g3 = ray['s'], ray['r'], ray['g']**3
        for k, lam2 in FREQS.items():
            def j_of_s(sv):
                i = np.argmin(np.abs(s - sv)); e = emis[i]
                return np.array([e, 0.7*e*np.cos(2*CHI0_TRUE), 0.7*e*np.sin(2*CHI0_TRUE), 0.0])
            def K_of_s(sv):
                i = np.argmin(np.abs(s - sv))
                return transfer_matrix(0,0,0,0, 0.0, 0.0, 2.0*rho_F(r[i])*lam2)
            S = integrate_ray(s, j_of_s, K_of_s, np.zeros(4))[-1]*g3
            sums[k] += S[:3]
    I_clean.append(sums['230'][0])
    Q230_clean.append(sums['230'][1])
    U230_clean.append(sums['230'][2])
    Q213_clean.append(sums['213'][1])
    U213_clean.append(sums['213'][2])
    print(f"  {kt+1}/{NPHASE}", end='\r')
print(f"\n  done in {time.time()-t0:.0f} s")

I_clean = np.array(I_clean)
Q230_clean = np.array(Q230_clean); U230_clean = np.array(U230_clean)
Q213_clean = np.array(Q213_clean); U213_clean = np.array(U213_clean)
chi230_clean = 0.5*np.arctan2(U230_clean, Q230_clean)
chi213_clean = 0.5*np.arctan2(U213_clean, Q213_clean)
dchi_clean = wrap_diff(chi213_clean - chi230_clean)

rng = np.random.default_rng(42)
I_noisy = I_clean * (1.0 + NOISE_AMP*rng.standard_normal(NPHASE))
Q230_noisy = Q230_clean + 0.01*I_clean*rng.standard_normal(NPHASE)
U230_noisy = U230_clean + 0.01*I_clean*rng.standard_normal(NPHASE)
Q213_noisy = Q213_clean + 0.01*I_clean*rng.standard_normal(NPHASE)
U213_noisy = U213_clean + 0.01*I_clean*rng.standard_normal(NPHASE)
chi230_noisy = 0.5*np.arctan2(U230_noisy, Q230_noisy) + rng.standard_normal(NPHASE)*NOISE_ANG
chi213_noisy = 0.5*np.arctan2(U213_noisy, Q213_noisy) + rng.standard_normal(NPHASE)*NOISE_ANG
dchi_noisy = wrap_diff(chi213_noisy - chi230_noisy)

np.savez('observation64_retro.npz', t=times, I_clean=I_clean, I_noisy=I_noisy,
    dchi_clean=dchi_clean, dchi_noisy=dchi_noisy,
    chi230_noisy=chi230_noisy, chi213_noisy=chi213_noisy,
    s_true=S_TRUE, phi0_true=PHI0_TRUE, chi0_true=CHI0_TRUE)

peak_dchi = np.degrees(np.abs(dchi_clean).max())
sigma_dchi = np.degrees(np.sqrt(2)*np.radians(1.0))
snr = peak_dchi / sigma_dchi
print(f"peak |dchi_clean|: {peak_dchi:.2f} deg")
print(f"noise level (sqrt(2)*1 deg): {sigma_dchi:.2f} deg")
print(f"depth-signal SNR per phase: {snr:.2f}")
print(f"peak-to-trough I: {I_clean.max()-I_clean.min():.1f}   5% noise level: {0.05*I_clean.max():.1f}")
print("PASS" if (snr > 5 and 'observation64_retro.npz' in os.listdir('.')) else "FAIL")
