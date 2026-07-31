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
NPIX, EXTENT = 32, 15.0
TH_O = np.radians(20.0)
FREQS = {'230': 1.0, '213': (230.0/213.0)**2}
DLAM2 = FREQS['213'] - FREQS['230']
NAZ = 48

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

phis = np.linspace(0, 2*np.pi, NAZ, endpoint=False)
I230arr, Q230arr, U230arr, Q213arr, U213arr, Ctab, deptharr = [], [], [], [], [], [], []

for kph, phb in enumerate(phis):
    xb, yb = R_ORB*np.cos(phb), R_ORB*np.sin(phb)
    I230, Q230, U230, Q213, U213, Csum, Isum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
            if k == '230':
                I230 += S[0]; Q230 += S[1]; U230 += S[2]
            elif k == '213':
                Q213 += S[1]; U213 += S[2]
        Csum += Ss['230'][0] * np.sum(emis*ray['C'])/np.sum(emis)
        Isum += Ss['230'][0]
    if Isum > 0:
        Ctab.append(Csum / Isum)
    else:
        Ctab.append(0.0)
    deptharr.append(8*np.cos(phb)*np.sin(TH_O))
    I230arr.append(I230); Q230arr.append(Q230); U230arr.append(U230)
    Q213arr.append(Q213); U213arr.append(U213)
    print(f"  phi {kph+1:2d}/{NAZ}", end='\r')

print("\nSaving table...")
np.savez('obs_table.npz', phis=phis, I=np.array(I230arr), Q230=np.array(Q230arr),
         U230=np.array(U230arr), Q213=np.array(Q213arr), U213=np.array(U213arr),
         C=np.array(Ctab), depth=np.array(deptharr))

# Spot check
spot_phis = [0.11, 2.3, 4.87]
for phi in spot_phis:
    xb, yb = R_ORB*np.cos(phi), R_ORB*np.sin(phi)
    I_direct = []
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
            if k == '230':
                I_direct.append(S[0])
    I_interp = np.interp(phi, np.concatenate((phis, phis + 2*np.pi)), 
                         np.concatenate((I230arr, I230arr)))
    print(f"Spot check for phi={phi:.2f}: direct {np.mean(I_direct):.4f}, "
          f"interpolated {I_interp:.4f}, diff {(np.mean(I_direct) - I_interp)/np.mean(I_direct)*100:.2f}%")

# Degeneracy demo
T = 2*np.pi/OMEGA
t_k = np.linspace(0, T, 24)
phi_A = OMEGA * t_k
phi_B = -OMEGA * t_k + np.pi

I_A, dchi_A, depth_A = [], [], []
I_B, dchi_B, depth_B = [], [], []

for phi in phi_A:
    xb, yb = R_ORB*np.cos(phi), R_ORB*np.sin(phi)
    I230, Q230, U230, Q213, U213 = 0.0, 0.0, 0.0, 0.0, 0.0
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
            if k == '230':
                I230 += S[0]; Q230 += S[1]; U230 += S[2]
            elif k == '213':
                Q213 += S[1]; U213 += S[2]
    I_A.append(I230)
    dchi_A.append(wrap_diff(0.5*np.arctan2(U213, Q213) - 0.5*np.arctan2(U230, Q230)))
    depth_A.append(8*np.cos(phi)*np.sin(TH_O))

for phi in phi_B:
    xb, yb = R_ORB*np.cos(phi), R_ORB*np.sin(phi)
    I230, Q230, U230, Q213, U213 = 0.0, 0.0, 0.0, 0.0, 0.0
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
            if k == '230':
                I230 += S[0]; Q230 += S[1]; U230 += S[2]
            elif k == '213':
                Q213 += S[1]; U213 += S[2]
    I_B.append(I230)
    dchi_B.append(wrap_diff(0.5*np.arctan2(U213, Q213) - 0.5*np.arctan2(U230, Q230)))
    depth_B.append(8*np.cos(phi)*np.sin(TH_O))

max_I_diff = np.max(np.abs(np.array(I_A) - np.array(I_B))) / np.max(np.array(I_A)) * 100
depth_corr = np.corrcoef(depth_A, depth_B)[0,1]
dchi_corr = np.corrcoef(dchi_A, dchi_B)[0,1]

print(f"max|I_A - I_B|/max(I_A) = {max_I_diff:.2f}%")
print(f"depth correlation = {depth_corr:.4f}")
print(f"dchi correlation = {dchi_corr:.4f}")
print(f'intensity degeneracy leak (lensing asymmetry) = {max_I_diff:.2f}%')

if max_I_diff < 2.0 and depth_corr < -0.99 and dchi_corr < -0.5:
    print("PASS")
else:
    print("FAIL")
