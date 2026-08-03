import sys
import numpy as np
import os, time
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray
from coeffs import transfer_matrix

NPIX = int(sys.argv[1]) if len(sys.argv) > 1 else 48
OUT = sys.argv[2] if len(sys.argv) > 2 else f'obs_table{NPIX}.npz'

DEVNULL = open(os.devnull, 'w')
A, R_ORB, SIG = 0.0, 8.0, 1.0
OMEGA = 1.0/(R_ORB**1.5 + A)
UT = (R_ORB**1.5 + A)/np.sqrt(R_ORB**3 - 3.0*R_ORB**2 + 2.0*A*R_ORB**1.5)
EXTENT = 15.0
TH_O = np.radians(20.0)
FREQS = {'230': 1.0, '213': (230.0/213.0)**2}
NAZ = 48

def rho_F(r):
    return 0.3/(1.0 + (r/4.0)**2)

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

def render(phb):
    xb, yb = R_ORB*np.cos(phb), R_ORB*np.sin(phb)
    out = {k: np.zeros(3) for k in FREQS}
    Csum = Isum = 0.0
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
            Ss[k] = S
            out[k] += S[:3]
        Csum += Ss['230'][0] * np.sum(emis*ray['C'])/np.sum(emis)
        Isum += Ss['230'][0]
    Cw = Csum/Isum if Isum > 0 else 0.0
    return out, Cw

phis = np.linspace(0, 2*np.pi, NAZ, endpoint=False)
Iar, Q230ar, U230ar, Q213ar, U213ar, Car = [], [], [], [], [], []
t0 = time.time()
for kph, phb in enumerate(phis):
    out, Cw = render(phb)
    Iar.append(out['230'][0]); Q230ar.append(out['230'][1]); U230ar.append(out['230'][2])
    Q213ar.append(out['213'][1]); U213ar.append(out['213'][2]); Car.append(Cw)
    el = time.time()-t0
    print(f"  phi {kph+1:2d}/{NAZ}  ({el:.0f}s, ~{el/(kph+1)*(NAZ-kph-1):.0f}s left)", end='\r')
print()

np.savez(OUT, phis=phis, I=np.array(Iar), Q230=np.array(Q230ar), U230=np.array(U230ar),
         Q213=np.array(Q213ar), U213=np.array(U213ar), C=np.array(Car),
         depth=8*np.cos(phis)*np.sin(TH_O))
print(f"saved {OUT}")

phis_e = np.append(phis, phis[0]+2*np.pi)
I_e = np.append(Iar, Iar[0])
ok = True
for phi in [0.11, 2.3, 4.87]:
    out, _ = render(phi)
    Ii = np.interp(phi, phis_e, I_e)
    d = (out['230'][0]-Ii)/out['230'][0]*100
    ok = ok and abs(d) < 2.0
    print(f"spot phi={phi:.2f}: direct {out['230'][0]:.4f}  interp {Ii:.4f}  diff {d:+.2f}%")
print("spot checks:", "PASS" if ok else "FAIL")
