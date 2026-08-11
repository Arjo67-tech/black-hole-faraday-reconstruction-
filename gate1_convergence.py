import numpy as np
import torch, os
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray
from coeffs import transfer_matrix

DEVNULL = open(os.devnull, 'w')
A, R_ORB, SIG = 0.0, 8.0, 1.0
OMEGA = 1.0/(R_ORB**1.5 + A)
UT = (R_ORB**1.5 + A)/np.sqrt(R_ORB**3 - 3.0*R_ORB**2 + 2.0*A*R_ORB**1.5)
TH_O = np.radians(20.0)
PHB, CHI0, LAM2 = 0.7, 0.3, 1.0
xb0, yb0 = R_ORB*np.cos(PHB), R_ORB*np.sin(PHB)

def rho_F(r): return 0.3/(1.0 + (r/4.0)**2)
def wrap_diff(d): return (d + np.pi/2) % np.pi - np.pi/2

# one ray that passes near the blob (aim at its sky position, roughly)
ALPHA, BETA = 5.32, -6.29

for npts in [500, 1000, 2000, 4000]:
    with redirect_stdout(DEVNULL):
        ray = trace_ray(A, TH_O, ALPHA, BETA, n_points=npts)
    r = ray['r'][::-1]; th = ray['th'][::-1]; ph = ray['ph'][::-1]
    dl = np.sqrt(np.diff(r)**2 + (r[:-1]*np.diff(th))**2
                 + (r[:-1]*np.sin(th[:-1])*np.diff(ph))**2)
    s = np.concatenate(([0.0], np.cumsum(dl)))
    seg = rho_F(0.5*(r[:-1]+r[1:]))*dl
    C = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))
    x = r*np.sin(th)*np.cos(ph); y = r*np.sin(th)*np.sin(ph); z = r*np.cos(th)
    g3 = (1.0/(UT*(1.0 - OMEGA*(-ALPHA*np.sin(TH_O)))))**3
    e = np.exp(-((x-xb0)**2 + (y-yb0)**2 + z**2)/(2*SIG**2))
    if e.max() < 1e-8:
        print(f"n={npts}: ray misses blob, pick different ALPHA,BETA"); break

    # closed form (trapezoid weights)
    w = np.empty_like(s)
    w[1:-1] = 0.5*(s[2:]-s[:-2]); w[0] = 0.5*(s[1]-s[0]); w[-1] = 0.5*(s[-1]-s[-2])
    ew = e*w
    I_cf = ew.sum()*g3
    ph2 = 2.0*(CHI0 + LAM2*C)
    Q_cf = 0.7*(ew*np.cos(ph2)).sum()*g3
    U_cf = 0.7*(ew*np.sin(ph2)).sum()*g3

    # RK4 with nearest-point j and K
    def j_of_s(sv):
        i = np.argmin(np.abs(s - sv))
        return np.array([e[i], 0.7*e[i]*np.cos(2*CHI0), 0.7*e[i]*np.sin(2*CHI0), 0.0])
    def K_of_s(sv):
        i = np.argmin(np.abs(s - sv))
        return transfer_matrix(0,0,0,0, 0.0, 0.0, 2.0*rho_F(r[i])*LAM2)
    Srk = integrate_ray(s, j_of_s, K_of_s, np.zeros(4))[-1]*g3

    dI = 100*abs(I_cf-Srk[0])/Srk[0]
    aT = 0.5*np.arctan2(U_cf, Q_cf); aR = 0.5*np.arctan2(Srk[2], Srk[1])
    dA = abs(np.degrees(wrap_diff(aT-aR)))
    print(f"n={npts:5d}:  I diff {dI:.4f}%   angle diff {dA:.4f} deg")
