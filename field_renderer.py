import numpy as np
import torch
import os, time
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray

TD = torch.float64
A, R_ORB, SIG = 0.0, 8.0, 1.0
OMEGA = 1.0/(R_ORB**1.5 + A)
UT = (R_ORB**1.5 + A)/np.sqrt(R_ORB**3 - 3.0*R_ORB**2 + 2.0*A*R_ORB**1.5)

def rho_F(r):
    return 0.3/(1.0 + (r/4.0)**2)

def wrap_diff(d):
    return (d + np.pi/2) % np.pi - np.pi/2

def load_or_trace(npix=64, extent=15.0, th_o_deg=20.0, cache='rays64_i20.npz'):
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"loaded ray cache {cache}")
        return {k: d[k] for k in d.files}
    th_o = np.radians(th_o_deg)
    DEVNULL = open(os.devnull, 'w')
    Xs, Ys, Zs, Rs, Cs, Ws, Gs = [], [], [], [], [], [], []
    t0 = time.time()
    for beta in np.linspace(-extent, extent, npix):
        for alpha in np.linspace(-extent, extent, npix):
            with redirect_stdout(DEVNULL):
                ray = trace_ray(A, th_o, alpha, beta)
            r = ray['r'][::-1]; th = ray['th'][::-1]; ph = ray['ph'][::-1]
            dl = np.sqrt(np.diff(r)**2 + (r[:-1]*np.diff(th))**2
                         + (r[:-1]*np.sin(th[:-1])*np.diff(ph))**2)
            s = np.concatenate(([0.0], np.cumsum(dl)))
            seg = rho_F(0.5*(r[:-1]+r[1:]))*dl
            C = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))
            w = np.empty_like(s)
            w[1:-1] = 0.5*(s[2:]-s[:-2]); w[0] = 0.5*(s[1]-s[0]); w[-1] = 0.5*(s[-1]-s[-2])
            Xs.append(r*np.sin(th)*np.cos(ph)); Ys.append(r*np.sin(th)*np.sin(ph))
            Zs.append(r*np.cos(th)); Rs.append(r); Cs.append(C); Ws.append(w)
            Gs.append(1.0/(UT*(1.0 - OMEGA*(-alpha*np.sin(th_o)))))
    out = dict(X=np.stack(Xs), Y=np.stack(Ys), Z=np.stack(Zs), R=np.stack(Rs),
               C=np.stack(Cs), W=np.stack(Ws), G=np.array(Gs))
    np.savez(cache, **out)
    print(f"traced {npix*npix} rays in {time.time()-t0:.0f} s, cached to {cache}")
    return out

class FieldRenderer:
    """Renders any emissivity field F(x,y,z), defined in the co-rotating frame,
    through the cached curved-spacetime rays with chromatic Faraday rotation.
    NOTE: redshift g is the r=8 circular-orbit value per ray (matches all Phase 4
    data). Per-point g(r) is the 5.4 upgrade if fields spread far in radius."""
    def __init__(self, rays):
        self.X = torch.tensor(rays['X'], dtype=TD)
        self.Y = torch.tensor(rays['Y'], dtype=TD)
        self.Z = torch.tensor(rays['Z'], dtype=TD)
        self.Rs = torch.tensor(rays['R'], dtype=TD)
        self.Ct = torch.tensor(rays['C'], dtype=TD)
        self.W = torch.tensor(rays['W'], dtype=TD)
        self.G3 = torch.tensor(rays['G']**3, dtype=TD)

    def render(self, field, t, s_dir, lam2, chi0, mode='rigid'):
        # co-rotation: sample the static field at lab coords rotated back by theta
        if mode == 'rigid':
            th = s_dir * OMEGA * t
            cph, sph = np.cos(th), np.sin(th)
        else:  # keplerian: per-point rotation rate Omega(r) (caveat: also applied
               # inside ISCO r<6 where circular orbits don't exist; emission there
               # is negligible for these scenes)
            th = s_dir * t / (self.Rs**1.5 + A)
            cph, sph = torch.cos(th), torch.sin(th)
        xp = self.X*cph + self.Y*sph
        yp = -self.X*sph + self.Y*cph
        e = field(xp, yp, self.Z)
        ew = e * self.W
        I = ew.sum(-1) * self.G3
        phase = 2.0*(chi0 + lam2*self.Ct)
        Q = 0.7*(ew*torch.cos(phase)).sum(-1) * self.G3
        U = 0.7*(ew*torch.sin(phase)).sum(-1) * self.G3
        return I, Q, U

def gaussian_field(xc, yc, sig=SIG):
    def f(x, y, z):
        return torch.exp(-((x-xc)**2 + (y-yc)**2 + z**2)/(2*sig**2))
    return f

if __name__ == '__main__':
    rays = load_or_trace()
    R = FieldRenderer(rays)
    obs = np.load('observation64.npz')
    S_TRUE, PHI0, CHI0 = 1, 0.7, 0.3
    LAM = {'230': 1.0, '213': (230.0/213.0)**2}
    times = obs['t']

    # ---- GATE A: co-rotating static field == moving blob, exactly ----
    fld = gaussian_field(R_ORB*np.cos(PHI0), R_ORB*np.sin(PHI0))
    maxd = 0.0
    for t in times:
        I1, Q1, U1 = R.render(fld, t, S_TRUE, 1.0, CHI0, mode='rigid')
        phb = S_TRUE*OMEGA*t + PHI0
        fmv = gaussian_field(R_ORB*np.cos(phb), R_ORB*np.sin(phb))
        I2, Q2, U2 = R.render(fmv, 0.0, S_TRUE, 1.0, CHI0, mode='rigid')
        sc = I2.max().item()
        maxd = max(maxd, (I1-I2).abs().max().item()/sc,
                   (Q1-Q2).abs().max().item()/sc, (U1-U2).abs().max().item()/sc)
    okA = maxd < 1e-8
    print(f"GATE A: max |co-rotating - moving blob| = {maxd:.2e}  "
          f"({'PASS' if okA else 'FAIL'}, bound 1e-8: same math, float roundoff only)")

    # ---- GATE B: regenerate observation64's clean curves through the new path ----
    Ic, dc = [], []
    for t in times:
        out = {}
        for k, lam2 in LAM.items():
            I, Q, U = R.render(fld, t, S_TRUE, lam2, CHI0, mode='rigid')
            out[k] = (I.sum().item(), Q.sum().item(), U.sum().item())
        Ic.append(out['230'][0])
        c230 = 0.5*np.arctan2(out['230'][2], out['230'][1])
        c213 = 0.5*np.arctan2(out['213'][2], out['213'][1])
        dc.append(wrap_diff(c213 - c230))
    Ic, dc = np.array(Ic), np.array(dc)
    ierr = 100*np.max(np.abs(Ic - obs['I_clean'])/obs['I_clean'])
    derr = np.degrees(np.max(np.abs(wrap_diff(dc - obs['dchi_clean']))))
    okB = (ierr < 3.0) and (derr < 0.5)
    print(f"GATE B: vs observation64 clean curves — max I err {ierr:.2f}% (bound 3%), "
          f"max dchi err {derr:.3f} deg (bound 0.5)  {'PASS' if okB else 'FAIL'}")
    print("  (bounds set by the RK4 pipeline's nearest-point sampling, cf. gate1_convergence.py)")

    # ---- DEMO C: Keplerian shear (informational) ----
    T_ORB = 2*np.pi/OMEGA
    I0, _, _ = R.render(fld, 0.0, S_TRUE, 1.0, CHI0, mode='keplerian')
    I1, _, _ = R.render(fld, T_ORB, S_TRUE, 1.0, CHI0, mode='keplerian')
    dOdr = 1.5/R_ORB**2.5
    print(f"DEMO C (keplerian): after one period, peak-pixel ratio "
          f"{I1.max().item()/I0.max().item():.3f}, total-I ratio {I1.sum().item()/I0.sum().item():.3f}")
    print(f"  predicted shear spread across +-1 sigma: {np.degrees(dOdr*2*SIG*T_ORB):.0f} deg "
          f"-- a rigid blob is an idealization; real hotspots smear into arcs within one orbit")

    print("OVERALL:", "PASS" if (okA and okB) else "FAIL")
