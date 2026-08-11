import numpy as np
import torch
import os, time
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

def rho_F(r):
    return 0.3/(1.0 + (r/4.0)**2)

def wrap_diff(d):
    return (d + np.pi/2) % np.pi - np.pi/2

print("caching rays...")
t0 = time.time()
Xs, Ys, Zs, Ss, Rs, Cs, Ws, Gs = [], [], [], [], [], [], [], []
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
        w = np.empty_like(s)
        w[1:-1] = 0.5*(s[2:] - s[:-2]); w[0] = 0.5*(s[1]-s[0]); w[-1] = 0.5*(s[-1]-s[-2])
        Xs.append(r*np.sin(th)*np.cos(ph)); Ys.append(r*np.sin(th)*np.sin(ph))
        Zs.append(r*np.cos(th)); Ss.append(s); Rs.append(r); Cs.append(C); Ws.append(w)
        Gs.append(1.0/(UT*(1.0 - OMEGA*(-alpha*np.sin(TH_O)))))
X_np, Y_np, Z_np = np.stack(Xs), np.stack(Ys), np.stack(Zs)
S_np, R_np, C_np, W_np = np.stack(Ss), np.stack(Rs), np.stack(Cs), np.stack(Ws)
g_np = np.array(Gs)
print(f"  {len(g_np)} rays in {time.time()-t0:.0f} s")

td = torch.float64
X = torch.tensor(X_np, dtype=td); Y = torch.tensor(Y_np, dtype=td)
Z = torch.tensor(Z_np, dtype=td); Ct = torch.tensor(C_np, dtype=td)
W = torch.tensor(W_np, dtype=td); G3 = torch.tensor(g_np**3, dtype=td)

def render(xb, yb, chi0, lam2):
    """closed-form: each emitted packet exits rotated by lam2*C(point)"""
    e = torch.exp(-((X-xb)**2 + (Y-yb)**2 + Z**2)/(2*SIG**2))
    ew = e*W
    I = ew.sum(-1)*G3
    phase = 2.0*(chi0 + lam2*Ct)
    Q = 0.7*(ew*torch.cos(phase)).sum(-1)*G3
    U = 0.7*(ew*torch.sin(phase)).sum(-1)*G3
    return I, Q, U

# ---------- GATE 1: match integrate_ray (RK4) per ray and in image totals ----------
PHB, CHI0 = 0.7, 0.3
xb0, yb0 = R_ORB*np.cos(PHB), R_ORB*np.sin(PHB)
lam2s = {'230': 1.0, '213': (230.0/213.0)**2, '345': (230.0/345.0)**2}

def rk4_ray(idx, lam2):
    s, r = S_np[idx], R_np[idx]
    e = np.exp(-((X_np[idx]-xb0)**2 + (Y_np[idx]-yb0)**2 + Z_np[idx]**2)/(2*SIG**2))
    def j_of_s(sv):
        i = np.argmin(np.abs(s - sv))
        return np.array([e[i], 0.7*e[i]*np.cos(2*CHI0), 0.7*e[i]*np.sin(2*CHI0), 0.0])
    def K_of_s(sv):
        i = np.argmin(np.abs(s - sv))
        return transfer_matrix(0,0,0,0, 0.0, 0.0, 2.0*rho_F(r[i])*lam2)
    return integrate_ray(s, j_of_s, K_of_s, np.zeros(4))[-1]*(g_np[idx]**3)

ok1 = True
for name, lam2 in lam2s.items():
    I, Q, U = render(torch.tensor(xb0), torch.tensor(yb0), torch.tensor(CHI0), lam2)
    In, Qn, Un = I.numpy(), Q.numpy(), U.numpy()
    idxs = np.where(In > 1e-8*In.max())[0]
    maxIerr = maxAng = 0.0
    tI = tQ = tU = rI = rQ = rU = 0.0
    for idx in idxs:
        Srk = rk4_ray(idx, lam2)
        maxIerr = max(maxIerr, abs(In[idx]-Srk[0])/Srk[0])
        a_t = 0.5*np.arctan2(Un[idx], Qn[idx]); a_r = 0.5*np.arctan2(Srk[2], Srk[1])
        maxAng = max(maxAng, abs(np.degrees(wrap_diff(a_t - a_r))))
        tI += In[idx]; tQ += Qn[idx]; tU += Un[idx]
        rI += Srk[0]; rQ += Srk[1]; rU += Srk[2]
    totErr = max(abs(tI-rI)/rI, abs(tQ-rQ)/abs(rI), abs(tU-rU)/abs(rI))
    print(f"lam2={name}: {len(idxs)} rays  max per-ray I err {100*maxIerr:.3f}%  "
          f"max angle err {maxAng:.4f} deg  image-total err {100*totErr:.3f}%")
    ok1 = ok1 and (maxIerr < 0.03) and (maxAng < 5.0) and (totErr < 0.03)   # bounds set by RK4 nearest-point sampling at n=500; gate1_convergence.py shows both engines converge to the same answer (0.20%->0.01% over 500->4000 pts)
print("GATE 1 (matches RK4 within n=500 harness precision, see gate1_convergence.py):", "PASS" if ok1 else "FAIL")

# ---------- GATE 2: speed ----------
t0 = time.time()
with torch.no_grad():
    for k in range(24):
        phb = 2*np.pi*k/24
        for lam2 in lam2s.values():
            render(torch.tensor(R_ORB*np.cos(phb)), torch.tensor(R_ORB*np.sin(phb)),
                   torch.tensor(CHI0), lam2)
dt = time.time()-t0
print(f"GATE 2: 24 phases x 3 freqs in {dt:.2f} s  "
      f"(numpy RK4 pipeline: ~300-400 s at this grid -> ~{350/dt:.0f}x speedup)")

# ---------- GATE 3: gradients — recover blob position by descent ----------
with torch.no_grad():
    I_t, Q_t, U_t = render(torch.tensor(xb0), torch.tensor(yb0), torch.tensor(CHI0), 1.0)
xb = torch.tensor(4.5, dtype=td, requires_grad=True)
yb = torch.tensor(2.5, dtype=td, requires_grad=True)
opt = torch.optim.Adam([xb, yb], lr=0.1)
for it in range(150):
    opt.zero_grad()
    I, Q, U = render(xb, yb, torch.tensor(CHI0), 1.0)
    loss = ((I-I_t)**2).sum() + ((Q-Q_t)**2).sum() + ((U-U_t)**2).sum()
    loss.backward()
    opt.step()
err = np.hypot(xb.item()-xb0, yb.item()-yb0)
print(f"GATE 3: started (4.50, 2.50), truth ({xb0:.3f}, {yb0:.3f}), "
      f"recovered ({xb.item():.3f}, {yb.item():.3f})  err {err:.4f} M")
ok3 = err < 0.05
print("GATE 3 (gradient recovery < 0.05 M):", "PASS" if ok3 else "FAIL")
print("OVERALL:", "PASS" if (ok1 and ok3) else "FAIL")
