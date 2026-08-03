import numpy as np
d = np.load('obs_table.npz')
phis, C = d['phis'], d['C']
DLAM2 = (230.0/213.0)**2 - 1.0
def wrap_diff(x): return (x + np.pi/2) % np.pi - np.pi/2
chi230 = 0.5*np.arctan2(d['U230'], d['Q230'])
chi213 = 0.5*np.arctan2(d['U213'], d['Q213'])
dchi = wrap_diff(chi213 - chi230)
pred = DLAM2 * C
resid = dchi - pred
c1 = 2*np.mean(dchi*np.cos(phis)); s1 = 2*np.mean(dchi*np.sin(phis))
print(f"C range: {C.min():.3f} (front) to {C.max():.3f} (back)   flat-space theory: 0.710 to 1.061")
print(f"cos-phi amplitude of dchi: {np.degrees(c1):+.2f} deg   <- the front/back signal")
print(f"sin-phi amplitude of dchi: {np.degrees(s1):+.2f} deg   <- should be ~0 by symmetry")
print(f"rms(dchi - DLAM2*C):       {np.degrees(np.std(resid)):.2f} deg   <- pixelation jitter")
print("\n phi_deg     C     dchi_deg  pred_deg  resid")
for i in range(0, len(phis), 4):
    print(f"{np.degrees(phis[i]):7.1f} {C[i]:7.3f} {np.degrees(dchi[i]):8.2f} {np.degrees(pred[i]):8.2f} {np.degrees(resid[i]):+7.2f}")
