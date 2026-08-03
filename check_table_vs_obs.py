import numpy as np
tab = np.load('obs_table48.npz'); obs = np.load('observation48.npz')
OMEGA = 1.0/8.0**1.5
def wrap_diff(d): return (d + np.pi/2) % np.pi - np.pi/2
def ext(c): return np.append(c, c[0])
pe = np.append(tab['phis'], tab['phis'][0]+2*np.pi)
phi = (1*OMEGA*obs['t'] + 0.7) % (2*np.pi)
I_m = np.interp(phi, pe, ext(tab['I']))
dchi_m = wrap_diff(0.5*np.arctan2(np.interp(phi, pe, ext(tab['U213'])), np.interp(phi, pe, ext(tab['Q213'])))
                 - 0.5*np.arctan2(np.interp(phi, pe, ext(tab['U230'])), np.interp(phi, pe, ext(tab['Q230']))))
F = np.sum(obs['I_clean']*I_m)/np.sum(I_m**2)
print(f"flux scale (expect ~1.0): {F:.4f}")
print(f"max |I mismatch|: {100*np.max(np.abs(obs['I_clean']-F*I_m)/obs['I_clean']):.2f}%")
print(f"max |dchi mismatch|: {np.degrees(np.max(np.abs(wrap_diff(obs['dchi_clean']-dchi_m)))):.3f} deg")
C = tab['C']
print(f"C: min {C.min():.3f} at phi={np.degrees(tab['phis'][np.argmin(C)]):.0f} deg, "
      f"max {C.max():.3f} at phi={np.degrees(tab['phis'][np.argmax(C)]):.0f} deg")
print("first 12 C values:", np.round(C[:12], 3))
