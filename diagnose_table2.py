import numpy as np
d = np.load('obs_table.npz')
phis, C = d['phis'], d['C']
DLAM2 = (230.0/213.0)**2 - 1.0
def wrap_diff(x): return (x + np.pi/2) % np.pi - np.pi/2
chi230 = 0.5*np.arctan2(d['U230'], d['Q230'])
chi213 = 0.5*np.arctan2(d['U213'], d['Q213'])
dchi = wrap_diff(chi213 - chi230)
pred = DLAM2 * C

print(f"C:    min {C.min():.3f}  max {C.max():.3f}  peak-to-peak {C.max()-C.min():.3f}")
print(f"dchi: min {np.degrees(dchi.min()):.2f} deg  max {np.degrees(dchi.max()):.2f} deg  "
      f"peak-to-peak {np.degrees(dchi.max()-dchi.min()):.2f} deg")
print(f"rms(dchi - pred): {np.degrees(np.std(dchi-pred)):.2f} deg  (pixelation jitter)")
print(f"signal/jitter ratio: {(dchi.max()-dchi.min())/np.std(dchi-pred):.2f}")

# find which azimuths give min/max C -- that tells us where "front" and "back" really are
i_min, i_max = np.argmin(C), np.argmax(C)
print(f"\nmin C at phi={np.degrees(phis[i_min]):.0f} deg, max C at phi={np.degrees(phis[i_max]):.0f} deg")
print(f"(pure front/back would put these at 0 and 180 deg)")
