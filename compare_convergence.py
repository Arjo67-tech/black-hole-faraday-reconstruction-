import numpy as np
t48 = np.load('obs_table48.npz')
t64 = np.load('obs_table64.npz')
def resample(tab, phis_new):
    pe = np.append(tab['phis'], tab['phis'][0]+2*np.pi)
    Ce = np.append(tab['C'], tab['C'][0])
    return np.interp(phis_new, pe, Ce)
phis_common = t48['phis']
C48 = t48['C']
C64_on_48grid = resample(t64, phis_common)
diff = np.abs(C48 - C64_on_48grid)
print(f"max |C48 - C64| on common grid: {diff.max():.4f}")
print(f"mean |C48 - C64|: {diff.mean():.4f}")
print(f"C48 range: {C48.min():.3f} to {C48.max():.3f}")
print(f"C64 range: {t64['C'].min():.3f} to {t64['C'].max():.3f}")
print("\nside by side (every 4th point):")
for i in range(0, len(phis_common), 4):
    print(f"  phi={np.degrees(phis_common[i]):6.1f}  C48={C48[i]:.3f}  C64={C64_on_48grid[i]:.3f}  diff={diff[i]:+.3f}")
