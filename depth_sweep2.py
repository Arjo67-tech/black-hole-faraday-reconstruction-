import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OBS_FILE = sys.argv[1] if len(sys.argv) > 1 else 'observation.npz'
tab = np.load('obs_table.npz')
obs = np.load(OBS_FILE)

OMEGA = 1.0/8.0**1.5
SIGMA_DCHI = np.radians(1.5)
NBOOT = 200
SIGMAS_I = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

def wrap_diff(d):
    return (d + np.pi/2) % np.pi - np.pi/2
def ext(c):
    return np.append(c, c[0])

phis_e = np.append(tab['phis'], tab['phis'][0] + 2*np.pi)
I_e = ext(tab['I'])
Q230_e, U230_e = ext(tab['Q230']), ext(tab['U230'])
Q213_e, U213_e = ext(tab['Q213']), ext(tab['U213'])

t = obs['t']
I_clean, dchi_clean = obs['I_clean'], obs['dchi_clean']
s_true = int(obs['s_true'])

phi0_grid = np.linspace(0, 2*np.pi, 128, endpoint=False)
models = {}
for s in [1, -1]:
    for phi0 in phi0_grid:
        phi = (s*OMEGA*t + phi0) % (2*np.pi)
        I_m = np.interp(phi, phis_e, I_e)
        dchi_m = wrap_diff(0.5*np.arctan2(np.interp(phi, phis_e, U213_e),
                                          np.interp(phi, phis_e, Q213_e))
                         - 0.5*np.arctan2(np.interp(phi, phis_e, U230_e),
                                          np.interp(phi, phis_e, Q230_e)))
        models[(s, phi0)] = (I_m, dchi_m)

def run_fit(I_d, dchi_d, sig_I, use_I, use_dchi):
    best, chi2_s = None, {1: np.inf, -1: np.inf}
    for (s, phi0), (I_m, dchi_m) in models.items():
        chi2 = 0.0
        if use_I:
            F = np.sum(I_d*I_m)/np.sum(I_m**2)
            sigma = sig_I * np.maximum(F*I_m, 1e-3*np.max(np.abs(F*I_m)))
            chi2 += np.sum(((I_d - F*I_m)/sigma)**2)   # model-based weighting
        if use_dchi:
            chi2 += np.sum((wrap_diff(dchi_d - dchi_m)/SIGMA_DCHI)**2)
        chi2_s[s] = min(chi2_s[s], chi2)
        if best is None or chi2 < best[0]:
            best = (chi2, s)
    return (best[1] == s_true), abs(chi2_s[1] - chi2_s[-1])

rng = np.random.default_rng(11)
curves = {1: [], 2: [], 3: []}
print(f"data file: {OBS_FILE}")
for sig_I in SIGMAS_I:
    ok = {1: 0, 2: 0, 3: 0}; g = {1: [], 2: [], 3: []}
    for b in range(NBOOT):
        I_d = I_clean*(1 + sig_I*rng.standard_normal(len(t)))
        dchi_d = dchi_clean + np.radians(np.sqrt(2)*1.0)*rng.standard_normal(len(t))
        for m, (uI, ud) in {1: (True, False), 2: (True, True), 3: (False, True)}.items():
            c, gap = run_fit(I_d, dchi_d, sig_I, uI, ud)
            ok[m] += c; g[m].append(gap)
    for m in [1, 2, 3]:
        curves[m].append(100.0*ok[m]/NBOOT)
    print(f"sigma_I={sig_I:.2f}:  M1 {curves[1][-1]:5.1f}%  M2 {curves[2][-1]:5.1f}%  "
          f"M3 {curves[3][-1]:5.1f}%   (median dchi2: {np.median(g[1]):.1f} / {np.median(g[2]):.1f} / {np.median(g[3]):.1f})")

i15, i20 = SIGMAS_I.index(0.15), SIGMAS_I.index(0.20)
p1 = curves[2][i15] >= 90.0 and curves[2][i20] >= 90.0
p2 = (curves[2][i20] - curves[1][i20]) >= 10.0
print("\ncheck (M2 >= 90% at 15% and 20% noise):     ", "PASS" if p1 else "FAIL")
print("check (M2 beats M1 by >= 10 pts at 20%):    ", "PASS" if p2 else "FAIL")
print("OVERALL:", "PASS" if (p1 and p2) else "FAIL")

plt.figure(figsize=(8, 5))
labels = {1: 'intensity only (derotation analog)', 2: 'intensity + Faraday depth', 3: 'depth alone'}
for m, style in [(1, 's--'), (2, 'o-'), (3, '^:')]:
    plt.plot(100*np.array(SIGMAS_I), curves[m], style, label=labels[m])
plt.axhline(50, color='gray', lw=0.8)
plt.xlabel('intensity noise (%)'); plt.ylabel('correct front/back direction (%)')
plt.ylim(40, 105); plt.legend(); plt.grid(True)
out = 'depth_sweep2.png' if OBS_FILE == 'observation.npz' else 'depth_sweep2_decoupled.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"saved {out}")
