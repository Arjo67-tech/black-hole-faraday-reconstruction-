import sys
import numpy as np

OBS_FILE = sys.argv[1] if len(sys.argv) > 1 else 'observation64.npz'
TAB = sys.argv[2] if len(sys.argv) > 2 else 'obs_table64.npz'
tab = np.load(TAB); obs = np.load(OBS_FILE)
print(f"table file: {TAB}\ndata file: {OBS_FILE}")

OMEGA = 1.0/8.0**1.5
SIGMA_DCHI = np.radians(1.5)
NBOOT = 200
SIGMAS_I = [0.10, 0.20, 0.30]
NWIN = 14   # contiguous phases used, out of 24

def wrap_diff(d): return (d + np.pi/2) % np.pi - np.pi/2
def ext(c): return np.append(c, c[0])
phis_e = np.append(tab['phis'], tab['phis'][0] + 2*np.pi)
I_e = ext(tab['I'])
Q230_e, U230_e = ext(tab['Q230']), ext(tab['U230'])
Q213_e, U213_e = ext(tab['Q213']), ext(tab['U213'])

t_all = obs['t']; I_all = obs['I_clean']; d_all = obs['dchi_clean']
s_true = int(obs['s_true'])
N = len(t_all)

def model_curves(t):
    models = {}
    for s in [1, -1]:
        for phi0 in np.linspace(0, 2*np.pi, 128, endpoint=False):
            phi = (s*OMEGA*t + phi0) % (2*np.pi)
            I_m = np.interp(phi, phis_e, I_e)
            dchi_m = wrap_diff(0.5*np.arctan2(np.interp(phi, phis_e, U213_e),
                                              np.interp(phi, phis_e, Q213_e))
                             - 0.5*np.arctan2(np.interp(phi, phis_e, U230_e),
                                              np.interp(phi, phis_e, Q230_e)))
            models[(s, phi0)] = (I_m, dchi_m)
    return models

def run_fit(models, I_d, dchi_d, sig_I, use_I, use_dchi):
    best, chi2_s = None, {1: np.inf, -1: np.inf}
    for (s, phi0), (I_m, dchi_m) in models.items():
        chi2 = 0.0
        if use_I:
            F = np.sum(I_d*I_m)/np.sum(I_m**2)
            sigma = sig_I * np.maximum(F*I_m, 1e-3*np.max(np.abs(F*I_m)))
            chi2 += np.sum(((I_d - F*I_m)/sigma)**2)
        if use_dchi:
            chi2 += np.sum((wrap_diff(dchi_d - dchi_m)/SIGMA_DCHI)**2)
        chi2_s[s] = min(chi2_s[s], chi2)
        if best is None or chi2 < best[0]:
            best = (chi2, s)
    return best[1] == s_true

rng = np.random.default_rng(23)
print(f"\ncontiguous window: {NWIN} of {N} phases")
summary = {}
for w0 in [0, 8, 16]:
    idx = (w0 + np.arange(NWIN)) % N
    t = t_all[idx]; I_c = I_all[idx]; d_c = d_all[idx]
    models = model_curves(t)
    print(f"\nwindow start index {w0}:")
    for sig_I in SIGMAS_I:
        ok = {1: 0, 2: 0, 3: 0}
        for b in range(NBOOT):
            I_d = I_c*(1 + sig_I*rng.standard_normal(NWIN))
            d_d = d_c + np.radians(np.sqrt(2)*1.0)*rng.standard_normal(NWIN)
            for m, (uI, ud) in {1:(True,False), 2:(True,True), 3:(False,True)}.items():
                ok[m] += run_fit(models, I_d, d_d, sig_I, uI, ud)
        print(f"  sigma_I={sig_I:.2f}:  M1 {100*ok[1]/NBOOT:5.1f}%  M2 {100*ok[2]/NBOOT:5.1f}%  M3 {100*ok[3]/NBOOT:5.1f}%")
        summary[(w0, sig_I)] = (100*ok[1]/NBOOT, 100*ok[2]/NBOOT)

m2_20 = [summary[(w, 0.20)][1] for w in [0, 8, 16]]
gap_20 = np.mean([summary[(w, 0.20)][1] - summary[(w, 0.20)][0] for w in [0, 8, 16]])
p1 = min(m2_20) >= 70.0
p2 = gap_20 >= 10.0
print(f"\nM2 at 20% noise across windows: {[f'{v:.1f}' for v in m2_20]}")
print(f"mean M2-M1 gap at 20% noise: {gap_20:.1f} pts")
print("check (M2 >= 70% in every window at 20%):", "PASS" if p1 else "FAIL")
print("check (mean gap >= 10 pts at 20%):      ", "PASS" if p2 else "FAIL")
print("OVERALL:", "PASS" if (p1 and p2) else "FAIL")
