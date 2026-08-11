import sys
import numpy as np

OBS_FILE = sys.argv[1]
TAB = sys.argv[2] if len(sys.argv) > 2 else 'obs_table64.npz'
tab = np.load(TAB); obs = np.load(OBS_FILE)
print(f"table file: {TAB}\ndata file: {OBS_FILE}")

OMEGA = 1.0/8.0**1.5
SIGMA_DCHI = np.radians(1.5)
NBOOT = 200
SIGMAS_I = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

def wrap_diff(d): return (d + np.pi/2) % np.pi - np.pi/2
def ext(c): return np.append(c, c[0])
phis_e = np.append(tab['phis'], tab['phis'][0] + 2*np.pi)
I_e = ext(tab['I'])
Q230_e, U230_e = ext(tab['Q230']), ext(tab['U230'])
Q213_e, U213_e = ext(tab['Q213']), ext(tab['U213'])

t = obs['t']; I_clean = obs['I_clean']; dchi_clean = obs['dchi_clean']
s_true = int(obs['s_true'])

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

def run_fit(I_d, dchi_d, sig_I, use_I, use_dchi):
    best, chi2_s = None, {1: np.inf, -1: np.inf}
    for (s, phi0), (I_m, dchi_m) in models.items():
        chi2 = 0.0
        if use_I:
            F = np.sum(I_d*I_m)/np.sum(I_m**2)
            sigma = sig_I * np.maximum(F*I_m, 1e-3*np.max(np.abs(F*I_m)))
            chi2 += np.sum(((I_d - F*I_m)/sigma)**2)
        if use_dchi:
            # amplitude nuisance: judge direction by SHAPE, not gas normalization
            G = np.sum(dchi_d*dchi_m)/np.sum(dchi_m**2)
            G = max(G, 0.0)
            chi2 += np.sum((wrap_diff(dchi_d - G*dchi_m)/SIGMA_DCHI)**2)
        chi2_s[s] = min(chi2_s[s], chi2)
        if best is None or chi2 < best[0]:
            best = (chi2, s)
    return (best[1] == s_true), abs(chi2_s[1] - chi2_s[-1])

rng = np.random.default_rng(31)
for sig_I in SIGMAS_I:
    ok = {1: 0, 2: 0, 3: 0}; g = {1: [], 2: [], 3: []}
    for b in range(NBOOT):
        I_d = I_clean*(1 + sig_I*rng.standard_normal(len(t)))
        d_d = dchi_clean + np.radians(np.sqrt(2)*1.0)*rng.standard_normal(len(t))
        for m, (uI, ud) in {1:(True,False), 2:(True,True), 3:(False,True)}.items():
            c, gap = run_fit(I_d, d_d, sig_I, uI, ud)
            ok[m] += c; g[m].append(gap)
    print(f"sigma_I={sig_I:.2f}:  M1 {100*ok[1]/NBOOT:5.1f}%  M2 {100*ok[2]/NBOOT:5.1f}%  "
          f"M3 {100*ok[3]/NBOOT:5.1f}%   (median dchi2: {np.median(g[1]):.1f} / {np.median(g[2]):.1f} / {np.median(g[3]):.1f})")
