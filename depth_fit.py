import numpy as np

tab = np.load('obs_table.npz')
obs = np.load('observation.npz')

OMEGA = 1.0/8.0**1.5
DLAM2 = (230.0/213.0)**2 - 1.0
SIGMA_DCHI = np.radians(1.5)
SIGMA_I_FRAC = 0.05
NBOOT = 200

phis_t = tab['phis']
def wrap_diff(d):
    return (d + np.pi/2) % np.pi - np.pi/2

# wraparound-extended table columns for interpolation
def ext(col):
    return np.append(col, col[0])
phis_e = np.append(phis_t, phis_t[0] + 2*np.pi)
I_e = ext(tab['I']); Q230_e = ext(tab['Q230']); U230_e = ext(tab['U230'])
Q213_e = ext(tab['Q213']); U213_e = ext(tab['U213'])

def model_at(s, phi0, t):
    phi = (s*OMEGA*t + phi0) % (2*np.pi)
    I_m = np.interp(phi, phis_e, I_e)
    Q230 = np.interp(phi, phis_e, Q230_e); U230 = np.interp(phi, phis_e, U230_e)
    Q213 = np.interp(phi, phis_e, Q213_e); U213 = np.interp(phi, phis_e, U213_e)
    dchi_m = wrap_diff(0.5*np.arctan2(U213, Q213) - 0.5*np.arctan2(U230, Q230))
    return I_m, dchi_m

t = obs['t']
I_clean = obs['I_clean']; dchi_clean = obs['dchi_clean']
s_true = int(obs['s_true']); phi0_true = float(obs['phi0_true'])
T_ORB = 2*np.pi/OMEGA

phi0_grid = np.linspace(0, 2*np.pi, 128, endpoint=False)
s_grid = [1, -1]

# precompute model curves for every hypothesis (2 x 128)
models = {}
for s in s_grid:
    for phi0 in phi0_grid:
        models[(s, phi0)] = model_at(s, phi0, t)

def true_pos(tk):
    ph = s_true*OMEGA*tk + phi0_true
    return np.array([8*np.cos(ph), 8*np.sin(ph), 0*ph])

def fit_pos_err(s, phi0):
    ph = s*OMEGA*t + phi0
    px, py = 8*np.cos(ph), 8*np.sin(ph)
    ph_t = s_true*OMEGA*t + phi0_true
    tx, ty = 8*np.cos(ph_t), 8*np.sin(ph_t)
    return np.sqrt(np.mean((px-tx)**2 + (py-ty)**2))

def run_fit(I_data, dchi_data, use_dchi):
    best = None
    chi2_by_s = {1: np.inf, -1: np.inf}
    for (s, phi0), (I_m, dchi_m) in models.items():
        F = np.sum(I_data*I_m)/np.sum(I_m**2)
        chi2 = np.sum(((I_data - F*I_m)/(SIGMA_I_FRAC*I_data))**2)
        if use_dchi:
            chi2 += np.sum((wrap_diff(dchi_data - dchi_m)/SIGMA_DCHI)**2)
        if chi2 < chi2_by_s[s]:
            chi2_by_s[s] = chi2
        if best is None or chi2 < best[0]:
            best = (chi2, s, phi0)
    _, s_fit, phi0_fit = best
    correct = (s_fit == s_true)
    poserr = fit_pos_err(s_fit, phi0_fit)
    # direction significance: chi2 gap between the two s hypotheses
    dchi2_dir = abs(chi2_by_s[1] - chi2_by_s[-1])
    # phi0 error (wrap-aware), only meaningful if direction correct
    dphi0 = np.angle(np.exp(1j*(phi0_fit - phi0_true)))
    return correct, poserr, dchi2_dir, np.degrees(abs(dphi0))

rng = np.random.default_rng(7)
results = {1: [], 2: []}
for b in range(NBOOT):
    if b == 0:
        I_d = obs['I_noisy']; dchi_d = obs['dchi_noisy']
    else:
        I_d = I_clean*(1 + SIGMA_I_FRAC*rng.standard_normal(len(t)))
        dchi_d = dchi_clean + np.radians(np.sqrt(2)*1.0)*rng.standard_normal(len(t))
    results[1].append(run_fit(I_d, dchi_d, use_dchi=False))
    results[2].append(run_fit(I_d, dchi_d, use_dchi=True))

print(f"{'':28s} {'Method 1 (I only)':>20s} {'Method 2 (I + dchi)':>20s}")
for name, idx in [("correct direction %", 0), ("median 3D pos err (M)", 1),
                  ("median direction dchi2", 2)]:
    v1 = [r[idx] for r in results[1]]
    v2 = [r[idx] for r in results[2]]
    if idx == 0:
        print(f"{name:28s} {100*np.mean(v1):>19.1f}% {100*np.mean(v2):>19.1f}%")
    else:
        print(f"{name:28s} {np.median(v1):>20.2f} {np.median(v2):>20.2f}")

# phi0 error over correct-direction realizations only
p1 = [r[3] for r in results[1] if r[0]]
p2 = [r[3] for r in results[2] if r[0]]
print(f"{'median phi0 err (deg)*':28s} {np.median(p1) if p1 else float('nan'):>20.2f} {np.median(p2) if p2 else float('nan'):>20.2f}")
print("  (*over correct-direction realizations only)")

m2_correct = np.mean([r[0] for r in results[2]])
m2_poserr = np.median([r[1] for r in results[2]])
pA = m2_correct >= 0.95
pB = m2_poserr < 1.0
print("\ncheck (Method 2 correct >= 95%):     ", "PASS" if pA else "FAIL")
print("check (Method 2 median pos err < 1M):", "PASS" if pB else "FAIL")
print("OVERALL:", "PASS" if (pA and pB) else "FAIL")
