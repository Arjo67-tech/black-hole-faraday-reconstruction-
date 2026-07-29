import numpy as np
from coeffs import transfer_matrix
from integrate import integrate_ray

# Constants
RM_TOTAL = 10.0
S0 = [1, 0, 0, 0]
s_grid_2000 = np.linspace(0, 1, 2000)
s_grid_4000 = np.linspace(0, 1, 4000)
s_grid_1000 = np.linspace(0, 1, 1000)

# Emission region j(s) for s < 0.3
def j_of_s_emission(s):
    if s < 0.3:
        return np.array([1, 0.7, 0, 0])
    else:
        return np.zeros(4)

# Faraday rotation region K(s) for s >= 0.3
def K_of_s_faraday(s, lam2):
    rho_V = 2 * RM_TOTAL / 0.7 * lam2
    if s < 0.3:
        return transfer_matrix(0, 0, 0, 0, 0, 0, 0)
    else:
        return transfer_matrix(0, 0, 0, 0, 0, 0, rho_V)

# Sweep over lam2 values
lam2_values = [0.0, 0.02, 0.05, 0.1, 0.15]
results = []

for lam2 in lam2_values:
    K_of_s = lambda s: K_of_s_faraday(s, lam2)
    S_2000 = integrate_ray(s_grid_2000, j_of_s_emission, K_of_s, S0)
    chi = 0.5 * np.arctan2(S_2000[-1, 2], S_2000[-1, 1])
    if lam2 == 0.0:
        print(f"lam2: {lam2}, chi: {chi}, chi/lam2: None (division by zero)")
    else:
        results.append((lam2, chi, chi / lam2))
        print(f"lam2: {lam2}, chi: {chi}, chi/lam2: {chi / lam2}")

# Check PASS criteria
if len(results) > 0:
    chi_over_lam2_values = [result[2] for result in results]
    if np.allclose(chi_over_lam2_values, RM_TOTAL * s_grid_2000[-1], atol=0.001):
        print("PASS: chi/lam2 is constant across the sweep and equal to 10.0 within 0.1%.")
    else:
        print("FAIL: chi/lam2 is not constant across the sweep or not equal to 10.0 within 0.1%.")

# Convergence check
lam2 = 0.1
K_of_s = lambda s: K_of_s_faraday(s, lam2)
S_4000 = integrate_ray(s_grid_4000, j_of_s_emission, K_of_s, S0)
chi_true = 0.5 * np.arctan2(S_4000[-1, 2], S_4000[-1, 1])

S_2000 = integrate_ray(s_grid_2000, j_of_s_emission, K_of_s, S0)
chi_2000 = 0.5 * np.arctan2(S_2000[-1, 2], S_2000[-1, 1])

S_1000 = integrate_ray(s_grid_1000, j_of_s_emission, K_of_s, S0)
chi_1000 = 0.5 * np.arctan2(S_1000[-1, 2], S_1000[-1, 1])

error_ratio = abs((chi_1000 - chi_true) / (chi_2000 - chi_true))

if error_ratio > 16:
    print("PASS: Convergence check passed, error shrinks by ~16x when steps double.")
else:
    print("FAIL: Convergence check failed, error does not shrink by ~16x when steps double.")
