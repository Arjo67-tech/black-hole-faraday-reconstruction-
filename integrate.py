import numpy as np

def integrate_ray(s_grid, j_of_s, K_of_s, S0):
    def rk4_step(S, s, h, j, K):
        k1 = h * (j(s) - K(s) @ S)
        k2 = h * (j(s + 0.5*h) - K(s + 0.5*h) @ (S + 0.5*k1))
        k3 = h * (j(s + 0.5*h) - K(s + 0.5*h) @ (S + 0.5*k2))
        k4 = h * (j(s + h) - K(s + h) @ (S + k3))
        return S + (k1 + 2*k2 + 2*k3 + k4) / 6

    S = np.zeros((len(s_grid), 4))
    S[0] = S0
    for i in range(1, len(s_grid)):
        h = s_grid[i] - s_grid[i-1]
        S[i] = rk4_step(S[i-1], s_grid[i-1], h, j_of_s, K_of_s)
    
    return S

def fit_slope(lam2_values, chi_values):
    A = np.vstack([lam2_values, np.ones(len(lam2_values))]).T
    m, c = np.linalg.lstsq(A, chi_values, rcond=None)[0]
    return m
