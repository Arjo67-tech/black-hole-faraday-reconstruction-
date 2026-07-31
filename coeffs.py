import numpy as np

def transfer_matrix(alpha_I, alpha_Q, alpha_U, alpha_V, rho_Q, rho_U, rho_V):
    K = np.array([
        [alpha_I, alpha_Q, alpha_U, alpha_V],
        [alpha_Q, alpha_I, rho_V, -rho_U],
        [alpha_U, -rho_V, alpha_I, rho_Q],
        [alpha_V, rho_U, -rho_Q, alpha_I]
    ])
    return K

def j_of_s(s):
    if 0.08 <= s <= 0.12:
        return np.array([1.0, 0.7, 0.0, 0.0])
    else:
        return np.zeros(4)

def K_of_s(s, lam2):
    PHI = 10.0
    if 0.3 <= s <= 0.7:
        rho_V = 2 * PHI * (10 / 0.4) * lam2
    else:
        rho_V = 0.0
    alpha_I = 0.0
    alpha_Q = 0.0
    alpha_U = 0.0
    alpha_V = 0.0
    return transfer_matrix(alpha_I, alpha_Q, alpha_U, alpha_V, 0.0, 0.0, rho_V)
