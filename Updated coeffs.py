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
    return np.array([1.0, 0.7, 0.0, 0.0])

def K_of_s(s, lam2):
    PHI = 10.0
    rho_V = 2 * PHI * lam2
    alpha_I = 1.0
    alpha_Q = 0.0
    alpha_U = 0.0
    alpha_V = 0.0
    return transfer_matrix(alpha_I, alpha_Q, alpha_U, alpha_V, 0.0, 0.0, rho_V)
