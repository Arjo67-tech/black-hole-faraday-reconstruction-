import numpy as np

def transfer_matrix(alpha_I, alpha_Q, alpha_U, alpha_V, rho_Q, rho_U, rho_V):
    K = np.array([
        [alpha_I, alpha_Q, alpha_U, alpha_V],
        [alpha_Q, alpha_I, rho_V, -rho_U],
        [alpha_U, -rho_V, alpha_I, rho_Q],
        [alpha_V, rho_U, -rho_Q, alpha_I]
    ])
    return K
