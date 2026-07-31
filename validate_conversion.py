import numpy as np
from coeffs import transfer_matrix, j_of_s, K_of_s
from integrate import integrate_ray

def main():
    # Parameters
    s_grid = np.linspace(0, 1, 2000)
    S0 = [1, 0, 0.5, 0]
    rho_Q = 2 * np.pi
    alpha_I = 0
    alpha_Q = rho_Q
    alpha_U = 0
    alpha_V = 0

    # Transfer matrix coefficients
    def j_of_s_wrapper(s):
        return j_of_s(s)

    def K_of_s_wrapper(s, lam2):
        return K_of_s(s, lam2)

    # Integration
    S_final = integrate_ray(s_grid, j_of_s_wrapper, K_of_s_wrapper, S0)

    # Check conditions
    U_values = S_final[:, 2]
    V_values = S_final[:, 3]

    # Check if U and V trade sinusoidally with spatial period exactly 1.0
    if not np.allclose(U_values, -V_values[::-1]) or not np.allclose(V_values, -U_values[::-1]):
        print("FAIL: U and V do not trade sinusoidally with spatial period exactly 1.0")
        return

    # Check if U(s=1) back to 0.5 within 0.1%
    if not np.isclose(U_values[-1], 0.5, atol=0.001):
        print("FAIL: U(s=1) is not back to 0.5 within 0.1%")
        return

    # Check if I unchanged
    I_values = S_final[:, 0]
    if not np.allclose(I_values, I_values[0]):
        print("FAIL: I is changed")
        return

    # Check if Q stays 0
    Q_values = S_final[:, 1]
    if not np.allclose(Q_values, 0):
        print("FAIL: Q is not zero")
        return

    print("PASS: All conditions met")

if __name__ == '__main__':
    main()
