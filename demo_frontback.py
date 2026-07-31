import numpy as np
from coeffs import transfer_matrix
from integrate import integrate_ray

S0 = np.zeros(4)
s_grid = np.linspace(0, 1, 4000)
RM = 10.0
COL_LO, COL_HI = 0.3, 0.7
WIDTH = COL_HI - COL_LO

def make_j(s_center, chi0=0.0):
    def j(s):
        if s_center - 0.02 <= s <= s_center + 0.02:
            return np.array([1.0, 0.7*np.cos(2*chi0), 0.7*np.sin(2*chi0), 0.0])
        return np.zeros(4)
    return j

def make_K(lam2):
    def K(s):
        rho_V = 2.0 * RM / WIDTH * lam2 if COL_LO <= s <= COL_HI else 0.0
        return transfer_matrix(0, 0, 0, 0, 0.0, 0.0, rho_V)
    return K

def exit_state(s_center, lam2, chi0=0.0):
    S = integrate_ray(s_grid, make_j(s_center, chi0), make_K(lam2), S0)
    I, Q, U, V = S[-1]
    return 0.5*np.arctan2(U, Q), S[-1]

lam2s = np.array([0.0, 0.05, 0.1, 0.15])
chi_A = np.array([exit_state(0.1, l)[0] for l in lam2s])
chi_B = np.array([exit_state(0.9, l)[0] for l in lam2s])

slope_A = np.polyfit(lam2s, chi_A, 1)[0]
slope_B = np.polyfit(lam2s, chi_B, 1)[0]
print(f"slope A (behind column): {slope_A:.4f}   expect 10.0")
print(f"slope B (in front):      {slope_B:.4f}   expect 0.0")
print("PASS" if abs(slope_A-10.0) < 0.1 and abs(slope_B) < 0.1 else "FAIL")

chiA, SA = exit_state(0.1, 0.1, chi0=0.0)
chiB, SB = exit_state(0.9, 0.1, chi0=1.0)
print("\nAt lam2=0.1 alone:")
print("A (behind, intrinsic 0.0):   ", np.round(SA, 6))
print("B (in front, intrinsic 1.0): ", np.round(SB, 6))
print(f"chi_A = {chiA:.4f}, chi_B = {chiB:.4f}  -> identical at one frequency")
print("Only the lam2 sweep separates them.")
