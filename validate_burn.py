import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coeffs import transfer_matrix
from integrate import integrate_ray

PHI = 10.0
p0 = 0.7

def j_of_s(s):
    return np.array([1.0, 0.7, 0.0, 0.0])

def make_K(lam2):
    def K(s):
        return transfer_matrix(0, 0, 0, 0, 0.0, 0.0, 2 * PHI * lam2)
    return K

lam2_values = np.logspace(np.log10(0.001), np.log10(0.5), 60)
s_grid = np.linspace(0, 1, 2000)

p_num, chi_num = [], []
for lam2 in lam2_values:
    S = integrate_ray(s_grid, j_of_s, make_K(lam2), np.zeros(4))
    I, Q, U, V = S[-1]
    p_num.append(np.sqrt(Q**2 + U**2) / I)
    chi_num.append(0.5 * np.arctan2(U, Q))
p_num = np.array(p_num); chi_num = np.array(chi_num)

p_th = p0 * np.abs(np.sin(PHI * lam2_values) / (PHI * lam2_values))
chi_th = 0.5 * PHI * lam2_values

away = p_th >= 0.05
err = np.max(np.abs((p_num[away] - p_th[away]) / p_th[away]))
null_lam2 = lam2_values[np.argmin(p_num)]
i20 = np.argmin(np.abs(lam2_values - np.pi/20))
mask = lam2_values < 0.25
slope = np.polyfit(lam2_values[mask], chi_num[mask], 1)[0]

print(f"Max rel error of p (away from nulls): {err*100:.2f}%   (pass < 0.5%)")
print(f"First null at lam2 = {null_lam2:.4f}   (theory pi/10 = {np.pi/10:.4f}, grid is coarse)")
print(f"p/p0 at lam2=pi/20: {p_num[i20]/p0:.4f}   (theory 0.6366)")
print(f"chi slope pre-null: {slope:.4f}   (expect 5.0, HALF the screen slope)")
print("PASS" if (err < 0.005 and abs(slope - 5.0) < 0.05) else "FAIL")

plt.figure(figsize=(8,5))
plt.plot(lam2_values, p_th, '--', label='Burn 1966 theory')
plt.plot(lam2_values, p_num, 'o', ms=3, label='numeric')
plt.xscale('log'); plt.xlabel('lam2'); plt.ylabel('p'); plt.legend(); plt.grid(True)
plt.savefig('p_vs_lam2.png', dpi=120)
