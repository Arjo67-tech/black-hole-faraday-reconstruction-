import numpy as np
import matplotlib.pyplot as plt

from coeffs import j_of_s, K_of_s
from integrate import integrate_ray

def main():
    PHI = 10.0
    p0 = 0.7
    lam2_values = np.logspace(-3, -1, 60)
    
    results = []
    
    for lam2 in lam2_values:
        s_grid = np.linspace(0, 1, 100)
        S0 = np.array([1.0, 0.7, 0.0, 0.0])
        
        K_of_s_func = lambda s: K_of_s(s, lam2)
        j_of_s_func = j_of_s
        
        S = integrate_ray(s_grid, j_of_s_func, K_of_s_func, S0)
        
        I, Q, U, V = S[-1]
        p_numeric = np.sqrt(Q**2 + U**2) / I
        chi_numeric = 0.5 * np.arctan2(U, Q)
        
        p_theory = np.abs(np.sin(PHI * lam2) / (PHI * lam2))
        chi_theory = 0.5 * PHI * lam2
        
        results.append((lam2, p_numeric, p_theory, chi_numeric, chi_theory))
    
    # Print table
    print("lam2\t\tp_numeric\tp_theory\tchi_numeric\tchi_theory")
    for result in results:
        print(f"{result[0]:.4f}\t{result[1]:.6f}\t{result[2]:.6f}\t{result[3]:.6f}\t{result[4]:.6f}")
    
    # Check PASS criteria
    p_values = np.array([res[1] for res in results])
    p_theory_values = np.array([res[2] for res in results])
    chi_values = np.array([res[3] for res in results])
    chi_theory_values = np.array([res[4] for res in results])
    
    max_rel_error_p = np.max(np.abs((p_values - p_theory_values) / p_theory_values))
    first_null_index = np.argmax(p_values < 0.5 * p0)
    lam2_first_null = lam2_values[first_null_index]
    p_at_pi_over_20 = p_values[np.argmin(np.abs(lam2_values - np.pi / 20))]
    chi_slope = (chi_theory_values[-1] - chi_theory_values[0]) / (lam2_values[-1] - lam2_values[0])
    
    print(f"Max relative error of p vs theory: {max_rel_error_p * 100:.2f}%")
    print(f"First depolarization null at lam2 = {lam2_first_null:.4f}")
    print(f"At lam2 = pi/20: p/p0 = {p_at_pi_over_20 / p0:.6f}")
    print(f"Chi slope: {chi_slope:.2f} (expected 5.0)")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lam2_values, p_values, label='Numeric')
    plt.plot(lam2_values, p_theory_values, label='Theory', linestyle='--')
    plt.xlabel('lam2')
    plt.ylabel('p')
    plt.legend()
    plt.title('Polarization Fraction vs lam2')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('p_vs_lam2.png')
    plt.show()

if __name__ == '__main__':
    main()
