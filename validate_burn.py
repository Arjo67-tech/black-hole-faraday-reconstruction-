import numpy as np
import matplotlib.pyplot as plt

from coeffs import j_of_s, K_of_s
from integrate import integrate_ray

def main():
    PHI = 10.0
    p0 = 0.7
    lam2_values = np.logspace(np.log10(0.001), np.log10(0.5), 60)
    
    results = []
    
    for lam2 in lam2_values:
        s_grid = np.linspace(0, 1, 100)
        S0 = np.array([0.0, 0.0, 0.0, 0.0])
        
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
    chi_numeric_values = np.array([res[3] for res in results])
    chi_theory_values = np.array([res[4] for res in results])
    
    # Exclude points within 20% of the null (where p_theory < 0.05)
    non_null_indices = p_theory_values >= 0.05
    max_rel_error_p_away_from_nulls = np.max(np.abs((p_values[non_null_indices] - p_theory_values[non_null_indices]) / p_theory_values[non_null_indices]))
    
    # Find the actual null by locating where p_numeric is at its minimum
    first_null_index = np.argmin(p_values)
    lam2_first_null = lam2_values[first_null_index]
    
    # Compute chi slope from a linear fit of chi_numeric_values vs lam2_values
    chi_slope, _ = np.polyfit(lam2_values, chi_numeric_values, 1)
    
    p_at_pi_over_20 = p_values[np.argmin(np.abs(lam2_values - np.pi / 20))]
    
    print(f"Max relative error of p vs theory (away from nulls): {max_rel_error_p_away_from_nulls * 100:.2f}%")
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
