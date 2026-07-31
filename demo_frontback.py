import numpy as np
from integrate import integrate_ray, fit_slope  # Import fit_slope from integrate.py
from coeffs import j_of_s, K_of_s

def run_simulation(s_start, lam2_values):
    s_grid = np.linspace(0, 1, 100)
    chi_values = []
    
    for lam2 in lam2_values:
        S0 = np.array([1.0, 0.0, 0.0, 0.0])
        S = integrate_ray(s_grid, j_of_s, lambda s: K_of_s(s, lam2), S0)
        chi = S[-1, 1] / S[-1, 0]
        chi_values.append(chi)
    
    slope = fit_slope(lam2_values, chi_values)  # Now fit_slope is defined
    return slope

def main():
    lam2_values = [0.0, 0.05, 0.1, 0.15]
    
    # Run A: emitter at s=0.1
    slope_A = run_simulation(0.1, lam2_values)
    
    # Run B: emitter at s=0.9
    slope_B = run_simulation(0.9, lam2_values)
    
    print(f"Slope A: {slope_A}")
    print(f"Slope B: {slope_B}")
    
    if np.isclose(slope_A, 10.0, atol=0.01) and np.isclose(slope_B, 0.0, atol=0.01):
        print("PASS")
    else:
        print("FAIL")
    
    # Run A with intrinsic angle 0 at lam2=0.1
    s_grid = np.linspace(0, 1, 100)
    S0_A = np.array([1.0, 0.0, 0.0, 0.0])
    S_A = integrate_ray(s_grid, j_of_s, lambda s: K_of_s(s, 0.1), S0_A)
    
    # Run B with intrinsic angle 1.0 rad at lam2=0.1
    S0_B = np.array([np.cos(1.0), np.sin(1.0), 0.0, 0.0])
    S_B = integrate_ray(s_grid, j_of_s, lambda s: K_of_s(s, 0.1), S0_B)
    
    print("Final Stokes vector for Run A (intrinsic angle 0):", S_A[-1])
    print("Final Stokes vector for Run B (intrinsic angle 1.0 rad):", S_B[-1])

if __name__ == '__main__':
    main()
