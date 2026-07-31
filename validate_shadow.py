import numpy as np
from contextlib import redirect_stdout
import os
from geodesic_bridge import trace_ray

def find_boundary(a, th_o, beta, alpha_min, alpha_max, tolerance=1e-3):
    def is_captured(alpha):
        with redirect_stdout(open(os.devnull, 'w')):
            ray = trace_ray(a, th_o, alpha, beta)
        return ray['captured']

    while abs(alpha_max - alpha_min) > tolerance:
        alpha_mid = (alpha_min + alpha_max) / 2
        if is_captured(alpha_mid):
            alpha_max = alpha_mid
        else:
            alpha_min = alpha_mid

    return (alpha_min + alpha_max) / 2

def main():
    th_o_17 = np.radians(17.0)
    th_o_60 = np.radians(60.0)
    a_0 = 0.0
    a_9 = 0.9
    beta = 0.0

    # Test at th_o = 17 degrees
    boundary_17_pos = find_boundary(a_0, th_o_17, beta, 1.0, 10.0)
    boundary_17_neg = find_boundary(a_0, th_o_17, beta, -10.0, -1.0)

    # Test at th_o = 60 degrees
    boundary_60_pos = find_boundary(a_0, th_o_60, beta, 1.0, 10.0)
    boundary_60_neg = find_boundary(a_0, th_o_60, beta, -10.0, -1.0)

    # Test at a = 0.9
    boundary_9_pos = find_boundary(a_9, th_o_17, beta, 1.0, 10.0)
    boundary_9_neg = find_boundary(a_9, th_o_17, beta, -10.0, -1.0)

    # Print results
    print(f"Boundary at th_o = 17 degrees (positive): {boundary_17_pos:.4f}")
    print(f"Boundary at th_o = 17 degrees (negative): {boundary_17_neg:.4f}")
    print(f"Boundary at th_o = 60 degrees (positive): {boundary_60_pos:.4f}")
    print(f"Boundary at th_o = 60 degrees (negative): {boundary_60_neg:.4f}")
    print(f"Boundary at a = 0.9 (positive): {boundary_9_pos:.4f}")
    print(f"Boundary at a = 0.9 (negative): {boundary_9_neg:.4f}")

    # Check PASS criteria
    pass_criteria_17 = abs(boundary_17_pos - np.sqrt(27)) < 0.002 and abs(boundary_17_neg + np.sqrt(27)) < 0.002
    pass_criteria_60 = abs(boundary_60_pos - np.sqrt(27)) < 0.002 and abs(boundary_60_neg + np.sqrt(27)) < 0.002
    pass_criteria_9 = (boundary_9_pos < np.sqrt(27) and boundary_9_neg > np.sqrt(27))

    overall_pass = pass_criteria_17 and pass_criteria_60 and pass_criteria_9

    print("PASS" if overall_pass else "FAIL")

if __name__ == '__main__':
    main()
