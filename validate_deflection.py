import numpy as np
from contextlib import redirect_stdout
import os
from geodesic_bridge import trace_ray

def calculate_deflection(alpha):
    with redirect_stdout(open(os.devnull, 'w')):
        ray = trace_ray(a=0.0, th_o=np.radians(90.001), alpha=alpha, beta=0.0)
    
    ph_unwrapped = np.unwrap(ray['ph'])
    deflection_measured = ph_unwrapped[-1] - ph_unwrapped[0] - np.pi
    deflection_theory = 4 / alpha + (15 * np.pi / 4) / alpha**2
    deflection_4_over_b = 4 / alpha
    
    percent_difference = ((deflection_measured - deflection_theory) / deflection_theory) * 100
    
    return {
        'alpha': alpha,
        'measured_deflection': deflection_measured,
        'theory': deflection_theory,
        '4_over_b': deflection_4_over_b,
        'percent_difference': percent_difference
    }

def main():
    alphas = [20.0, 30.0, 50.0, 100.0]
    results = []

    for alpha in alphas:
        result = calculate_deflection(alpha)
        results.append(result)

    print(f"{'b':<10} {'measured deflection':<20} {'theory':<20} {'4/b alone':<20} {'percent difference':<20}")
    for result in results:
        print(f"{result['alpha']:<10.2f} {result['measured_deflection']:<20.6f} {result['theory']:<20.6f} {result['4_over_b']:<20.6f} {result['percent_difference']:<20.2f}%")

    overall_pass = all(abs(result['percent_difference']) < 3 for result in results)
    print("PASS" if overall_pass else "FAIL")

if __name__ == '__main__':
    main()
