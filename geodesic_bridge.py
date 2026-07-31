import numpy as np

def trace_ray(a, th_o, alpha, beta, n_points=500):
    # Placeholder for kgeo's analytic raytracer function
    def kgeo_analytic_raytracer(a, th_o, alpha, beta, n_points):
        # This is a placeholder implementation. Replace with actual kgeo API call.
        r = np.linspace(10, 2, n_points)  # Example r values from large radius to horizon
        th = np.full(n_points, th_o)
        ph = np.linspace(0, 2 * np.pi, n_points)
        captured = (r[-1] < 3)  # Example condition for capture at the horizon
        return r, th, ph, captured

    r, th, ph, captured = kgeo_analytic_raytracer(a, th_o, alpha, beta, n_points)
    return {'r': r, 'th': th, 'ph': ph, 'captured': captured}

if __name__ == '__main__':
    a = 0
    th_o = np.radians(17)

    # Ray 1: alpha=20, beta=0 (must escape)
    result1 = trace_ray(a, th_o, 20, 0)
    min_r1 = np.min(result1['r'])
    print(f"Ray 1 - Min r: {min_r1}")
    if 18 <= min_r1 <= 19:
        print("PASS")
    else:
        print("FAIL")

    # Ray 2: alpha=0.5, beta=0 (must be captured)
    result2 = trace_ray(a, th_o, 0.5, 0)
    final_r2 = result2['r'][-1]
    print(f"Ray 2 - Final r: {final_r2}")
    if np.isclose(final_r2, 2):
        print("PASS")
    else:
        print("FAIL")
