import numpy as np
from kgeo.kerr_raytracing_ana import raytrace_ana

def trace_ray(a, th_o, alpha, beta, n_points=500, r_o=1000.0):
    """
    Trace one light ray through Kerr spacetime with kgeo's analytic raytracer.
    a: spin (0 <= |a| < 1); th_o: observer inclination in radians
    (must not be exactly 0 or pi/2); alpha, beta: screen coords in M.
    Returns dict: 'r','th','ph' arrays along the path, plus 'captured'.
    """
    geos = raytrace_ana(a=float(a),
                        observer_coords=[0.0, float(r_o), float(th_o), 0.0],
                        image_coords=[np.array([float(alpha)]),
                                      np.array([float(beta)])],
                        ngeo=n_points,
                        do_phi_and_t=True,
                        savedata=False, plotdata=False)

    gc = getattr(geos, 'geo_coords', None)
    if gc is None:
        raise AttributeError(f"Geodesics has no geo_coords; attrs: {dir(geos)}")

    r  = np.squeeze(np.array(gc[1]))
    th = np.squeeze(np.array(gc[2]))
    ph = np.squeeze(np.array(gc[3]))

    r_plus = 1.0 + np.sqrt(1.0 - float(a)**2)   # event horizon radius
    captured = bool(r[-1] < 3.5 or np.min(r) < r_plus + 0.05)

    return {'r': r, 'th': th, 'ph': ph, 'captured': captured}

if __name__ == '__main__':
    th_o = np.radians(17.0)

    print("Ray 1: alpha=20, beta=0 (should escape)")
    ray1 = trace_ray(0.0, th_o, 20.0, 0.0)
    minr1 = np.min(ray1['r'])
    print(f"  min(r) = {minr1:.4f}   (expect ~18.9)")
    print(f"  captured = {ray1['captured']}")
    pass1 = (17.0 < minr1 < 20.0) and not ray1['captured']

    print("Ray 2: alpha=0.5, beta=0 (should be captured)")
    ray2 = trace_ray(0.0, th_o, 0.5, 0.0)
    print(f"  final r = {ray2['r'][-1]:.4f}   (expect near horizon r=2)")
    print(f"  captured = {ray2['captured']}")
    pass2 = ray2['captured'] and ray2['r'][-1] < 3.0

    print("PASS" if (pass1 and pass2) else "FAIL")
