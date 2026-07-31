import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from geodesic_bridge import trace_ray
from integrate import integrate_ray

DEVNULL = open(os.devnull, 'w')

def render(scene_emissivity, a, th_o, extent, npix):
    """Render a scene by tracing rays and integrating along paths."""
    t0 = time.time()
    alphas = np.linspace(-extent, extent, npix)
    betas = np.linspace(-extent, extent, npix)
    image = np.zeros((npix, npix))
    
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            with redirect_stdout(DEVNULL):
                ray = trace_ray(a, th_o, alpha, beta)
            
            r = ray['r']
            th = ray['th']
            ph = ray['ph']
            
            # Build path length array
            dr = np.diff(r)
            dth = np.diff(th)
            dph = np.diff(ph)
            dl = np.sqrt(dr**2 + (r[:-1]*dth)**2 + (r[:-1]*np.sin(th[:-1])*dph)**2)
            s = np.concatenate(([0], np.cumsum(dl)))
            
            # Define j_of_s: emissivity along the path
            def j_of_s(s_val):
                idx = np.argmin(np.abs(s - s_val))
                r_i, th_i, ph_i = r[idx], th[idx], ph[idx]
                emis = scene_emissivity(r_i, th_i, ph_i)
                return np.array([emis, 0.0, 0.0, 0.0])
            
            def K_of_s(s_val):
                return np.zeros((4, 4))
            
            S = integrate_ray(s, j_of_s, K_of_s, np.zeros(4))
            image[i, j] = S[-1, 0]
    
    t1 = time.time()
    print(f"  Render time: {t1-t0:.1f} s")
    return image, (t1-t0)

# Scene A: Gaussian blob
print("Scene A: Gaussian blob at r=30")
def blob_emissivity(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return np.exp(-((x-30)**2 + y**2 + z**2) / (2*1.5**2))

image_A, t_A = render(blob_emissivity, 0.0, np.radians(5), 40, 64)
pix_area = (80.0 / 64)**2
flux_curved = np.sum(image_A) * pix_area

# Flat-space comparison
print("  Computing flat-space baseline...")
flux_flat = 0.0
for alpha in np.linspace(-40, 40, 64):
    for beta in np.linspace(-40, 40, 64):
        # Straight line along z from -100 to 100 through this pixel
        s_vals = np.linspace(-100, 100, 400)
        s_contrib = 0.0
        for s in s_vals:
            x, y, z = alpha, beta, s
            emis = np.exp(-((x-30)**2 + y**2 + z**2) / (2*1.5**2))
            s_contrib += emis * (200.0 / 400)
        flux_flat += s_contrib * pix_area

pct_diff = 100.0 * (flux_curved - flux_flat) / flux_flat
print(f"  Curved flux: {flux_curved:.4f}")
print(f"  Flat flux:   {flux_flat:.4f}")
print(f"  Difference:  {pct_diff:.2f}%")
p_A = abs(pct_diff) < 5.0
print(f"  PASS" if p_A else "  FAIL")

# Scene B: Glowing ring
print("\nScene B: Glowing ring at r=6")
def ring_emissivity(r, th, ph):
    z = r * np.cos(th)
    return np.exp(-(r - 6)**2 / (2*0.3**2)) * np.exp(-(z**2) / (2*0.3**2))

image_B, t_B = render(ring_emissivity, 0.0, np.radians(60), 12, 64)

plt.figure(figsize=(8, 8))
extent_pix = 12
plt.imshow(image_B, origin='lower', extent=[-extent_pix, extent_pix, -extent_pix, extent_pix], cmap='hot')
plt.colorbar(label='Brightness')
plt.xlabel('alpha (M)')
plt.ylabel('beta (M)')
plt.title('Glowing ring at r=6M, th_o=60 deg')
plt.savefig('render_ring.png', dpi=100, bbox_inches='tight')
print(f"  Saved render_ring.png")

print(f"\nTotal time: {t_A + t_B:.1f} s")
print(f"Overall: {'PASS' if p_A else 'FAIL'}")
