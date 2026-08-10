# bhfara — Phase 1: validated polarized radiative transfer integrator

This module solves the polarized radiative transfer equation dS/ds = j − K·S for the
Stokes vector S = [I, Q, U, V], with the absorption/rotation matrix K in the convention:
row I: [aI, aQ, aU, aV] / row Q: [aQ, aI, rV, −rU] / row U: [aU, −rV, aI, rQ] /
row V: [aV, rU, −rQ, aI]. Fixed-step RK4 (integrate.py), coefficients built by
transfer_matrix() (coeffs.py). All tests are self-contained scripts defining their own
plasma; coeffs.py and integrate.py are the frozen, validated core.

Validation results (all PASS):
- validate_screen.py — external Faraday screen: chi/lam2 = 9.999 across the sweep
  (theory 10.0, within 0.1%). Rotation scales exactly as wavelength squared.
- validate_burn.py — Burn (1966) slab, mixed emission+rotation: max relative error vs
  the exact sinc solution 0.00% away from nulls; p/p0 at pi/20 = 0.6371 (theory 0.6366);
  chi slope pre-null = 5.0000 — exactly HALF the screen slope, because light born at
  depth s only rotates through the remaining (1−s) of the slab.
- validate_conversion.py — Faraday conversion only: full U→V→−U→−V→U cycle with spatial
  period exactly 1.0; U returns to 0.500000; I and Q untouched.
- demo_frontback.py — the project thesis in 1D: identical emitters behind vs in front of
  a rotating column are indistinguishable at any single frequency (final Stokes vectors
  match to 5 decimals), but separate cleanly in a frequency sweep: slope 9.9983 vs 0.0000.

Real units (real_units.py), Sgr A* RM = −5e5 rad/m²: rotation −48.7° at 230 GHz,
−56.8° at 213 GHz, −21.6° at 345 GHz, −348° at 86 GHz (near-full wrap — the n·180°
ambiguity at long wavelengths). Difference between ALMA sidebands 213/229 GHz: −7.65°,
the measurable depth signal this project is built on.

## Phase 2: curved spacetime (all PASS)
- geodesic_bridge over kgeo raytrace_ana: grazing ray min(r)=18.9131 (theory ~18.9), central ray captured at horizon r=2.0000
- shadow edge by bisection: ±5.1960 at both 17 and 60 deg inclination (theory sqrt(27)=5.1962); spin a=0.9 asymmetric: +5.5036 / -4.3033
- Einstein deflection: matches 3rd-order GR series, residuals 0.58% -> 0.01% shrinking exactly as truncation predicts
- first images: blob flux within 0.7% of analytic integral; ring at r=6 shows direct ellipse + secondary image hugging the photon ring
- orbiting hotspot at r=8: face-on peak/trough 1.29 (pure gravitational redshift, g centered on 1/u^t=0.79), edge-on 14.55 (Doppler beaming g^3 ~9x plus lensing), g range 0.478-2.277 straddling 1
- orbital period 142.2 M = 48 min for Sgr A* — GRAVITY observed ~45 min real flare orbits

## Phase 3, Step 5: inference from noisy data (PASS)
Corrupted the simulated Stokes data with realistic ALMA-class noise (1%
amplitude, 1 degree angle), then recovered the hidden flux-weighted gas
column from sideband angle differences alone via column = dchi/dlam2.
Truth landed within the 2-sigma recovery band at 96% of phases (honest
bars: ~95% expected by construction). Bootstrap errors shrink exactly as
sqrt(N) when averaging phases: 0.1665 -> 0.0589 over 1 -> 8 phases.
First blind recovery of a hidden physical quantity from noisy synthetic
measurements — the full project's inference logic demonstrated end to end.

## Phase 4: the depth experiment (COMPLETE)
Question: does Faraday rotation add recoverable depth information to black
hole hotspot reconstruction, beyond what intensity fitting (the constant-
derotation approach of Levis et al. 2024) already provides?

Setup: blob orbiting at r=8M, observer at 20 deg inclination (Sgr A*-like,
where the front/back mirror degeneracy is strongest). Method 1 fits
intensity only; Method 2 adds the 213-229 GHz sideband polarization-angle
difference (the Faraday depth signal); Method 3 uses the depth signal alone.
200 noise realizations per point; observations and fitting tables rendered
on different pixel grids (decoupled) at converged 64-pixel resolution.

Result (consistent across three independent starting phases 0.7/2.0/4.5 rad):
- Method 1 collapses from ~82% correct front/back direction at 5% intensity
  noise to chance (~50%) by 20-30% noise -- discrimination dies as 1/sigma^2,
  measured quantitatively.
- Method 2 holds 85-94% across the entire 5-50% noise sweep. At high noise
  its discrimination (median delta-chi2 ~5) is supplied essentially entirely
  by the Faraday channel (Method 1's falls to ~0.3-0.6).
- Method 3 alone sits at chance: the depth curve is nearly direction-blind
  without intensity pinning the orbital phase.
Conclusion: the two channels are complementary -- intensity anchors WHERE
the blob is in its cycle; the chromatic Faraday signal determines WHICH WAY
it moves through the magnetized gas. Under realistic source variability
(>=20%, cf. Wielgus et al. 2022 light curves), the Faraday channel supplies
all front/back depth information.

Methodological note: an earlier version of this experiment gave inflated
results (100%/91.5%) traced to shared pixelation between data and model on
an under-resolved 32-pixel grid (blob ~1 pixel wide). Caught by a grid-
decoupling test, resolved by convergence testing (32->48->64; peak depth
signal converged 10.45 -> 8.86 -> 8.87 deg; residual table change 48->64
~10% of signal range, documented as the precision limit). A silent
hardcoded-filename bug in the sweep script was also caught by making
scripts print their input files. Final numbers are grid-decoupled and
converged.
