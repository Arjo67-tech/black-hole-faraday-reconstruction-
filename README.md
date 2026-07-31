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
