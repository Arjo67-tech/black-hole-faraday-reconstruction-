import numpy as np
from coeffs import transfer_matrix
from integrate import integrate_ray

# Faraday conversion test: no emission, no absorption, no rotation.
# Only rho_Q = 2*pi. Light starts as [1, 0, 0.5, 0] and U <-> V
# should trade sinusoidally with spatial period exactly 1.0.

s_grid = np.linspace(0, 1, 2000)
S0 = np.array([1.0, 0.0, 0.5, 0.0])
rho_Q = 2 * np.pi

def j_zero(s):
    return np.zeros(4)

def K_conv(s):
    return transfer_matrix(0, 0, 0, 0, rho_Q, 0, 0)

S = integrate_ray(s_grid, j_zero, K_conv, S0)

I, Q, U, V = S[:, 0], S[:, 1], S[:, 2], S[:, 3]

# Theory: U(s) = 0.5*cos(2*pi*s), V(s) = -0.5*sin(2*pi*s) (sign depends
# on convention; magnitude and period are what we check hard).
U_theory = 0.5 * np.cos(2 * np.pi * s_grid)

print(f"U at s=0:    {U[0]:.6f}   (expect 0.5)")
print(f"U at s=0.25: {U[np.argmin(np.abs(s_grid-0.25))]:.6f}   (expect ~0, all in V)")
print(f"V at s=0.25: {V[np.argmin(np.abs(s_grid-0.25))]:.6f}   (expect ±0.5)")
print(f"U at s=0.5:  {U[np.argmin(np.abs(s_grid-0.5))]:.6f}   (expect -0.5)")
print(f"U at s=1:    {U[-1]:.6f}   (expect 0.5, full cycle)")

ok = True
if abs(U[-1] - 0.5) > 0.0005:
    print("FAIL: U(s=1) not back to 0.5 within 0.1%"); ok = False
if np.max(np.abs(U - U_theory)) > 0.001:
    print("FAIL: U(s) does not follow 0.5*cos(2*pi*s)"); ok = False
if np.max(np.abs(I - 1.0)) > 1e-6:
    print("FAIL: I changed"); ok = False
if np.max(np.abs(Q)) > 1e-6:
    print("FAIL: Q nonzero"); ok = False
if abs(np.max(np.abs(V)) - 0.5) > 0.001:
    print("FAIL: V amplitude not 0.5"); ok = False
if ok:
    print("PASS: full U->V->-U->-V->U cycle, period 1.0, I and Q untouched")
