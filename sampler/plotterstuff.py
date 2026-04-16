import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Example of a nonconvex but beta-weakly convex function h
# Choose h(u) = 0.3 u^4 - u^2 + 0.5 u
#
# h''(u) = 3.6 u^2 - 2  >= -2 for all u
# so h is beta-weakly convex with beta = 2.
# Then h_tilde(u) := h(u) + (beta/2) u^2 is convex:
# h_tilde(u) = 0.3 u^4 + 0.5 u, and h_tilde''(u) = 3.6 u^2 >= 0.
# -----------------------------
beta = 2.0

def h(u):
    return 0.3*u**4 - u**2 + 0.5*u

def hprime(u):
    return 1.2*u**3 - 2.0*u + 0.5

def h_tilde(u):
    return h(u) + 0.5*beta*u**2  # convexified version

def h_tilde_prime(u):
    return hprime(u) + beta*u

# Build h from supporting parabolas:
# For convex h_tilde, we have
#   h_tilde(u) = sup_v [ h_tilde(v) + <h_tilde'(v), u - v> ]  (tangent lines)
# Therefore
#   h(u) = sup_v [ h_tilde(v) + h_tilde'(v)(u-v) ] - (beta/2)u^2
# The inside is a concave parabola in u for each v.
def supporting_parabola(u, v):
    # tangent line to h_tilde at v, then subtract (beta/2)u^2
    L = h_tilde(v) + h_tilde_prime(v) * (u - v)
    return L - 0.5*beta*u**2

# Grid for plotting u
u = np.linspace(-2.5, 2.5, 2000)

# Sample many support points v to approximate the supremum
v_samples = np.linspace(-2.5, 2.5, 401)

# Compute envelope q(u) = max_v supporting_parabola(u, v)
Q = np.full_like(u, -np.inf, dtype=float)
# (vectorized over u; loop over v is fine for 401 points)
for v in v_samples:
    Q = np.maximum(Q, supporting_parabola(u, v))

# Optional: show a handful of individual parabolas for illustration
v_show = np.linspace(-2.0, 2.0, 15)

# Diagnostics: envelope should match h closely if v_samples is dense
max_err = np.max(np.abs(h(u) - Q))
print("max |h(u) - envelope(u)| over grid =", max_err)

plt.figure(figsize=(9,5))
plt.plot(u, h(u), linewidth=2, label="h(u) (nonconvex, beta-weakly convex)")
plt.plot(u, Q, linewidth=2, linestyle="--", label="max of supporting parabolas (finite sup)")

for v in v_show:
    plt.plot(u, supporting_parabola(u, v), linewidth=1, alpha=0.5)

plt.title("Building a weakly-convex h(u) as the supremum of supporting parabolas")
plt.xlabel("u")
plt.ylabel("value")
plt.grid(True, alpha=0.25)
plt.legend()
plt.show()