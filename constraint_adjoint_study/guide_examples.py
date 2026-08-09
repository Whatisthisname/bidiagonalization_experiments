#!/usr/bin/env python
"""Verifies every number in GUIDE_gauge_freedom.md.

Example A: redundant 3-row constraint set, 1-dim multiplier gauge family,
           gradient = 3 for every gauge (t = 0, minimal-norm, mu2 = 0).
Example B: absurd gauge t = 1e8 destroys the gradient in float32 (0.0
           instead of 3), matching the eps*|mu|/|grad| law.
Example C: two equivalent SQUARE formulations; the ill-mixed one has a
           unique but huge multiplier and also returns 0.0 in float32.
Example D: reprojection in 2D: an unstable iteration (eigenvalues 3 and
           1/3) with a pinned identity l.w = 0; naive float32 dies in ~8
           steps (error ~ eps*9^n), re-enforcing the pin each step keeps
           the error at the precision floor.
Exercises: verifies the solutions to exercises 1-12 (section 15 of the
           guide) and the section-14 dimension counts.
"""

import numpy as np


def gradient(J_theta, mu, dtype=np.float64):
    """Assembly step (G): grad = J_theta^T mu, in the given precision."""
    Jt = np.asarray(J_theta, dtype)
    m = np.asarray(mu, dtype)
    return float(Jt @ m)


def example_A():
    print("=== Example A: redundant set, gauge family, invariant gradient ===")
    # rows: c1 = x1 - th, c2 = x2 - 2 th, c3 = 2 x1 - x2 (redundant)
    J_x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    J_th = np.array([-1.0, -2.0, 0.0])
    g_x = np.array([1.0, 1.0])  # grad_x of rho = x1 + x2

    def mu_of(t):
        return np.array([-1.0 - 2.0 * t, -1.0 + t, t])

    nu = np.array([-2.0, 1.0, 1.0])  # claimed null vector
    print(f"|J_x^T nu| = {np.linalg.norm(J_x.T @ nu):.1e} (null vector check)")
    t_ls = -1.0 / 6.0  # minimal-norm parameter
    for name, t in [("reduced (t=0)", 0.0), ("min-norm (t=-1/6)", t_ls), ("pin mu2=0 (t=1)", 1.0)]:
        mu = mu_of(t)
        res = np.linalg.norm(J_x.T @ mu + g_x)
        print(
            f"  {name:>18}: mu = {np.array2string(mu, precision=3)}, "
            f"|J_x^T mu + g_x| = {res:.1e}, grad = {gradient(J_th, mu):.15f}"
        )
    # minimal-norm check: mu(t_ls) orthogonal to nu
    print(f"  min-norm orthogonality mu.nu = {mu_of(t_ls) @ nu:.1e}")


def example_B():
    print("\n=== Example B: absurd gauge t = 1e8, float32 assembly ===")
    # rows: c1, c2, c3' = c1 + c2 = x1 + x2 - 3 th
    J_th = np.array([-1.0, -2.0, -3.0])
    t = 1e8
    mu = np.array([-1.0 - t, -1.0 - t, t])
    g64 = gradient(J_th, mu, np.float64)
    g32 = gradient(J_th, mu, np.float32)
    eps32 = float(np.finfo(np.float32).eps)
    floor = eps32 * np.abs(mu).max() / 3.0
    print(f"  float64 gradient = {g64:.10f}   (exact answer: 3)")
    print(f"  float32 gradient = {g32:.10f}")
    print(f"  law: eps32 * |mu|_inf / |grad| = {floor:.1f} " f"(>= 1 means total loss, as observed)")


def example_C():
    print("\n=== Example C: equivalent square formulations F1 vs F2 ===")
    g_x = np.array([1.0, 1.0])
    # F1: {x1 - th, x2 - 2 th}
    J_x1 = np.eye(2)
    J_th1 = np.array([-1.0, -2.0])
    mu1 = np.linalg.solve(J_x1.T, -g_x)
    # F2: {x1 - th, 1e8 (x1 - th) + (x2 - 2 th)}  (same manifold)
    m = 1e8
    J_x2 = np.array([[1.0, 0.0], [m, 1.0]])
    J_th2 = np.array([-1.0, -(m + 2.0)])
    mu2 = np.linalg.solve(J_x2.T, -g_x)
    for name, Jt, mu in [("F1", J_th1, mu1), ("F2", J_th2, mu2)]:
        g64 = gradient(Jt, mu, np.float64)
        g32 = gradient(Jt, mu, np.float32)
        print(f"  {name}: mu = {np.array2string(mu, precision=3)}, " f"grad64 = {g64:.10f}, grad32 = {g32:.10f}")
    print("  -> unique multipliers, same exact gradient; the ill-mixed " "formulation still dies in float32.")


# Example D data: G with eigenvalues 3 (unstable, u) and 1/3 (stable, v);
# left eigenvector l for eigenvalue 3; l.v = 0, l.u = 8; pinned identity
# l.w_n = 0 along the exact trajectory w_n = 3^{-n} v.
G_D = np.array([[3.0, 1.0], [0.0, 1.0 / 3.0]])
u_D = np.array([1.0, 0.0])
v_D = np.array([3.0, -8.0])
l_D = np.array([8.0, 3.0])


def example_D():
    print("\n=== Example D: reprojection in two dimensions, float32 ===")
    G32 = G_D.astype(np.float32)
    l32, u32 = l_D.astype(np.float32), u_D.astype(np.float32)
    print("  step | naive rel err | reprojected rel err")
    for n_steps in [5, 7, 10, 20, 40, 60]:
        w_true = (3.0**-n_steps) * v_D
        w_naive = v_D.astype(np.float32)
        w_rep = v_D.astype(np.float32)
        for _ in range(n_steps):
            w_naive = G32 @ w_naive
            w_rep = G32 @ w_rep
            w_rep = w_rep - ((l32 @ w_rep) / np.float32(8.0)) * u32  # reprojection
        e_naive = np.linalg.norm(w_naive.astype(np.float64) - w_true) / np.linalg.norm(w_true)
        e_rep = np.linalg.norm(w_rep.astype(np.float64) - w_true) / np.linalg.norm(w_true)
        print(f"  {n_steps:4d} | {e_naive:13.3e} | {e_rep:.3e}")
    print("  -> naive: error ~ eps32 * 9^n, dead by step ~8-10; reprojected: precision floor.")


def exercises():
    print("\n=== Exercises: verifying the solutions in guide section 15 ===")

    # Exercise 1: (a, b) = (7 + t, t); a - b = 7 for all t; example gauges.
    for t in [0.0, -3.5, 12.0]:
        a, b = 7.0 + t, t
        assert a - b == 7.0
    print("  Ex1: (7+t, t) gives a - b = 7 for t in {0, -3.5, 12}   OK")

    # Exercise 2: M = [1 1 1], b = 2.
    sol_a = np.array([2.0, 0.0, 0.0])
    sol_b = np.array([0.0, 2.0, 0.0])
    assert sol_a.sum() == 2.0 and sol_b.sum() == 2.0  # both are solutions
    w_good = np.array([1.0, 1.0, 1.0])
    w_bad = np.array([1.0, 2.0, 3.0])
    g_good = (float(w_good @ sol_a), float(w_good @ sol_b))
    g_bad = (float(w_bad @ sol_a), float(w_bad @ sol_b))
    assert g_good == (2.0, 2.0) and g_bad == (2.0, 4.0)
    print(f"  Ex2: w=(1,1,1) invariant (g = {g_good}); w=(1,2,3) not (g = {g_bad})   OK")

    # Exercise 3: sensitivity reading of the weights on the square set
    # {x1 - th = e1, x2 - 2 th = e2}, rho = x1 + x2: d rho / d e_i = -mu_i.
    mu3 = np.array([-1.0, -1.0])  # J_x = I -> mu = -grad_x rho
    theta = 0.7
    de = 1e-6
    rho = lambda e1, e2: (theta + e1) + (2 * theta + e2)  # x solved explicitly
    sens = np.array(
        [
            (rho(de, 0.0) - rho(0.0, 0.0)) / de,
            (rho(0.0, de) - rho(0.0, 0.0)) / de,
        ]
    )
    assert np.allclose(sens, -mu3, atol=1e-9)
    print(f"  Ex3: d rho/d eps = {np.round(sens, 9)} = -mu with mu = (-1,-1)   OK (weights = sensitivities)")

    # Exercise 4: 4x2 Jacobian, gauge dim 2, null basis of J_x^T.
    J_x4 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    assert np.linalg.matrix_rank(J_x4) == 2
    nu_a = np.array([-1.0, -1.0, 1.0, 0.0])
    nu_b = np.array([-2.0, 1.0, 0.0, 1.0])
    assert np.linalg.norm(J_x4.T @ nu_a) == 0.0 and np.linalg.norm(J_x4.T @ nu_b) == 0.0
    assert np.linalg.matrix_rank(np.stack([nu_a, nu_b])) == 2  # independent
    print("  Ex4: d_O=4, d_U=2, r=2; null basis (-1,-1,1,0), (-2,1,0,1) verified   OK")

    # Exercise 5: J_theta^T nu = 0 for example A's null vector.
    J_th = np.array([-1.0, -2.0, 0.0])
    nu = np.array([-2.0, 1.0, 1.0])
    assert J_th @ nu == 0.0
    print("  Ex5: J_theta^T nu = 0   OK (forced by the invariance lemma)")

    # Exercise 6: example A with loss rho = x2.
    J_x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    g_x = np.array([0.0, 1.0])
    for t in np.linspace(-5, 5, 7):
        mu = np.array([-2.0 * t, -1.0 + t, t])
        assert np.linalg.norm(J_x.T @ mu + g_x) < 1e-12  # solves (A)
        assert abs(J_th @ mu - 2.0) < 1e-12  # gradient = 2 for all t
    print("  Ex6: family mu(t) = (-2t, -1+t, t) solves (A); gradient = 2 for all t   OK")

    # Exercise 7: minimal-norm parameter t = -1/6 via orthogonality to nu.
    t_star = -1.0 / 6.0
    mu_star = np.array([-1.0 - 2.0 * t_star, -1.0 + t_star, t_star])
    assert abs(mu_star @ nu) < 1e-15
    ts = np.linspace(-1, 1, 2001)
    norms = [np.linalg.norm([-1 - 2 * t, -1 + t, t]) for t in ts]
    assert abs(ts[int(np.argmin(norms))] - t_star) < 1e-3  # numeric minimum agrees
    print(f"  Ex7: mu(-1/6) . nu = {mu_star @ nu:.1e}; norm minimum at t = {ts[int(np.argmin(norms))]:.4f}   OK")

    # Exercise 8: float32 budget |mu| <= 1e-3 * 3 / eps32 = 2.5e4.
    eps32 = float(np.finfo(np.float32).eps)
    budget = 1e-3 * 3.0 / eps32
    assert abs(budget - 2.5e4) / 2.5e4 < 0.01
    print(f"  Ex8: |mu| budget = 1e-3 * 3 / eps32 = {budget:.3g} (~2.5e4)   OK")

    # Exercise 9: family F3 with mixing constant c.
    g_x = np.array([1.0, 1.0])
    for c in [0.0, 100.0, -7.0, 1e8]:
        J_x2 = np.array([[1.0, 0.0], [c, 1.0]])
        J_th2 = np.array([-1.0, -(c + 2.0)])
        mu = np.linalg.solve(J_x2.T, -g_x)
        assert np.allclose(mu, [c - 1.0, -1.0])
        assert abs(J_th2 @ mu - 3.0) < 1e-6 * max(1.0, abs(c))  # exact-arithmetic gradient
    # float32 fate: c = 100 fine, c = 1e8 total loss (threshold ~ 3/eps32 = 2.5e7)
    g32_small = gradient(np.array([-1.0, -102.0]), np.array([99.0, -1.0]), np.float32)
    g32_huge = gradient(np.array([-1.0, -(1e8 + 2.0)]), np.array([1e8 - 1.0, -1.0]), np.float32)
    thresh = 3.0 / eps32
    print(
        f"  Ex9: mu(c) = (c-1, -1), grad = 3 for all c; float32 grad: c=100 -> {g32_small:.4f}, c=1e8 -> {g32_huge:.4f}"
    )
    print(f"       loss threshold |c| ~ 3/eps32 = {thresh:.3g} (~2.5e7)   OK")
    assert abs(g32_small - 3.0) < 1e-3 and abs(g32_huge - 3.0) > 1.0

    # Exercise 10: eigen-structure of example D.
    assert np.allclose(G_D @ v_D, v_D / 3.0)
    assert np.allclose(l_D @ G_D, 3.0 * l_D)
    assert l_D @ v_D == 0.0 and l_D @ u_D == 8.0
    w = np.array([0.3, -1.7])  # arbitrary: pinned functional triples each step
    assert np.isclose(l_D @ (G_D @ w), 3.0 * (l_D @ w))
    print("  Ex10: G v = v/3, l^T G = 3 l^T, l.v = 0, l.u = 8, l.(Gw) = 3 l.w   OK")

    # Exercise 11: loss step counts n* = log10(1/eps)/log10(9).
    n32 = np.log10(1.0 / eps32) / np.log10(9.0)
    eps64 = float(np.finfo(np.float64).eps)
    n64 = np.log10(1.0 / eps64) / np.log10(9.0)
    assert 7.0 < n32 < 8.0 and 16.0 < n64 < 17.0
    print(f"  Ex11: total loss at n ~ {n32:.1f} steps (float32), {n64:.1f} steps (float64)   OK")

    # Exercise 12: reprojection is exact surgery: a v + b u -> a v.
    a, b = -0.4, 2.9
    w = a * v_D + b * u_D
    assert np.isclose(l_D @ w, 8.0 * b)
    w_after = w - ((l_D @ w) / 8.0) * u_D
    assert np.allclose(w_after, a * v_D)
    w_exact = a * v_D  # b = 0: update is the identity
    assert np.allclose(w_exact - ((l_D @ w_exact) / 8.0) * u_D, w_exact)
    print("  Ex12: l.w = 8b; update maps a v + b u -> a v; identity on exact data   OK")

    # Section 14 dictionary check: study dimension counts for N=7, M=5, K=4.
    N, M, K = 7, 5, 4
    d_U = N * K + M * K + M + 2 * K
    d_O = d_U + K * K
    assert (d_U, d_O, d_O - d_U) == (61, 77, 16)
    print(f"  Sec14: N=7, M=5, K=4 -> d_U = {d_U}, d_O = {d_O}, gauge dim = {d_O - d_U}   OK (matches exp01)")


if __name__ == "__main__":
    example_A()
    example_B()
    example_C()
    example_D()
    exercises()
