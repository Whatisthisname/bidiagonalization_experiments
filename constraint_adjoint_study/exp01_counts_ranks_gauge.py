#!/usr/bin/env python
"""exp01: verify counting, ranks, gauge freedom, gradient invariance.

Claims tested (see NOTES_01):
  C1  d_O - d_U = K^2 by construction; constraint residuals ~ 0 at the
      computed forward solution, for BOTH constraint sets.
  C2  rank(J_red) = d_U (invertible);  rank(J_full) = d_U;
      dim null(J_full^T) = K^2.
  C3  The unique reduced multiplier and any least-squares full multiplier
      give the same gradient as plain autodiff.
  C4  Adding null(J_full^T) elements to the full multiplier does not change
      the gradient.
"""

import numpy as np
import jax
import jax.numpy as jnp

from common import (
    autodiff_reference_grad,
    con_full,
    con_red,
    dense_adjoint_gradient,
    dims,
    flat,
    flatteners,
    forward,
    jacobians,
    make_linear_loss,
    random_problem,
    rel_err,
)


def run(N, M, K, seed=0):
    print(f"\n=== (N, M, K) = ({N}, {M}, {K}) ===")
    d_U, d_O = dims(N, M, K)
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)

    r_full = con_full(x, th)
    r_red = con_red(x, th)
    print(f"C1  d_U = {d_U}, d_O = {d_O}, d_O - d_U = {d_O - d_U} (= K^2 = {K*K})")
    print(f"C1  rows(con_full) = {r_full.shape[0]}, rows(con_red) = {r_red.shape[0]}")
    print(f"C1  |con_full(x*)| = {np.linalg.norm(r_full):.2e}, " f"|con_red(x*)| = {np.linalg.norm(r_red):.2e}")

    Jx_full, Jth_full = jacobians(con_full, x, th, N, M, K)
    Jx_red, Jth_red = jacobians(con_red, x, th, N, M, K)
    s_full = np.linalg.svd(Jx_full, compute_uv=False)
    s_red = np.linalg.svd(Jx_red, compute_uv=False)
    tol_f = s_full[0] * max(Jx_full.shape) * np.finfo(np.float64).eps
    tol_r = s_red[0] * max(Jx_red.shape) * np.finfo(np.float64).eps
    rank_full = int((s_full > tol_f).sum())
    rank_red = int((s_red > tol_r).sum())
    print(
        f"C2  rank(J_full [{Jx_full.shape}]) = {rank_full} (expect {d_U}); "
        f"smallest sv above tol: {s_full[rank_full-1]:.2e}"
    )
    print(f"C2  rank(J_red  [{Jx_red.shape}]) = {rank_red} (expect {d_U}); " f"cond(J_red) = {s_red[0]/s_red[-1]:.2e}")
    null_dim = Jx_full.shape[0] - rank_full
    print(f"C2  dim null(J_full^T) = {null_dim} (expect K^2 = {K*K})")

    # gradient comparisons
    loss, _ = make_linear_loss(N, M, K)
    grad_ref = autodiff_reference_grad(A, v, K, loss)
    gx = jax.grad(lambda xf: loss(flatteners(N, M, K)[0](xf)))(flat(x))
    xd_grad = flatteners(N, M, K)[0](gx)  # dict form of grad_x rho

    g_red, mu_red, _, _ = dense_adjoint_gradient(con_red, x, th, xd_grad, N, M, K)
    g_full, mu_full, _, _ = dense_adjoint_gradient(con_full, x, th, xd_grad, N, M, K)
    gref_flat = np.asarray(flat(grad_ref))
    print(f"C3  |grad(con_red)  - autodiff| / |autodiff| = {rel_err(g_red, gref_flat):.2e}")
    print(f"C3  |grad(con_full) - autodiff| / |autodiff| = {rel_err(g_full, gref_flat):.2e}")

    # C4: gradient invariance along null(J_full^T)
    U, S, Vt = np.linalg.svd(Jx_full.T)  # Jx_full.T is d_U x d_O
    null_basis = Vt[rank_full:, :]  # rows span null(J_full^T)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(5):
        z = rng.standard_normal(null_basis.shape[0])
        mu_pert = mu_full + 10.0 * (null_basis.T @ z)
        # check it still solves the adjoint system
        res_stat = np.linalg.norm(Jx_full.T @ mu_pert + np.asarray(flat(xd_grad)))
        g_pert = Jth_full.T @ mu_pert
        worst = max(worst, rel_err(g_pert, gref_flat))
        assert res_stat < 1e-8, res_stat
    print(f"C4  worst gradient deviation over 5 random gauge moves (scale 10): {worst:.2e}")
    print(
        f"C4  |mu_full(lstsq) - mu_red embedded|: multipliers themselves are gauge-dependent -> "
        f"norm(mu_full) = {np.linalg.norm(mu_full):.3f}, norm(mu_red) = {np.linalg.norm(mu_red):.3f}"
    )


if __name__ == "__main__":
    run(7, 5, 4)
    run(6, 6, 3)
    run(5, 8, 4)  # wide
