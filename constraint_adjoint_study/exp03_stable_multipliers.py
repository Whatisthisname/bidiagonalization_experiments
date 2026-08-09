#!/usr/bin/env python
"""exp03: the stable recurrence's multipliers solve the ORIGINAL-constraint
stationarity system and satisfy K^2 extra selection identities.

Claims tested (see NOTES_03):
  C0  The transparent reimplementation (stable_recurrence.py) reproduces the
      library's custom VJP and plain autodiff (float64, with and without
      mid-flight reprojection).
  C1  Its multipliers (Phi, Xi, Sigma, Omega, kappa, eta) satisfy
      J_full^T mu + grad_x rho = 0   (stationarity of con_full).
  C2  Selection identities:  for m <= n:   l_m^T phi_n = db_{n-1} [m = n-1]
                             for m <= n+1: r_m^T xi_n  = da_n    [m = n]
  C3  The reduced-gauge multiplier (exp02) satisfies stationarity but
      VIOLATES the selection identities; the difference of the two multiplier
      vectors lies in null(J_full^T); the two gradients agree anyway.
"""

import numpy as np
import jax

from common import (
    autodiff_reference_grad,
    con_full,
    dims,
    flat,
    flatteners,
    forward,
    jacobians,
    make_linear_loss,
    random_problem,
    rel_err,
)
from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence
from stable_recurrence import stable_adjoint

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo_bidiag import bidiagonalize  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def flatten_mu_full(m, N, M, K):
    """con_full row order: recR (M x K), recL (N x K), init (M), orthL tril,
    orthR tril, orthres (K)."""
    il, jl = np.tril_indices(K)
    return np.concatenate(
        [
            m["Xi"].ravel(),
            m["Phi"].ravel(),
            m["kappa"],
            m["Sigma"][il, jl],
            m["Omega"][il, jl],
            m["eta"],
        ]
    )


def selection_residuals(m, x, gx):
    L, R = np.asarray(x["L"]), np.asarray(x["R"])
    K = L.shape[1]
    da, db = np.asarray(gx["alphas"]), np.asarray(gx["betas"])
    LtPhi = L.T @ m["Phi"]
    RtXi = R.T @ m["Xi"]
    res_l, res_r = [], []
    for n in range(K):
        for mm in range(n + 1):
            tgt = db[n - 1] if (n >= 1 and mm == n - 1) else 0.0
            res_l.append(LtPhi[mm, n] - tgt)
        for mm in range(min(n + 1, K - 1) + 1):
            tgt = da[n] if mm == n else 0.0
            res_r.append(RtXi[mm, n] - tgt)
    return np.array(res_l), np.array(res_r)


def run(N, M, K, seed=0):
    print(f"\n=== (N, M, K) = ({N}, {M}, {K}) ===")
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    loss, _ = make_linear_loss(N, M, K)
    unflat_x, _ = flatteners(N, M, K)
    gx_flat = jax.grad(lambda xf: loss(unflat_x(xf)))(flat(x))
    gx = unflat_x(gx_flat)
    gref = autodiff_reference_grad(A, v, K, loss)

    # C0: reimplementation sanity, incl. against the library custom vjp
    bd = bidiagonalize(num_matvecs=K, custom_vjp=True, reorthogonalize=True)

    def obj(A_, v_):
        out = bd(lambda vec, p: p @ vec, v_, A_)
        xd = dict(L=out.ls, R=out.rs, alphas=out.as_, betas=out.bs, res=out.res, c=jnp.asarray(out.c))
        return loss(xd)

    gA_lib, gv_lib = jax.grad(obj, argnums=(0, 1))(A, v)
    for rp in (True, False):
        m = stable_adjoint(x, th, gx, reproject=rp)
        print(
            f"C0  reproject={rp!s:5}: grad_A vs autodiff {rel_err(m['grad_A'], gref['A']):.2e}, "
            f"vs library-vjp {rel_err(m['grad_A'], np.asarray(gA_lib)):.2e}; "
            f"grad_v vs autodiff {rel_err(m['grad_v'], gref['v']):.2e}"
        )

    m = stable_adjoint(x, th, gx, reproject=True)

    # C1: stationarity of con_full
    Jx, Jth = jacobians(con_full, x, th, N, M, K)
    mu_stable = flatten_mu_full(m, N, M, K)
    stat = Jx.T @ mu_stable + np.asarray(gx_flat)
    print(
        f"C1  |J_full^T mu_stable + grad_x| / |grad_x| = "
        f"{np.linalg.norm(stat)/np.linalg.norm(np.asarray(gx_flat)):.2e}"
    )

    # C2: selection identities
    rl, rr = selection_residuals(m, x, gx)
    print(
        f"C2  selection residuals: L-side max {np.abs(rl).max():.2e} "
        f"({rl.size} identities), R-side max {np.abs(rr).max():.2e} "
        f"({rr.size} identities); total {rl.size + rr.size} "
        f"(= K^2 + 2K - 1 = {K*K + 2*K - 1})"
    )

    # C3: reduced-gauge comparison
    mred = solve_reduced_adjoint_recurrence(x, th, gx)
    m_embed = dict(
        Phi=-mred["lam"],
        Xi=-mred["rho"],
        kappa=mred["kappa"],
        Sigma=np.diag(mred["sig"]),
        Omega=np.diag(mred["omg"]),
        eta=np.zeros(K),
    )
    mu_red = flatten_mu_full(m_embed, N, M, K)
    stat_red = Jx.T @ mu_red + np.asarray(gx_flat)
    rl2, rr2 = selection_residuals(m_embed, x, gx)
    diff = mu_stable - mu_red
    print(
        f"C3  reduced gauge: stationarity residual {np.linalg.norm(stat_red)/np.linalg.norm(np.asarray(gx_flat)):.2e}, "
        f"selection violation L {np.abs(rl2).max():.2e} R {np.abs(rr2).max():.2e}"
    )
    print(
        f"C3  |mu_stable - mu_red| = {np.linalg.norm(diff):.3f} "
        f"(same gradient though: {rel_err(Jth.T @ mu_red, Jth.T @ mu_stable):.2e}); "
        f"|J_full^T (mu_s - mu_r)| = {np.linalg.norm(Jx.T @ diff):.2e} -> null vector"
    )


if __name__ == "__main__":
    run(7, 5, 4)
    run(6, 6, 3)
    run(5, 8, 4)
