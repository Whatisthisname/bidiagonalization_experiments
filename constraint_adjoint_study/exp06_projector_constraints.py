#!/usr/bin/env python
"""exp06: square projector-form (Gram-Schmidt) constraint sets.

These are the natural candidates for the user's conjecture: square constraint
sets, equivalent to bidiagonalization on-manifold, whose Jacobians contain
projector structure -- maybe their (unique) adjoint IS the stable one?

Variants (all square, d_U rows, all vanish at the exact solution):
  V1 gs-full : alpha_n l_n = (I - L_<n L_<n^T) A r_n
               beta_n r_{n+1} = (I - R_<=n R_<=n^T) A^T l_n   (res at n=K)
  V2 gs-half : same l-side;  r-side keeps explicit -alpha_n r_n and projects
               only against R_<n.
  V3 gs-resid: projectors applied to the CLASSIC residuals
               (I - L_<n L_<n^T)(A r_n - beta_{n-1} l_{n-1}) etc.
  (con_red = no projectors at all, from exp02, for reference.)

Questions:
  Q1  Do their unique multipliers coincide with the stable-gauge multipliers
      (Phi, Xi blocks)?  [prediction from NOTES_04: NO -- res-block argument]
  Q2  Do their multipliers satisfy the stable selection identities?
  Q3  Are their dense float32 adjoint solves more accurate than con_red's?
      How do they compare to the gauge-fixed square system of exp04?
      What are the condition numbers of the system matrices?
"""

import numpy as np
import jax
import jax.numpy as jnp

from common import (
    autodiff_reference_grad,
    con_full,
    con_red,
    dims,
    flat,
    flatteners,
    forward,
    jacobians,
    make_linear_loss,
    random_problem,
    rel_err,
    B_of,
)
from exp03_stable_multipliers import flatten_mu_full, selection_residuals
from exp04_gauge_fixed_dense_solve import coordinate_masks, selection_matrix
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent, cast
from stable_recurrence import stable_adjoint


def con_proj_factory(variant):
    def con(x, th):
        L, R, al, be, res, c = (x["L"], x["R"], x["alphas"], x["betas"], x["res"], x["c"])
        A, v = th["A"], th["v"]
        N, K = L.shape
        UT1 = jnp.triu(jnp.ones((K, K)), k=1)  # j < n   (row j, col n)
        UT0 = jnp.triu(jnp.ones((K, K)), k=0)  # j <= n
        init = R[:, 0] - c * v
        Lprev = jnp.concatenate([jnp.zeros((N, 1)), L[:, :-1]], axis=1)
        beprev = jnp.concatenate([jnp.zeros((1,)), be])
        Rnext = jnp.concatenate([R[:, 1:], res[:, None]], axis=1)
        benext = jnp.concatenate([be, jnp.ones((1,))])
        AR = A @ R
        ATL = A.T @ L
        if variant == "V1":
            recl = L * al[None, :] - AR + L @ (UT1 * (L.T @ AR))
            recr = Rnext * benext[None, :] - ATL + R @ (UT0 * (R.T @ ATL))
        elif variant == "V2":
            recl = L * al[None, :] - AR + L @ (UT1 * (L.T @ AR))
            recr = Rnext * benext[None, :] - ATL + R * al[None, :] + R @ (UT1 * (R.T @ ATL))
        elif variant == "V3":
            tl = AR - Lprev * beprev[None, :]
            recl = L * al[None, :] - tl + L @ (UT1 * (L.T @ tl))
            tr = ATL - R * al[None, :]
            recr = Rnext * benext[None, :] - tr + R @ (UT0 * (R.T @ tr))
        else:
            raise ValueError(variant)
        nrmL = jnp.sum(L * L, axis=0) - 1.0
        nrmR = jnp.sum(R * R, axis=0) - 1.0
        return jnp.concatenate([init, recl.ravel(), recr.ravel(), nrmL, nrmR])

    return con


def dense_solve_f32(Jx64, Jth64, gx64):
    Jx = np.asarray(Jx64, np.float32)
    Jth = np.asarray(Jth64, np.float32)
    gxf = np.asarray(gx64, np.float32)
    if Jx.shape[0] == Jx.shape[1]:
        mu = np.linalg.solve(Jx.T, -gxf)
    else:
        mu, *_ = np.linalg.lstsq(Jx.T, -gxf, rcond=None)
    return Jth.T @ mu, mu


def multiplier_comparison(N, M, K, seed=0):
    print(f"\n--- multiplier comparison, float64, (N,M,K)=({N},{M},{K}) ---")
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    loss, _ = make_linear_loss(N, M, K)
    unflat_x, _ = flatteners(N, M, K)
    gx_flat = np.asarray(jax.grad(lambda xf: loss(unflat_x(xf)))(flat(x)))
    gx = unflat_x(jnp.asarray(gx_flat))
    m_st = stable_adjoint(x, th, gx, reproject=True)
    gref = np.asarray(flat(autodiff_reference_grad(A, v, K, loss)))

    for variant in ["V1", "V2", "V3"]:
        con = con_proj_factory(variant)
        resid = np.linalg.norm(np.asarray(con(x, th)))
        Jx, Jth = jacobians(con, x, th, N, M, K)
        s = np.linalg.svd(Jx, compute_uv=False)
        mu = np.linalg.solve(Jx.T, -gx_flat)
        g = Jth.T @ mu
        # unpack multiplier blocks: rows [init(M), recl(NK), recr(MK), nrmL, nrmR]
        o = M
        lam = mu[o : o + N * K].reshape(N, K)
        o += N * K
        rho = mu[o : o + M * K].reshape(M, K)
        o += M * K
        # map to con_full orientation: recl = -(recL col) + proj -> Phi ~ -lam
        dPhi = rel_err(-lam, m_st["Phi"])
        dXi = rel_err(-rho, m_st["Xi"])
        m_emb = dict(Phi=-lam, Xi=-rho, kappa=mu[:M], Sigma=np.zeros((K, K)), Omega=np.zeros((K, K)), eta=np.zeros(K))
        rl, rr = selection_residuals(m_emb, x, gx)
        print(
            f"{variant}: |con(x*)|={resid:.1e}  cond(Jx)={s[0]/s[-1]:.1e}  "
            f"grad err={rel_err(g, gref):.1e}  "
            f"|Phi_proj-Phi_st|/|Phi_st|={dPhi:.2e}  "
            f"|Xi_proj-Xi_st|/|Xi_st|={dXi:.2e}  "
            f"selection viol: L {np.abs(rl).max():.1e} R {np.abs(rr).max():.1e}"
        )


def float32_dense_stability(sizes, seed=0):
    print("\n--- dense float32 adjoint solves at the float32 primal ---")
    print("error = rel. gradient error vs float64 reference; " "cond = cond_2 of the solved matrix")
    hdr = f"{'n':>4} {'K':>3} " + "".join(
        f"{name:>24}" for name in ["con_red", "V1 gs-full", "V2 gs-half", "V3 gs-resid", "gauge-fixed square"]
    )
    print(hdr)
    for n in sizes:
        N, M = n, n // 2
        K = M
        rng = np.random.default_rng(seed)
        A64 = rng.standard_normal((N, M))
        v64 = rng.standard_normal(M)
        gx = make_cotangent(N, M, K, seed + 1, np.float64)

        x64 = gkb_forward(A64, v64, K, np.float64)
        th64 = dict(A=A64, v=v64)
        ref = stable_adjoint(x64, th64, gx, reproject=True, dtype=np.float64)
        refA, refv = ref["grad_A"], ref["grad_v"]

        x32 = gkb_forward(A64, v64, K, np.float32)
        x32j = {k: jnp.asarray(np.asarray(vv, np.float64)) for k, vv in x32.items()}
        th32j = dict(A=jnp.asarray(A64), v=jnp.asarray(v64))  # exact params
        gx_flat = np.asarray(flat(cast(gx, np.float64)))

        cells = []
        for name, con in [
            ("con_red", con_red),
            ("V1", con_proj_factory("V1")),
            ("V2", con_proj_factory("V2")),
            ("V3", con_proj_factory("V3")),
        ]:
            Jx, Jth = jacobians(con, x32j, th32j, N, M, K)
            g32, _ = dense_solve_f32(Jx, Jth, gx_flat)
            gA = g32[: N * M].reshape(N, M)
            gv = g32[N * M :]
            err = np.sqrt(
                (np.linalg.norm(gA - refA) ** 2 + np.linalg.norm(gv - refv) ** 2)
                / (np.linalg.norm(refA) ** 2 + np.linalg.norm(refv) ** 2)
            )
            s = np.linalg.svd(np.asarray(Jx, np.float32), compute_uv=False)
            cells.append(f"{err:>11.1e}/{s[0]/s[-1]:>9.1e}")

        # gauge-fixed square system, float32
        Jx, Jth = jacobians(con_full, x32j, th32j, N, M, K)
        masks = coordinate_masks(N, M, K)
        keep = ~(masks["alphas"] | masks["betas"])
        gxd = {k: np.asarray(vv) for k, vv in cast(gx, np.float64).items()}
        Ssel, tsel = selection_matrix(x32, gxd, N, M, K)
        Big = np.vstack([np.asarray(Jx.T)[keep, :], Ssel]).astype(np.float32)
        rhs = np.concatenate([-gx_flat[keep], tsel]).astype(np.float32)
        mu = np.linalg.solve(Big, rhs)
        g32 = np.asarray(Jth, np.float32).T @ mu
        gA = g32[: N * M].reshape(N, M)
        gv = g32[N * M :]
        err = np.sqrt(
            (np.linalg.norm(gA - refA) ** 2 + np.linalg.norm(gv - refv) ** 2)
            / (np.linalg.norm(refA) ** 2 + np.linalg.norm(refv) ** 2)
        )
        s = np.linalg.svd(Big, compute_uv=False)
        cells.append(f"{err:>11.1e}/{s[0]/s[-1]:>9.1e}")
        print(f"{n:>4} {K:>3} " + "".join(f"{c:>24}" for c in cells))


if __name__ == "__main__":
    multiplier_comparison(7, 5, 4)
    multiplier_comparison(6, 6, 3)
    float32_dense_stability([16, 32, 48])
