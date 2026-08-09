#!/usr/bin/env python
"""exp08: finalists at larger size + robustness across seeds/ensembles.

(a) res-block obstruction, verified exactly:  for ANY square set in which res
    appears only in the last recurrence row (con_red, V1, V2, V3), the res
    stationarity forces  -rho~_K = dres,  whereas the stable gauge has
    xi_K = dres + R eta.  So the difference must equal R eta EXACTLY.
(b) float32 dense-solve comparison V3 vs gauge-fixed square vs the paper's
    stable recurrence, up to n = 64, multiple seeds.
(c) same for a Hilbert-type ill-conditioned ensemble.
"""

import numpy as np
import jax

from common import con_red, flat, flatteners, forward, jacobians, make_linear_loss, random_problem, rel_err
from exp03_stable_multipliers import flatten_mu_full
from exp04_gauge_fixed_dense_solve import coordinate_masks, selection_matrix
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent, cast
from exp06_projector_constraints import con_proj_factory, dense_solve_f32
from common import con_full
import jax.numpy as jnp
from stable_recurrence import stable_adjoint


def check_res_block(N, M, K, seed=0):
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    loss, _ = make_linear_loss(N, M, K)
    unflat_x, _ = flatteners(N, M, K)
    gx_flat = np.asarray(jax.grad(lambda xf: loss(unflat_x(xf)))(flat(x)))
    gx = unflat_x(jnp.asarray(gx_flat))
    m_st = stable_adjoint(x, th, gx, reproject=True)
    R = np.asarray(x["R"])
    Reta = R @ m_st["eta"]
    print(f"(N,M,K)=({N},{M},{K}):")
    for name, con in [
        ("con_red", con_red),
        ("V1", con_proj_factory("V1")),
        ("V2", con_proj_factory("V2")),
        ("V3", con_proj_factory("V3")),
    ]:
        Jx, _ = jacobians(con, x, th, N, M, K)
        mu = np.linalg.solve(np.asarray(Jx).T, -gx_flat)
        o = M + N * K
        rho = mu[o : o + M * K].reshape(M, K)
        gap = (-rho[:, K - 1]) - m_st["Xi"][:, K - 1]
        print(
            f"  {name:8s}: |(-rho_K) - xi_K^stable + R eta| = "
            f"{np.linalg.norm(gap + Reta):.2e}   (|R eta| = {np.linalg.norm(Reta):.2e})"
        )


def finalist_row(n, seed, ensemble="normal"):
    N, M = n, n // 2
    K = M
    rng = np.random.default_rng(seed)
    if ensemble == "normal":
        A64 = rng.standard_normal((N, M))
    elif ensemble == "hilbert":
        # Hilbert-type: H_ij = 1/(i+j+1), rectangular slice, mildly scaled
        i = np.arange(N)[:, None]
        j = np.arange(M)[None, :]
        A64 = 1.0 / (i + j + 1.0) + 1e-3 * rng.standard_normal((N, M))
    v64 = rng.standard_normal(M)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)

    x64 = gkb_forward(A64, v64, K, np.float64)
    th64 = dict(A=A64, v=v64)
    ref = stable_adjoint(x64, th64, gx, reproject=True, dtype=np.float64)
    refA, refv = ref["grad_A"], ref["grad_v"]

    def err_pair(gA, gv):
        return float(
            np.sqrt(
                (
                    np.linalg.norm(np.asarray(gA, np.float64) - refA) ** 2
                    + np.linalg.norm(np.asarray(gv, np.float64) - refv) ** 2
                )
                / (np.linalg.norm(refA) ** 2 + np.linalg.norm(refv) ** 2)
            )
        )

    x32 = gkb_forward(A64, v64, K, np.float32)
    x32j = {k: jnp.asarray(np.asarray(vv, np.float64)) for k, vv in x32.items()}
    th32j = dict(A=jnp.asarray(A64), v=jnp.asarray(v64))
    gx_flat = np.asarray(flat(cast(gx, np.float64)))

    out = {}
    # V3 dense float32
    Jx, Jth = jacobians(con_proj_factory("V3"), x32j, th32j, N, M, K)
    g32, _ = dense_solve_f32(Jx, Jth, gx_flat)
    out["V3-dense32"] = err_pair(g32[: N * M].reshape(N, M), g32[N * M :])
    # gauge-fixed square dense float32
    Jx, Jth = jacobians(con_full, x32j, th32j, N, M, K)
    masks = coordinate_masks(N, M, K)
    keep = ~(masks["alphas"] | masks["betas"])
    gxd = {k: np.asarray(vv) for k, vv in cast(gx, np.float64).items()}
    Ssel, tsel = selection_matrix(x32, gxd, N, M, K)
    Big = np.vstack([np.asarray(Jx.T)[keep, :], Ssel]).astype(np.float32)
    rhs = np.concatenate([-gx_flat[keep], tsel]).astype(np.float32)
    mu = np.linalg.solve(Big, rhs)
    g32 = np.asarray(Jth, np.float32).T @ mu
    out["gauge-dense32"] = err_pair(g32[: N * M].reshape(N, M), g32[N * M :])
    # paper's stable recurrence float32
    st = stable_adjoint(
        x32,
        dict(A=np.asarray(A64, np.float32), v=np.asarray(v64, np.float32)),
        cast(gx, np.float32),
        reproject=True,
        dtype=np.float32,
    )
    out["stable-rec32"] = err_pair(st["grad_A"], st["grad_v"])
    # reduced recurrence float32 for contrast
    from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence

    rd = solve_reduced_adjoint_recurrence(
        x32, dict(A=np.asarray(A64, np.float32), v=np.asarray(v64, np.float32)), cast(gx, np.float32), dtype=np.float32
    )
    out["reduced-rec32"] = err_pair(rd["grad_A"], rd["grad_v"])
    return out


if __name__ == "__main__":
    print("=== (a) res-block obstruction (float64, exact identity check) ===")
    check_res_block(7, 5, 4)
    check_res_block(6, 6, 3)

    for ens in ["normal", "hilbert"]:
        print(f"\n=== (b/c) finalists, ensemble = {ens} ===")
        print(
            f"{'n':>4} {'seed':>4}  "
            + "".join(f"{k:>16}" for k in ["V3-dense32", "gauge-dense32", "stable-rec32", "reduced-rec32"])
        )
        for n in [32, 48, 64]:
            for seed in [0, 1, 2]:
                r = finalist_row(n, seed, ens)
                print(
                    f"{n:>4} {seed:>4}  "
                    + "".join(
                        f"{r[k]:>16.2e}" for k in ["V3-dense32", "gauge-dense32", "stable-rec32", "reduced-rec32"]
                    )
                )
