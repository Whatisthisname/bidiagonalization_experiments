#!/usr/bin/env python
"""exp09: same finalist comparison as exp08 but at PARTIAL depth K = M/2,
where res is genuinely nonzero and the eta-part of the stable gauge matters.
Also repeats the multiplier-mismatch measurements at partial depth.
"""

import numpy as np

from exp08_finalists_and_robustness import check_res_block
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent, cast
from exp06_projector_constraints import con_proj_factory, dense_solve_f32
from exp04_gauge_fixed_dense_solve import coordinate_masks, selection_matrix
from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence
from common import con_full, flat, jacobians
from stable_recurrence import stable_adjoint
import jax.numpy as jnp


def row(n, seed, ensemble):
    N, M = n, n // 2
    K = M // 2  # partial depth
    rng = np.random.default_rng(seed)
    if ensemble == "normal":
        A64 = rng.standard_normal((N, M))
    else:
        i = np.arange(N)[:, None]
        j = np.arange(M)[None, :]
        A64 = 1.0 / (i + j + 1.0) + 1e-3 * rng.standard_normal((N, M))
    v64 = rng.standard_normal(M)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)

    x64 = gkb_forward(A64, v64, K, np.float64)
    th64 = dict(A=A64, v=v64)
    ref = stable_adjoint(x64, th64, gx, reproject=True, dtype=np.float64)
    refA, refv = ref["grad_A"], ref["grad_v"]
    res_norm = float(np.linalg.norm(x64["res"]))

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

    out = {"|res|": res_norm}
    Jx, Jth = jacobians(con_proj_factory("V3"), x32j, th32j, N, M, K)
    g32, _ = dense_solve_f32(Jx, Jth, gx_flat)
    out["V3-dense32"] = err_pair(g32[: N * M].reshape(N, M), g32[N * M :])

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

    st = stable_adjoint(
        x32,
        dict(A=np.asarray(A64, np.float32), v=np.asarray(v64, np.float32)),
        cast(gx, np.float32),
        reproject=True,
        dtype=np.float32,
    )
    out["stable-rec32"] = err_pair(st["grad_A"], st["grad_v"])
    rd = solve_reduced_adjoint_recurrence(
        x32, dict(A=np.asarray(A64, np.float32), v=np.asarray(v64, np.float32)), cast(gx, np.float32), dtype=np.float32
    )
    out["reduced-rec32"] = err_pair(rd["grad_A"], rd["grad_v"])
    return out


if __name__ == "__main__":
    print("=== res-block obstruction at partial depth (K < M) ===")
    check_res_block(12, 8, 4)
    for ens in ["normal", "hilbert"]:
        print(f"\n=== finalists, PARTIAL depth K = M/2, ensemble = {ens} ===")
        keys = ["V3-dense32", "gauge-dense32", "stable-rec32", "reduced-rec32"]
        print(f"{'n':>4} {'K':>3} {'seed':>4} {'|res|':>9}  " + "".join(f"{k:>16}" for k in keys))
        for n in [32, 48, 64, 96]:
            for seed in [0, 1, 2]:
                r = row(n, seed, ens)
                print(f"{n:>4} {n//4:>3} {seed:>4} {r['|res|']:>9.2e}  " + "".join(f"{r[k]:>16.2e}" for k in keys))
