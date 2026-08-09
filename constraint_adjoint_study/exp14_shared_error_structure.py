#!/usr/bin/env python
"""exp14: do the stable and V3 recurrences commit the SAME float32 errors?

The carry-correspondence theorem (NOTES_06 sec. 1) says the two recurrences
propagate identical out-of-past-band carries in exact arithmetic. If that
correspondence survives floating point to first order, then their float32
gradient ERRORS should be dominated by a common component (the shared
primal imperfection pushed through the shared propagation), i.e.

    mutual := |g_v3 - g_stable| / |g_ref|   <<   err(v3) ~ err(stable).

If instead mutual ~ err, the two methods make independent rounding errors
of the same size and the exp11 parity is just "both are stable".

Two primal modes, same cotangent and reference (float64 stable recurrence
on the float64 primal):
  native32: primal computed by GKB in float32 (realistic; primal error
            ~1e-4..1e-3 dominates) -- expect LARGE common component;
  cast64:   float64 primal rounded to float32 (primal error ~6e-8, nearly
            on-manifold) -- isolates the recurrences' own arithmetic.
"""

import numpy as np

from exp05_float32_recurrence_stability import gkb_forward, make_cotangent, cast
from exp11_v3_recurrence import v3_adjoint_recurrence
from stable_recurrence import stable_adjoint


def gvec(m):
    return np.concatenate([np.asarray(m["grad_A"], np.float64).ravel(), np.asarray(m["grad_v"], np.float64).ravel()])


def one(ens, n, depth, seed, mode):
    N, M = n, n // 2
    K = M if depth == "full" else M // 2
    rng = np.random.default_rng(seed)
    if ens == "normal":
        A = rng.standard_normal((N, M))
    else:
        ii = np.arange(N)[:, None]
        jj = np.arange(M)[None, :]
        A = 1.0 / (ii + jj + 1.0) + 1e-3 * rng.standard_normal((N, M))
    v = rng.standard_normal(M)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)

    x64 = gkb_forward(A, v, K, np.float64)
    th = dict(A=A, v=v)
    ref = gvec(stable_adjoint(x64, th, gx, reproject=True, dtype=np.float64))

    if mode == "native32":
        x32 = gkb_forward(A, v, K, np.float32)
    else:  # cast64
        x32 = {k: np.asarray(np.asarray(val, np.float64), np.float32) for k, val in x64.items()}
    th32 = dict(A=np.asarray(A, np.float32), v=np.asarray(v, np.float32))
    gx32 = cast(gx, np.float32)

    g_v3 = gvec(v3_adjoint_recurrence(x32, th32, gx32, dtype=np.float32))
    g_st = gvec(stable_adjoint(x32, th32, gx32, reproject=True, dtype=np.float32))
    rn = np.linalg.norm(ref)
    return (np.linalg.norm(g_v3 - ref) / rn, np.linalg.norm(g_st - ref) / rn, np.linalg.norm(g_v3 - g_st) / rn)


if __name__ == "__main__":
    for mode in ["native32", "cast64"]:
        print(f"\n=== primal mode: {mode} ===")
        print(
            f"{'ens':>8} {'n':>4} {'depth':>5} {'seed':>4} "
            f"{'err(v3)':>10} {'err(st)':>10} {'mutual':>10} "
            f"{'mutual/err':>10}"
        )
        for ens in ["normal", "hilbert"]:
            for n in [32, 64, 96]:
                for depth in ["full", "half"]:
                    for seed in [0, 1, 2]:
                        ev, es, mu = one(ens, n, depth, seed, mode)
                        print(
                            f"{ens:>8} {n:>4} {depth:>5} {seed:>4} "
                            f"{ev:>10.2e} {es:>10.2e} {mu:>10.2e} "
                            f"{mu / max(ev, es):>10.2f}"
                        )
