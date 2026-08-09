#!/usr/bin/env python
"""exp12: how are the three gauges' carries related, componentwise?

Gradient-level correspondence between the three multiplier sets
(sign conventions of this repo, verified in exp02/exp03/exp11):

    stable  : grad_A =  Phi R^T + L Xi^T
    reduced : grad_A = -(lam_red R^T + L rho_red^T)
    V3      : grad_A = -sum_n [ (I-P^L_{n-1}) lam_n r_n^T
                                + l_n ((I-P^R_n) rho_n)^T ]

so the natural carry comparisons are  Phi_n vs -lam_n  and  Xi_n vs -rho_n.

Claims tested:
  C1 (forced, sanity).  The OUT-OF-SPAN parts agree across all gauges:
     (I-P^L_K)(Phi_n + lam_n) = 0 and (I-P^R_K)(Xi_n + rho_n) = 0,
     for both V3 and reduced.  (Follows from equality of grad_A alone:
     multiply grad_A by r_n / l_n and project off the computed span.)
  C2 (empirical band structure).  Decompose the carry differences into
     "past band" (l_j, j <= n resp. r_j, j <= n+1), "future band"
     (j > n resp. j > n+1) and out-of-span.  Question: do the gauges
     differ ONLY in the past band (the components the stable selection
     pins and the V3 projectors annihilate), or also in the future band?
  C3 (intrinsic size).  Per-column norms |carry_n| for the three gauges
     at (N,M,K) = (32,16,16).  The reduced-gauge multiplier is expected
     to be EXPONENTIALLY LARGE IN EXACT ARITHMETIC (this is not rounding
     error; cross-checked at mpmath 60-digit precision in exp13).  The
     stable and V3 carries stay O(|cotangent|).
"""

import numpy as np
import jax
import jax.numpy as jnp

from common import flat, flatteners, forward, make_linear_loss, random_problem
from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent
from exp11_v3_recurrence import v3_adjoint_recurrence
from stable_recurrence import stable_adjoint


def band_decomposition(x, D_l, D_r):
    """D_l: (N,K) l-carry difference; D_r: (M,K) r-carry difference.
    Returns per-band max |component| over all columns.
    l bands for column n (1-indexed): past j<=n, future j>n, out-of-span.
    r bands for column n: past j<=n+1, future j>n+1, res-component,
    out-of-span (off span[R, res])."""
    L = np.asarray(x["L"])
    R = np.asarray(x["R"])
    res = np.asarray(x["res"])
    N, K = L.shape
    rn = np.linalg.norm(res)
    resu = res / rn if rn > 1e-12 else None

    CL = L.T @ D_l  # (K, K): CL[j-1, n-1] = l_j^T D_l[:, n]
    CR = R.T @ D_r
    out = dict.fromkeys(["l_past", "l_future", "l_off", "r_past", "r_future", "r_res", "r_off"], 0.0)
    for n in range(K):  # n is 0-indexed column
        out["l_past"] = max(out["l_past"], np.abs(CL[: n + 1, n]).max())
        if n + 1 < K:
            out["l_future"] = max(out["l_future"], np.abs(CL[n + 1 :, n]).max())
        off = D_l[:, n] - L @ CL[:, n]
        out["l_off"] = max(out["l_off"], np.linalg.norm(off))

        hi = min(n + 2, K)
        out["r_past"] = max(out["r_past"], np.abs(CR[:hi, n]).max())
        if hi < K:
            out["r_future"] = max(out["r_future"], np.abs(CR[hi:, n]).max())
        roff = D_r[:, n] - R @ CR[:, n]
        if resu is not None:
            rc = resu @ D_r[:, n]
            out["r_res"] = max(out["r_res"], abs(rc))
            roff = roff - resu * rc
        out["r_off"] = max(out["r_off"], np.linalg.norm(roff))
    return out


def run_bands(N, M, K, seed=0):
    print(f"\n=== band structure, (N,M,K)=({N},{M},{K}), seed={seed} ===")
    A, v = random_problem(N, M, K, seed=seed)
    x = forward(A, v, K)
    th = dict(A=A, v=v)
    loss, _ = make_linear_loss(N, M, K)
    unflat_x, _ = flatteners(N, M, K)
    gx = unflat_x(jax.grad(lambda xf: loss(unflat_x(xf)))(flat(x)))

    st = stable_adjoint(x, th, gx, reproject=True)
    v3 = v3_adjoint_recurrence(x, th, gx)
    rd = solve_reduced_adjoint_recurrence(x, th, gx)

    scale = max(np.abs(st["Phi"]).max(), np.abs(st["Xi"]).max())
    print(f"    (carry scale for reference: max|Phi|,|Xi| ~ {scale:.2e})")
    hdr = (
        f"{'pair':>16} {'l_past':>9} {'l_future':>9} {'l_off':>9} "
        f"{'r_past':>9} {'r_future':>9} {'r_res':>9} {'r_off':>9}"
    )
    print(hdr)
    for name, m in [("stable-vs-V3", v3), ("stable-vs-red", rd)]:
        D_l = st["Phi"] + m["lam"]  # Phi_n - (-lam_n)
        D_r = st["Xi"] + m["rho"]
        b = band_decomposition(x, D_l, D_r)
        print(
            f"{name:>16} {b['l_past']:>9.1e} {b['l_future']:>9.1e} "
            f"{b['l_off']:>9.1e} {b['r_past']:>9.1e} {b['r_future']:>9.1e} "
            f"{b['r_res']:>9.1e} {b['r_off']:>9.1e}"
        )


def run_norm_growth(N, M, K, seed=0):
    print(f"\n=== per-column carry norms, (N,M,K)=({N},{M},{K}) ===")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, M))
    v = rng.standard_normal(M)
    x = gkb_forward(A, v, K, np.float64)
    th = dict(A=A, v=v)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)

    st = stable_adjoint(x, th, gx, reproject=True)
    v3 = v3_adjoint_recurrence(x, th, gx)
    rd = solve_reduced_adjoint_recurrence(x, th, gx)

    print(f"{'n':>3} {'|Phi_n| stable':>15} {'|lam_n| V3':>12} " f"{'|lam_n| red':>13}")
    nr = np.zeros(K)
    for n in range(K):
        nr[n] = np.linalg.norm(rd["lam"][:, n])
        print(
            f"{n + 1:>3} {np.linalg.norm(st['Phi'][:, n]):>15.3e} "
            f"{np.linalg.norm(v3['lam'][:, n]):>12.3e} {nr[n]:>13.3e}"
        )
    # least-squares slope of log10|lam_red_n| vs n (backward growth)
    ns = np.arange(1, K + 1)
    slope = np.polyfit(ns, np.log10(nr), 1)[0]
    print(f"reduced-gauge growth: {-slope:.3f} decimal digits per BACKWARD " f"step (compare exp10 error-growth 0.59)")
    print(f"gradient norms for scale: |grad_A| = " f"{np.linalg.norm(st['grad_A']):.3e} (all three gauges agree)")


if __name__ == "__main__":
    run_bands(12, 8, 4)
    run_bands(9, 6, 3)
    run_bands(16, 12, 6, seed=3)
    run_norm_growth(32, 16, 16)
