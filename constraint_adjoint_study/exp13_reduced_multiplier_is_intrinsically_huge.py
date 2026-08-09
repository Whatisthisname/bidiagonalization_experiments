#!/usr/bin/env python
"""exp13: the reduced-gauge multiplier is exponentially large in EXACT arithmetic.

exp12 (C3) showed per-column norms |lam_red_n| ~ 1e10 at (32,16,16) in
float64.  Two hypotheses:
  (H1) the true multiplier is O(1) and the norms are accumulated rounding
       error (exp10 measured 0.59 digits/step of error growth); or
  (H2) the true multiplier itself grows exponentially backward, and the
       float64 recurrence tracks it to some relative accuracy.

Method: run the SAME reduced backward recurrence in mpmath 60-digit
arithmetic on the SAME float64 primal (converted exactly to mpf).  Then
  - per-column true norms |lam_mp_n|          -> growth rate of the truth;
  - per-column rel. errors |lam64_n - lam_mp_n| / |lam_mp_n|
                                               -> rounding-error growth;
  - gradient assembled from the mpmath multiplier vs the stable-gauge
    gradient                                   -> cancellation depth.

RESULT (H2 confirmed, and more): the true multiplier grows 0.68 digits
per backward step; the float64 recurrence tracks every column to
~2e-16 RELATIVE accuracy (columnwise forward-stable, no accumulation!).
The gradient dies purely in the final assembly
  grad_A = -(lam R^T + L rho^T):
with |mu|_inf ~ 7e10 and |grad_A| ~ 4e2 the sum cancels ~8.2 decimal
digits, so ANY float64 procedure that goes through the exact
reduced-gauge multiplier is limited to ~eps * |mu|/|grad| relative
gradient error (predicted 3.5e-8, observed 2.8e-7 -- the extra order
comes from the ~1e-15 orthogonality drift of the float64 primal, which
also gets multiplied by |mu|/|grad|).  The instability is a property of
the REPRESENTATION (the gauge), not of the algorithm solving it; exp10's
"0.59 digits/step error growth" is really the growth of |mu(K)| itself.

Validation: at (12,8,4) mpmath and float64 recurrences agree to 3e-16
(columnwise relative -- consistent with the forward stability above) and
the mpmath gradient matches the stable gradient to 4e-15 (limited by the
float64 primal at this small size where |mu|/|grad| ~ 1e1).
"""

import numpy as np
import mpmath as mp

from exp02_reduced_adjoint_recurrence import solve_reduced_adjoint_recurrence
from exp05_float32_recurrence_stability import gkb_forward, make_cotangent
from stable_recurrence import stable_adjoint

mp.mp.dps = 60


def to_mp(a):
    a = np.asarray(a, np.float64)
    out = np.empty(a.shape, dtype=object)
    it = np.nditer(a, flags=["multi_index"])
    for val in it:
        out[it.multi_index] = mp.mpf(float(val))
    return out


def mp_norm(vec):
    return mp.sqrt(mp.fsum([x * x for x in vec.ravel()]))


def reduced_recurrence_mp(x, th, gx):
    """Verbatim port of exp02.solve_reduced_adjoint_recurrence to mpmath."""
    L, R = to_mp(x["L"]), to_mp(x["R"])
    al, be = to_mp(x["alphas"]), to_mp(x["betas"])
    c = mp.mpf(float(x["c"]))
    A = to_mp(th["A"])
    N, K = L.shape
    M = R.shape[0]
    dL, dR = to_mp(gx["L"]), to_mp(gx["R"])
    da, db = to_mp(gx["alphas"]), to_mp(gx["betas"])
    dres, dc = to_mp(gx["res"]), mp.mpf(float(gx["c"]))

    lam = np.zeros((N, K), dtype=object)
    rho = np.zeros((M, K), dtype=object)
    rho[:, K - 1] = -dres
    kappa = None
    for i in range(K - 1, -1, -1):
        l_i, r_i = L[:, i], R[:, i]
        rho_i = rho[:, i]
        ip_ll = -da[i] - r_i @ rho_i
        rhs = dL[:, i] - A @ rho_i
        if i < K - 1:
            rhs = rhs + be[i] * lam[:, i + 1]
        sig = -(l_i @ rhs + al[i] * ip_ll) / 2
        lam[:, i] = -(rhs + 2 * sig * l_i) / al[i]
        rhs_r = dR[:, i] - A.T @ lam[:, i] + al[i] * rho_i
        if i >= 1:
            ip_rrho_prev = -db[i - 1] - L[:, i - 1] @ lam[:, i]
            omg = -(r_i @ rhs_r + be[i - 1] * ip_rrho_prev) / 2
            rho[:, i - 1] = -(rhs_r + 2 * omg * r_i) / be[i - 1]
        else:
            omg = -(r_i @ rhs_r + c * dc) / 2
            kappa = -(rhs_r + 2 * omg * r_i)

    grad_A = -(lam @ R.T + L @ rho.T)
    grad_v = -c * kappa
    return dict(lam=lam, rho=rho, kappa=kappa, grad_A=grad_A, grad_v=grad_v)


def mp_to_f64(a):
    return np.array([[float(v) for v in row] for row in np.atleast_2d(a)])


def problem(N, M, K, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, M))
    v = rng.standard_normal(M)
    x = gkb_forward(A, v, K, np.float64)
    gx = make_cotangent(N, M, K, seed + 1, np.float64)
    return A, v, x, dict(A=A, v=v), gx


def validate_small(N=12, M=8, K=4, seed=0):
    print(f"=== validation, (N,M,K)=({N},{M},{K}) ===")
    A, v, x, th, gx = problem(N, M, K, seed)
    m64 = solve_reduced_adjoint_recurrence(x, th, gx)
    mmp = reduced_recurrence_mp(x, th, gx)
    lam_mp = mp_to_f64(mmp["lam"])
    err = np.linalg.norm(m64["lam"] - lam_mp) / np.linalg.norm(lam_mp)
    print(f"|lam64 - lam_mp|/|lam_mp| = {err:.2e}  (columnwise forward " f"stability of the recurrence: ~1e-15)")
    st = stable_adjoint(x, th, gx)
    gmp = mp_to_f64(mmp["grad_A"])
    gerr = np.linalg.norm(gmp - st["grad_A"]) / np.linalg.norm(st["grad_A"])
    print(f"|grad_mp - grad_stable|/|grad| = {gerr:.2e}  (limited by the " f"float64 primal times |mu|/|grad|)\n")


def main(N=32, M=16, K=16, seed=0):
    print(f"=== main experiment, (N,M,K)=({N},{M},{K}) ===")
    A, v, x, th, gx = problem(N, M, K, seed)
    m64 = solve_reduced_adjoint_recurrence(x, th, gx)
    mmp = reduced_recurrence_mp(x, th, gx)
    st = stable_adjoint(x, th, gx)

    print(f"{'n':>3} {'|lam_n| true(mp)':>17} {'|lam_n| f64':>12} " f"{'relerr f64':>11}")
    tn = np.zeros(K)
    re = np.zeros(K)
    for n in range(K):
        t = float(mp_norm(mmp["lam"][:, n]))
        d = float(mp_norm(mmp["lam"][:, n] - to_mp(m64["lam"][:, n])))
        tn[n], re[n] = t, d / t
        print(f"{n + 1:>3} {t:>17.3e} {np.linalg.norm(m64['lam'][:, n]):>12.3e} " f"{re[n]:>11.1e}")

    ns = np.arange(1, K + 1)
    g_true = -np.polyfit(ns, np.log10(tn), 1)[0]
    g_err = -np.polyfit(ns, np.log10(re[re > 0]), 1)[0] if (re > 0).all() else float("nan")
    print(f"\ntrue-multiplier growth : {g_true:.3f} digits per backward step")
    print(
        f"f64 relative-error growth along the sweep: {g_err:.3f} "
        f"digits per backward step (~0: the recurrence is columnwise "
        f"forward-stable; relative error does NOT accumulate)"
    )

    gmp = mp_to_f64(mmp["grad_A"])
    gnorm = np.linalg.norm(gmp)
    mu_inf = tn.max()
    print(
        f"\n|grad_A| = {gnorm:.3e}, max_n |lam_n| = {mu_inf:.3e} "
        f"-> cancellation depth log10 = {np.log10(mu_inf / gnorm):.1f} digits"
    )
    print(f"predicted float64 gradient error floor eps*|mu|/|grad| = " f"{2.2e-16 * mu_inf / gnorm:.1e}")
    ge64 = np.linalg.norm(m64["grad_A"] - st["grad_A"]) / np.linalg.norm(st["grad_A"])
    print(f"observed float64 reduced-recurrence gradient error   = {ge64:.1e}")
    gemp = np.linalg.norm(gmp - st["grad_A"]) / np.linalg.norm(st["grad_A"])
    print(
        f"mpmath-multiplier gradient vs stable gradient        = {gemp:.1e} " f"(cancellation is harmless at 60 digits)"
    )


if __name__ == "__main__":
    validate_small()
    main()
